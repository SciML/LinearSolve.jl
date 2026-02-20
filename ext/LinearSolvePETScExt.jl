module LinearSolvePETScExt

using PETSc
import PETSc: MPI, LibPETSc
using SparseArrays: SparseMatrixCSC, nzrange, sparse
using LinearSolve: PETScAlgorithm, LinearCache, LinearSolve,
    OperatorAssumptions, LinearVerbosity
using SciMLBase: LinearSolution, build_linear_solution, ReturnCode, SciMLBase

# ── Helpers ───────────────────────────────────────────────────────────────────

# If no communicator was specified in PETScAlgorithm, default to COMM_SELF (serial).
# Users pass comm = MPI.COMM_WORLD for distributed solves.
resolve_comm(alg::PETScAlgorithm) = alg.comm === nothing ? MPI.COMM_SELF : alg.comm

# A communicator is parallel when it spans more than one MPI rank.
is_parallel(comm) = MPI.Comm_size(comm) > 1

# PETSc can be compiled for Float32, Float64, ComplexF32, or ComplexF64.
# getlib selects the matching C library at runtime based on the scalar type T.
get_petsclib(::Type{T} = Float64) where {T} = PETSc.getlib(; PetscScalar = T)

# ── Sparsity fingerprint ──────────────────────────────────────────────────────
#
#  A cheap structural hash used to detect whether the sparsity pattern of the
#  system matrix changed between successive solves, without comparing every
#  entry.
#
#  For SparseMatrixCSC: hashes colptr, rowval and the matrix dimensions.
#  Two matrices with identical structure but different nzval entries hash the
#  same — exactly what we need to recognise "values changed, pattern did not"
#  (Case 3 in solve!).
#
#  For dense matrices (AbstractMatrix fallback): always returns UInt(0).
#  Dense matrices have no sparse structure to compare.  PETSc is almost
#  exclusively used with sparse matrices (PDEs, FEM, NonlinearSolve.jl
#  Jacobians), so the dense path is a correctness fallback rather than an
#  optimised workflow.  Returning UInt(0) for all dense matrices means two
#  different dense inputs hash identically, but since the `!(cache.A isa
#  SparseMatrixCSC)` guard in solve! always routes dense matrices to a full
#  rebuild (Case 2), this is safe.

sparsity_fingerprint(A::SparseMatrixCSC) = hash(A.rowval, hash(A.colptr, hash(size(A))))
sparsity_fingerprint(::AbstractMatrix) = UInt(0)

# ── Cache ─────────────────────────────────────────────────────────────────────
#
#  PETScCache{T} owns all C-side PETSc objects for a single LinearSolve cache.
#  T is the PETSc scalar type (matching eltype of the system matrix) and is
#  used to correctly type the pre-allocated MPI gather buffer.
#
#  Fields
#  ──────
#  ksp           KSP (Krylov SubSpace) solver object.
#  petsclib      PETSc library handle; selects the scalar-type-specific library.
#  comm          MPI communicator for this solve.
#  petsc_A       System matrix in PETSc internal format.
#  petsc_P       Preconditioner matrix.  Aliases petsc_A when no separate
#                prec_matrix is provided (same object pointer, NOT a copy).
#  nullspace_obj MatNullSpace object; nothing if no null space is specified.
#  petsc_x       Distributed solution vector (owned rows on this MPI rank).
#  petsc_b       Distributed right-hand side vector.
#  vec_n         Length of the last-created vectors; used to detect resizing.
#  rstart/rend   Local row ownership range [rstart, rend) on this rank
#                (0-indexed, following PETSc convention).
#  sparsity_hash Fingerprint of the last-assembled matrix's sparsity pattern.
#                Compared against the new matrix at the start of each solve to
#                decide between Case 2 (rebuild) and Case 3 (values-only update).
#  initialized   True once PETSc has been initialized for this cache instance.
#  mpi_counts    Per-rank local vector lengths, cached after first parallel gather.
#  mpi_displs    Cumulative displacements for MPI.Allgatherv!, cached similarly.
#  local_buf     Pre-allocated send buffer for MPI.Allgatherv!, avoids per-solve
#                allocation on repeated parallel solves.

mutable struct PETScCache{T}
    ksp::Any
    petsclib::Any
    comm::Any
    petsc_A::Any
    petsc_P::Any
    nullspace_obj::Any
    petsc_x::Any
    petsc_b::Any
    vec_n::Int
    rstart::Int
    rend::Int
    sparsity_hash::UInt
    initialized::Bool
    mpi_counts::Union{Nothing, Vector{Int32}}
    mpi_displs::Union{Nothing, Vector{Int32}}
    local_buf::Union{Nothing, Vector{T}}
end

# Zero out all fields without touching any C-side PETSc objects.
# Only called after the PETSc objects have already been destroyed (or were
# never created in the first place).
function _nullify_all!(pcache::PETScCache)
    pcache.ksp = pcache.petsc_A = pcache.petsc_P = pcache.nullspace_obj = nothing
    pcache.petsc_x = pcache.petsc_b = nothing
    pcache.vec_n = pcache.rstart = pcache.rend = 0
    pcache.sparsity_hash = UInt(0)
    pcache.initialized = false
    return pcache.mpi_counts = pcache.mpi_displs = pcache.local_buf = nothing
end

"""
    cleanup_petsc_cache!(pcache::PETScCache)

Destroy all PETSc objects associated with `pcache` and reset its state.
Safe to call multiple times — subsequent calls are no-ops.

Destruction order is critical because PETSc objects hold internal references:
  1. KSP first — it holds references to petsc_A and petsc_P internally.
  2. MatNullSpace before matrices — it is attached to petsc_A.
  3. Vectors — independent of the matrices, order between them does not matter.
  4. Preconditioner matrix, only if it is a distinct object from petsc_A.
  5. System matrix last — nothing else should reference it at this point.
"""
function cleanup_petsc_cache!(pcache::PETScCache)
    # Nothing to destroy if PETSc was never initialised, or if it has already
    # been finalised (e.g. MPI.Finalize was called before the GC ran).
    if pcache.petsclib === nothing || (pcache.initialized && !PETSc.initialized(pcache.petsclib))
        _nullify_all!(pcache)
        return
    end
    try
        pcache.ksp !== nothing && PETSc.destroy(pcache.ksp)
        pcache.nullspace_obj !== nothing &&
            LibPETSc.MatNullSpaceDestroy(pcache.petsclib, pcache.nullspace_obj)
        pcache.petsc_x !== nothing && PETSc.destroy(pcache.petsc_x)
        pcache.petsc_b !== nothing && PETSc.destroy(pcache.petsc_b)
        # petsc_P aliases petsc_A when no separate preconditioner matrix was
        # provided.  Guard against double-free by checking object identity.
        pcache.petsc_P !== nothing && pcache.petsc_P !== pcache.petsc_A &&
            PETSc.destroy(pcache.petsc_P)
        pcache.petsc_A !== nothing && PETSc.destroy(pcache.petsc_A)
    catch
        # Swallow errors — cleanup is often called from a GC finalizer where
        # throwing is not safe.  Any leak here is preferable to a crash.
    end
    return _nullify_all!(pcache)
end

# Convenience overloads so callers don't have to reach into cacheval themselves.
cleanup_petsc_cache!(cache::LinearCache) = cleanup_petsc_cache!(cache.cacheval)
cleanup_petsc_cache!(sol::LinearSolution) = cleanup_petsc_cache!(sol.cache.cacheval)

# ── Cache init ────────────────────────────────────────────────────────────────

# Called by LinearSolve.init to allocate the solver-specific cache object.
# We create an empty shell here; actual PETSc objects are created lazily on
# the first call to solve! so PETSc is not initialised unless actually used.
# The finalizer is a safety net for GC-driven cleanup.  Users should call
# cleanup_petsc_cache! explicitly for deterministic, timely resource release,
# especially in MPI workflows where cleanup must be collective.
function LinearSolve.init_cacheval(
        alg::PETScAlgorithm, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    T = eltype(A)
    pcache = PETScCache{T}(
        nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, 0, 0, 0, UInt(0), false,
        nothing, nothing, nothing
    )
    finalizer(cleanup_petsc_cache!, pcache)
    return pcache
end

# ── Matrix Helpers ────────────────────────────────────────────────────────────

# Convert a Julia matrix to a PETSc sequential (single-rank) matrix.
#
# Sparse path — MatSeqAIJWithArrays:
#   Wraps the Julia SparseMatrixCSC data arrays (colptr, rowval, nzval)
#   with zero copy: PETSc holds a raw C pointer into the Julia memory.
#   This is safe because PETSc's sparse Krylov operations read but never
#   overwrite nzval.  Our update_mat_values! function writes new values
#   into A.nzval directly and then calls assemble! to notify PETSc (Case 3).
#
# Dense path — MatSeqDense:
#   A copy of A is passed.  The copy is necessary because PETSc's dense
#   preconditioners (e.g. LU factorisation) overwrite the matrix data in
#   place, which would silently corrupt the user's original array if we
#   passed it directly.  The copy is owned by the MatSeqDense object via
#   PETSc.jl's internal .array field, keeping it alive for the matrix lifetime.
#
#   Note: dense matrices are a rare fallback.  PETSc is designed for sparse
#   problems and most preconditioners work only with sparse formats.
function to_petsc_mat_seq(petsclib, A)
    if A isa SparseMatrixCSC
        return PETSc.MatSeqAIJWithArrays(petsclib, MPI.COMM_SELF, A)
    else
        return PETSc.MatSeqDense(petsclib, copy(A))
    end
end

# Convert a Julia matrix to a PETSc MPI-parallel matrix (MatMPIAIJ).
#
# Current strategy — Phase 1 (naive replication):
#   Every MPI rank holds the full Julia matrix in memory.  PETSc is asked to
#   decide the row distribution (PETSC_DECIDE, i.e. -1 for local sizes), and
#   each rank queries its ownership range and inserts only those rows.  PETSc
#   then performs the parallel Krylov solve with proper MPI communication.
#
# Limitation — memory is NOT reduced by parallelism:
#   A 10 GB matrix on 8 ranks still uses 80 GB total.  True memory scalability
#   (Phase 2) would require each rank to hold only its local chunk, which needs
#   a different LinearSolve.jl interface where users supply local matrix slices
#   rather than the full replicated matrix.
function to_petsc_mat_mpi(petsclib, A, comm)
    A isa SparseMatrixCSC || (A = sparse(A))
    M, N = size(A)

    # Create a parallel matrix and let PETSc distribute rows evenly.
    # Passing -1 for local sizes means PETSC_DECIDE.
    PA = LibPETSc.MatCreate(petsclib, comm)
    LibPETSc.MatSetSizes(
        petsclib, PA,
        petsclib.PetscInt(-1), petsclib.PetscInt(-1),
        petsclib.PetscInt(M), petsclib.PetscInt(N)
    )
    LibPETSc.MatSetFromOptions(petsclib, PA)
    LibPETSc.MatSetUp(petsclib, PA)

    # Query the half-open row range [rstart, rend) owned by this rank.
    # PETSc uses 0-based indexing; Julia uses 1-based.
    rstart, rend = Int.(LibPETSc.MatGetOwnershipRange(petsclib, PA))

    # Insert only the locally owned rows — the ownership guard skips all others.
    @inbounds for col in 1:N
        for idx in nzrange(A, col)
            row = A.rowval[idx]
            (rstart <= row - 1 < rend) && (PA[row, col] = A.nzval[idx])
        end
    end
    PETSc.assemble!(PA)
    return PA, rstart, rend
end

# Unified entry point: dispatch to sequential or MPI matrix construction.
# Returns (petsc_mat, rstart, rend) in all cases for a uniform call site.
# Serial: rstart = 0, rend = size(A, 1) — the entire matrix is local.
function to_petsc_mat(petsclib, A, comm)
    return is_parallel(comm) ? to_petsc_mat_mpi(petsclib, A, comm) :
        (to_petsc_mat_seq(petsclib, A), 0, size(A, 1))
end

# Update the numerical values of an existing PETSc sparse matrix in place.
# Called in Case 3: the sparsity pattern is unchanged but the values have
# changed (e.g. a new Newton Jacobian with the same graph as the previous one).
#
# Re-inserting via PA[row, col] = ... calls MatSetValues internally.
# The subsequent assemble! flushes the insertion stash and bumps PETSc's
# internal "matrix modified" state counter, which causes the KSP to recompute
# the preconditioner on the next KSPSolve automatically — no explicit
# KSPSetOperators call is needed.
#
# Only defined for SparseMatrixCSC because dense matrices always use a full
# rebuild (Case 2): PETSc's dense factorisation overwrites matrix data in place,
# making reliable in-place updates unsafe.
function update_mat_values!(petsclib, PA, A::SparseMatrixCSC, rstart, rend)
    N = size(A, 2)
    @inbounds for col in 1:N
        for idx in nzrange(A, col)
            row = A.rowval[idx]
            (rstart <= row - 1 < rend) && (PA[row, col] = A.nzval[idx])
        end
    end
    return PETSc.assemble!(PA)
end

# ── Vec read/write ────────────────────────────────────────────────────────────
#
#  PETSc vectors created via VecCreateMPI return a low-level LibPETSc.PetscVec
#  handle that is not compatible with the high-level PETSc.unsafe_localarray
#  API (which expects AbstractVec).  We therefore access the underlying memory
#  directly through the raw C pointer API:
#
#    VecGetArray         writable pointer to the local data segment
#    VecRestoreArray     release the writable pointer (mandatory after Get)
#    VecGetArrayRead     read-only pointer (does not invalidate cached norms)
#    VecRestoreArrayRead release the read-only pointer
#
#  VecAssemblyBegin/End is NOT needed after direct pointer writes — assembly is
#  only required after VecSetValues, which routes values through an off-process
#  communication stash.  Direct pointer access bypasses the stash entirely.

# Write src[rstart+1 : rend] into the local segment of PETSc Vec pv.
function write_local_values!(pv, src::AbstractVector, petsclib, rstart::Int, rend::Int)
    local_n = rend - rstart
    ptr = LibPETSc.VecGetArray(petsclib, pv)
    @inbounds for i in 1:local_n
        ptr[i] = src[rstart + i]
    end
    return LibPETSc.VecRestoreArray(petsclib, pv, ptr)
end

# Read the local segment of PETSc Vec pv into dst[rstart+1 : rend].
function read_local_values!(dst::AbstractVector, pv, petsclib, rstart::Int, rend::Int)
    local_n = rend - rstart
    ptr = LibPETSc.VecGetArrayRead(petsclib, pv)
    @inbounds for i in 1:local_n
        dst[rstart + i] = ptr[i]
    end
    return LibPETSc.VecRestoreArrayRead(petsclib, pv, ptr)
end

# ── Distributed vectors ───────────────────────────────────────────────────────

# Allocate a new MPI-distributed PETSc Vec of global length length(v) and
# initialise its locally owned segment [rstart+1 : rend] from v.
# In serial mode (COMM_SELF) the entire vector is local.
function create_distributed_vec(petsclib, v::AbstractVector, rstart, rend, comm)
    local_n = rend - rstart
    pv = LibPETSc.VecCreateMPI(
        petsclib, comm,
        petsclib.PetscInt(local_n), petsclib.PetscInt(length(v))
    )
    write_local_values!(pv, v, petsclib, rstart, rend)
    return pv
end

# Ensure petsc_x (solution) and petsc_b (RHS) exist and match the current
# problem size and MPI partition.  Vectors are recreated only when necessary
# — when the global length n, or the local ownership range (rstart, rend)
# changed — which avoids allocation overhead on repeated solves.
# Recreating vectors also invalidates the cached MPI gather metadata because
# the per-rank counts and displacements are partition-dependent.
function ensure_distributed_vecs!(pcache, petsclib, n, u, b, rstart, rend, comm)
    if pcache.vec_n != n || pcache.rstart != rstart || pcache.rend != rend ||
            pcache.petsc_x === nothing || pcache.petsc_b === nothing
        pcache.petsc_x !== nothing && PETSc.destroy(pcache.petsc_x)
        pcache.petsc_b !== nothing && PETSc.destroy(pcache.petsc_b)
        pcache.petsc_x = create_distributed_vec(petsclib, u, rstart, rend, comm)
        pcache.petsc_b = create_distributed_vec(petsclib, b, rstart, rend, comm)
        pcache.vec_n = n
        pcache.rstart = rstart
        pcache.rend = rend
        # Partition changed — cached gather metadata is stale.
        pcache.mpi_counts = pcache.mpi_displs = pcache.local_buf = nothing
    end
    return pcache.petsc_x, pcache.petsc_b
end

# ── Gather solution ───────────────────────────────────────────────────────────

# Compute and cache the MPI Allgatherv metadata for the current partition.
# Called lazily on the first parallel gather after vector creation; reused on
# all subsequent solves with the same partition to avoid per-solve allocation.
#   mpi_counts  number of locally owned rows on each rank
#   mpi_displs  cumulative offsets into the global receive buffer
#   local_buf   pre-allocated send buffer (avoids allocation in the hot path)
function setup_mpi_gather!(pcache::PETScCache{T}) where {T}
    pcache.mpi_counts !== nothing && return  # already set up for this partition
    local_n = pcache.rend - pcache.rstart
    counts = MPI.Allgather(Int32(local_n), pcache.comm)
    displs = cumsum([Int32(0); counts[1:(end - 1)]])
    pcache.mpi_counts = counts
    pcache.mpi_displs = displs
    return pcache.local_buf = Vector{T}(undef, local_n)
end

# Gather the full distributed solution from petsc_v into the Julia vector dst
# on ALL MPI ranks via MPI.Allgatherv.
#
# We gather to all ranks (not just rank 0) so that dst is immediately usable
# for residual checks and post-processing everywhere without a separate
# user-side broadcast.
#
# Serial fast path: all rows are already local — a single pointer read suffices.
function gather_solution!(dst::AbstractVector, petsc_v, pcache::PETScCache{T}) where {T}
    rstart, rend = pcache.rstart, pcache.rend
    petsclib = pcache.petsclib

    if !is_parallel(pcache.comm)
        read_local_values!(dst, petsc_v, petsclib, rstart, rend)
        return dst
    end

    # Read this rank's local segment into the pre-allocated send buffer, then
    # assemble the full vector on all ranks via a single Allgatherv call.
    setup_mpi_gather!(pcache)
    local_buf = pcache.local_buf::Vector{T}
    local_n = rend - rstart

    ptr = LibPETSc.VecGetArrayRead(petsclib, petsc_v)
    @inbounds for i in 1:local_n
        local_buf[i] = ptr[i]
    end
    LibPETSc.VecRestoreArrayRead(petsclib, petsc_v, ptr)

    recvbuf = MPI.VBuffer(dst, pcache.mpi_counts, pcache.mpi_displs)
    MPI.Allgatherv!(local_buf, recvbuf, pcache.comm)

    return dst
end

# ── Nullspace helpers ─────────────────────────────────────────────────────────

# Build a PETSc MatNullSpace from the algorithm's null-space specification.
#
# :none     — singular system without null-space handling; returns nothing.
#
# :constant — the globally constant vector spans the null space (e.g. pressure
#             in incompressible-flow problems with pure Neumann BCs).  PETSc
#             constructs this internally without user-supplied vectors.
#
# :custom   — an explicit orthonormal basis for the null space (e.g. rigid body
#             modes in structural mechanics).  The caller supplies the basis in
#             alg.nullspace_vecs.  PETSc copies the vectors into its internal
#             storage when MatNullSpaceCreate is called, so the temporary PETSc
#             Vec wrappers can be destroyed immediately afterwards.
#             nullspace_vecs should be unit-normalised; PETSc normalises them
#             internally but pre-normalising avoids numerical surprises with
#             multi-vector null spaces.
function build_nullspace(petsclib, alg::PETScAlgorithm, comm)
    if alg.nullspace === :none
        return nothing
    elseif alg.nullspace === :constant
        return LibPETSc.MatNullSpaceCreate(
            petsclib, comm, LibPETSc.PetscBool(true), 0, LibPETSc.PetscVec[]
        )
    else  # :custom
        PScalar = petsclib.PetscScalar
        # Wrap each Julia basis vector in a temporary PETSc Vec, hand them to
        # MatNullSpaceCreate, then destroy the temporaries immediately.
        petsc_vecs = LibPETSc.PetscVec[
            create_distributed_vec(petsclib, PScalar.(v), 0, length(v), comm)
                for v in alg.nullspace_vecs
        ]
        ns = LibPETSc.MatNullSpaceCreate(
            petsclib, comm, LibPETSc.PetscBool(false),
            length(petsc_vecs), petsc_vecs
        )
        foreach(PETSc.destroy, petsc_vecs)
        return ns
    end
end

# Attach the MatNullSpace to petsc_A via MatSetNullSpace so PETSc projects out
# the null-space component from the Krylov residual at every iteration.
# This prevents divergence on singular systems.  No-op when ns === nothing.
function attach_nullspace!(petsclib, ksp, petsc_A, ns)
    ns === nothing && return
    return LibPETSc.MatSetNullSpace(petsclib, petsc_A, ns)
end

# ── KSP build ─────────────────────────────────────────────────────────────────

# Full KSP construction: create PETSc matrices, configure the KSP solver and
# preconditioner, and optionally attach a null space.
#
# Tolerances (reltol, abstol, maxiters) come from the LinearSolve cache, which
# is populated from the keyword arguments passed to solve! / init.
#
# alg.ksp_options is splatted last and forwarded verbatim to PETSc's Options
# Database.  Options passed this way take precedence over the positional
# tolerance arguments if there is overlap (e.g. ksp_options = (ksp_rtol=1e-14,)
# overrides cache.reltol silently) — users should be aware of this precedence.
function build_ksp!(pcache, petsclib, cache, alg, comm)
    # Convert the Julia system matrix to a PETSc matrix.
    # rstart/rend are the locally owned row range [rstart, rend) on this rank.
    pcache.petsc_A, pcache.rstart, pcache.rend = to_petsc_mat(petsclib, cache.A, comm)

    # If a separate preconditioner matrix was provided, convert it as well.
    # Otherwise petsc_P is the same object as petsc_A (aliased, not copied),
    # meaning PETSc uses the system matrix for both the operator and the PC.
    pcache.petsc_P = alg.prec_matrix === nothing ? pcache.petsc_A :
        to_petsc_mat(petsclib, alg.prec_matrix, comm)[1]

    pcache.ksp = PETSc.KSP(
        pcache.petsc_A, pcache.petsc_P;
        ksp_type = string(alg.solver_type),
        pc_type = string(alg.pc_type),
        ksp_rtol = cache.reltol,
        ksp_atol = cache.abstol,
        ksp_max_it = cache.maxiters,
        alg.ksp_options...          # forwarded verbatim to the PETSc Options DB
    )

    # Warm-start: tell PETSc to use the current petsc_x content as the initial
    # Krylov guess instead of zeroing it before the solve.
    if alg.initial_guess_nonzero
        LibPETSc.KSPSetInitialGuessNonzero(petsclib, pcache.ksp, LibPETSc.PetscBool(true))
    end

    pcache.nullspace_obj = alg.nullspace !== :none ?
        build_nullspace(petsclib, alg, comm) : nothing
    return attach_nullspace!(petsclib, pcache.ksp, pcache.petsc_A, pcache.nullspace_obj)
end

# ── Solve ─────────────────────────────────────────────────────────────────────

function SciMLBase.solve!(cache::LinearCache, alg::PETScAlgorithm; kwargs...)

    pcache = cache.cacheval
    comm = resolve_comm(alg)

    # Select the PETSc library that matches the element type of the system matrix.
    # If the cache already has a petsclib (from a previous solve) reuse it —
    # eltype should not change between solves for the same cache.
    petsclib = pcache.petsclib === nothing ? get_petsclib(eltype(cache.A)) : pcache.petsclib
    PETSc.initialized(petsclib) || PETSc.initialize(petsclib)
    pcache.petsclib = petsclib
    pcache.comm = comm
    pcache.initialized = true

    # ── KSP build / reuse decision ────────────────────────────────────────────
    #
    #  Case 1 — first solve (ksp === nothing):
    #    Build everything from scratch.
    #
    #  Case 2 — isfresh, and sparsity changed or matrix is dense:
    #    Destroy existing PETSc objects and rebuild.
    #    Dense matrices always land here because PETSc's dense preconditioners
    #    overwrite matrix data in place during factorisation, making reliable
    #    in-place value updates unsafe.  In practice this is not a significant
    #    limitation — NonlinearSolve.jl and OrdinaryDiffEq.jl always produce
    #    SparseMatrixCSC Jacobians.
    #
    #  Case 3 — isfresh, same sparse sparsity pattern:
    #    Update matrix values in place (update_mat_values!) and reuse the KSP.
    #    This avoids the expensive KSP rebuild on every Newton/time step when
    #    only the numerical values change.  Preconditioner reuse is opt-in via
    #    reinit!(cache; reuse_precs = true) — see below.
    #
    #  isfresh = false — matrix unchanged since last solve:
    #    Only the RHS needs updating; reuse everything else as-is.

    rebuild_ksp = false
    if pcache.ksp === nothing
        rebuild_ksp = true   # Case 1
    elseif cache.isfresh
        new_hash = sparsity_fingerprint(cache.A)
        sparse_same = (cache.A isa SparseMatrixCSC) && (new_hash == pcache.sparsity_hash)
        rebuild_ksp = !sparse_same   # Case 2 if sparsity changed or matrix is dense
    end

    if rebuild_ksp
        # Cases 1 and 2: full build.
        # build_ksp! overwrites pcache fields; old PETSc objects referenced only
        # by the overwritten fields become unreachable and will be cleaned up by
        # the finalizer.  Explicit cleanup before calling build_ksp! could be
        # added here if deterministic memory release between rebuilds is needed.
        build_ksp!(pcache, petsclib, cache, alg, comm)
        pcache.sparsity_hash = sparsity_fingerprint(cache.A)
    else
        # Case 3: update numerical values in the existing PETSc matrix.
        update_mat_values!(petsclib, pcache.petsc_A, cache.A, pcache.rstart, pcache.rend)
        if alg.prec_matrix !== nothing
            update_mat_values!(
                petsclib, pcache.petsc_P, alg.prec_matrix, pcache.rstart, pcache.rend
            )
        end
        # Preconditioner reuse follows the LinearSolve.jl convention:
        #   reinit!(cache)                   sets precsisfresh = true  (default)
        #     → recompute the preconditioner (safe for any value change)
        #   reinit!(cache; reuse_precs=true) sets precsisfresh = false
        #     → skip preconditioner recomputation (safe only when values changed
        #       very little, e.g. near Newton convergence)
        if !cache.precsisfresh
            LibPETSc.KSPSetReusePreconditioner(
                petsclib, pcache.ksp, LibPETSc.PetscBool(true)
            )
        end
    end
    cache.isfresh = false

    # ── Vectors ───────────────────────────────────────────────────────────────
    # Ensure petsc_x and petsc_b are allocated and sized correctly, then copy
    # the current u (initial guess or warm-start) and b (RHS) into them.
    petsc_x, petsc_b = ensure_distributed_vecs!(
        pcache, petsclib, length(cache.b),
        cache.u, cache.b, pcache.rstart, pcache.rend, comm
    )
    write_local_values!(petsc_x, cache.u, petsclib, pcache.rstart, pcache.rend)
    write_local_values!(petsc_b, cache.b, petsclib, pcache.rstart, pcache.rend)

    # ── Solve ─────────────────────────────────────────────────────────────────
    if alg.transposed
        LibPETSc.KSPSolveTranspose(petsclib, pcache.ksp, petsc_b, petsc_x)
    else
        LibPETSc.KSPSolve(petsclib, pcache.ksp, petsc_b, petsc_x)
    end

    # Gather the distributed solution back into cache.u on ALL MPI ranks so it
    # is immediately usable for residual checks everywhere without a separate
    # user-side broadcast.
    gather_solution!(cache.u, petsc_x, pcache)

    # ── Return metadata ───────────────────────────────────────────────────────
    iters = Int(LibPETSc.KSPGetIterationNumber(petsclib, pcache.ksp))
    reason = Int(LibPETSc.KSPGetConvergedReason(petsclib, pcache.ksp))
    # KSPGetResidualNorm returns a real value even for complex PetscScalar, so
    # Float64 is always the correct target type here.
    resid = Float64(LibPETSc.KSPGetResidualNorm(petsclib, pcache.ksp))
    # reason > 0 : converged within tolerances
    # reason = 0 : iteration limit not yet reached (still iterating, unusual)
    # reason < 0 : diverged or other failure
    retcode = reason > 0 ? ReturnCode.Success :
        reason == 0 ? ReturnCode.Default : ReturnCode.Failure

    return build_linear_solution(alg, cache.u, resid, cache; retcode, iters)
end

end
