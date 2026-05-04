module LinearSolvePETScExt

using PETSc
import PETSc: MPI, LibPETSc
using SparseArrays: SparseMatrixCSC, nzrange, sparse
using SparseMatricesCSR: SparseMatrixCSR, getrowptr, getcolval
using LinearSolve: PETScAlgorithm, LinearCache, LinearSolve,
    OperatorAssumptions, LinearVerbosity
using SciMLBase: LinearSolution, build_linear_solution, ReturnCode, SciMLBase

# ── MPI communicator ──────────────────────────────────────────────────────────

# Default (serial) communicator resolution.  Return MPI.COMM_SELF or error if
# a non-serial comm is requested for a standard Julia matrix.
# Overloaded by LinearSolvePETScMPIExt to support PSparseMatrix / PVector.
function get_comm(alg::PETScAlgorithm, A)
    comm = alg.comm === nothing ? MPI.COMM_SELF : alg.comm
    comm == MPI.COMM_SELF || error(
        "PETScAlgorithm currently only supports MPI.COMM_SELF (serial solves). " *
            "Pass `comm = nothing` or `comm = MPI.COMM_SELF` to use serial mode, " *
            "or use a PSparseMatrix / PVector for MPI-parallel solves."
    )
    return comm
end

# Select the PETSc C library that matches the scalar type of the system matrix.
get_petsclib(::Type{T} = Float64) where {T} = PETSc.getlib(; PetscScalar = T)

# ── Sparsity pattern check ────────────────────────────────────────────────────

function sparsity_pattern_changed(old_colptr, old_rowval, new_A::SparseMatrixCSC)
    # 1. Quick length checks
    if length(old_colptr) != length(new_A.colptr) || length(old_rowval) != length(new_A.rowval)
        return true
    end

    # 2. Vectorized comparison (very fast in Julia)
    # This checks both sizes and element-wise equality
    return old_colptr != new_A.colptr || old_rowval != new_A.rowval
end

# ── Cache ─────────────────────────────────────────────────────────────────────

"""
    PETScCache{T}

Owns all PETSc C-side objects for one LinearSolve cache instance.
`T` is the PETSc scalar type (must match `eltype` of the system matrix).

Fields
──────
- `ksp`            — KSP (Krylov subspace) solver object.
- `petsclib`       — PETSc library handle; selects the scalar-type-specific C library.
- `comm`           — MPI communicator (always `COMM_SELF` in this implementation).
- `petsc_A`        — System matrix in PETSc format.
- `petsc_P`        — Preconditioner matrix.  Aliases `petsc_A` when no separate
                     `prec_matrix` is provided (same pointer, not a copy).
- `nullspace_obj`  — `MatNullSpace` object, or `nothing` if unused.
- `petsc_x`        — Solution vector.
- `petsc_b`        — Right-hand side vector.
- `vec_n`          — Length of the currently allocated vectors; used to detect resizing.
- `prev_colptr`    — Previous matrix's colptr array (for sparse matrices).
- `prev_rowval`    — Previous matrix's rowval array (for sparse matrices).
- `prev_size`      — Previous matrix's size (for dense matrices).
- `sparse_perm`    — Permutation vector of length `nnz` mapping PETSc's internal
                     CSR index to Julia's CSC `nzval` index.  Allocated once per
                     KSP build and reused on every Case 3 update with zero allocations.
- `sparse_scratch` — Scratch buffer of length `2*(m+1)` used by `_csc_to_csr_perm!`
                     to avoid allocation during permutation construction.
- `mpi_data`       — Opaque slot for MPI-specific per-matrix data (populated by
                     `LinearSolvePETScMPIExt` when A is a `PSparseMatrix`).  `nothing`
                     for all serial matrix types.
- `initialized`    — `true` once `PETSc.initialize` has been called for this cache.
"""
mutable struct PETScCache{T}
    ksp::Any
    petsclib::Any
    comm::Union{MPI.Comm, Nothing}
    petsc_A::Any
    petsc_P::Any
    nullspace_obj::Any
    petsc_x::Any
    petsc_b::Any
    vec_n::Int
    prev_colptr::Union{Vector{Int}, Nothing}
    prev_rowval::Union{Vector{Int}, Nothing}
    prev_size::Union{NTuple{2, Int}, Nothing}
    sparse_perm::Union{Vector{Int}, Nothing}
    sparse_scratch::Union{Vector{Int}, Nothing}
    mpi_data::Any
    initialized::Bool
end

# Reset all fields to their zero/nothing state.
# Only called after C-side PETSc objects have already been destroyed.
function _nullify_all!(pcache::PETScCache)
    pcache.ksp = pcache.petsc_A = pcache.petsc_P = pcache.nullspace_obj = nothing
    pcache.petsc_x = pcache.petsc_b = nothing
    pcache.sparse_perm = pcache.sparse_scratch = pcache.mpi_data = nothing
    pcache.vec_n = 0
    pcache.prev_colptr = pcache.prev_rowval = pcache.prev_size = nothing
    return pcache.initialized = false
end

"""
    cleanup_petsc_cache!(pcache::PETScCache)
    cleanup_petsc_cache!(cache::LinearCache)
    cleanup_petsc_cache!(sol::LinearSolution)

Destroy all PETSc objects owned by `pcache` and reset its state to empty.
Safe to call multiple times — subsequent calls after the first are no-ops.

Prefer calling this explicitly for deterministic resource release; the cache
also registers a GC finalizer as a safety net.

Destruction order matters because PETSc objects hold internal references:
  1. KSP — holds references to `petsc_A` and `petsc_P`.
  2. MatNullSpace — attached to `petsc_A`.
  3. Vectors — independent of matrices; order between them does not matter.
  4. Preconditioner matrix — only if distinct from `petsc_A` (avoid double-free).
  5. System matrix — destroyed last, once nothing else references it.
"""
function cleanup_petsc_cache!(pcache::PETScCache)
    if pcache.petsclib === nothing ||
            (pcache.initialized && !PETSc.initialized(pcache.petsclib))
        _nullify_all!(pcache)
        return
    end
    try
        pcache.ksp !== nothing && PETSc.destroy(pcache.ksp)
        pcache.nullspace_obj !== nothing &&
            LibPETSc.MatNullSpaceDestroy(pcache.petsclib, pcache.nullspace_obj)
        pcache.petsc_x !== nothing && PETSc.destroy(pcache.petsc_x)
        pcache.petsc_b !== nothing && PETSc.destroy(pcache.petsc_b)
        pcache.petsc_P !== nothing && pcache.petsc_P !== pcache.petsc_A &&
            PETSc.destroy(pcache.petsc_P)
        pcache.petsc_A !== nothing && PETSc.destroy(pcache.petsc_A)
    catch
        # Swallow errors — cleanup may be called from a GC finalizer where
        # throwing is unsafe.
    end
    return _nullify_all!(pcache)
end

cleanup_petsc_cache!(cache::LinearCache) = cleanup_petsc_cache!(cache.cacheval)
cleanup_petsc_cache!(sol::LinearSolution) = cleanup_petsc_cache!(sol.cache.cacheval)

# ── Cache initialisation ──────────────────────────────────────────────────────

# Called by LinearSolve.init to allocate the solver-specific cache.
# An empty shell is returned here; actual PETSc objects are created lazily on
# the first solve! call so PETSc is never initialised unless actually used.
function LinearSolve.init_cacheval(
        alg::PETScAlgorithm, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    T = eltype(A)
    pcache = PETScCache{T}(
        nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, 0, nothing, nothing, nothing, nothing, nothing,
        nothing, false
    )
    finalizer(cleanup_petsc_cache!, pcache)
    return pcache
end

# ── Matrix conversion ─────────────────────────────────────────────────────────

# Trait function: is A a sparse format that PETSc can handle as an AIJ matrix?
is_sparse_petsc(A) = A isa SparseMatrixCSC
is_sparse_petsc(::SparseMatrixCSR) = true

# SparseMatrixCSR is already in PETSc's native CSR order, so the conversion can
# reuse the row/column arrays directly after canonicalizing the index base.
function _to_petsc_canonical(
        A::SparseMatrixCSR{Bi, Tv, Ti},
        ::Type{PetscInt}, ::Type{PetscScalar}
    ) where {Bi, Tv, Ti, PetscInt, PetscScalar}
    m, n = size(A)
    src_rp = getrowptr(A)
    src_cv = getcolval(A)
    src_nz = A.nzval

    if Bi == 0
        if Ti == PetscInt
            Tv == PetscScalar && return A
            rp = src_rp
            cv = src_cv
        else
            rp = PetscInt.(src_rp)
            cv = PetscInt.(src_cv)
        end
    else
        rp = PetscInt[v - Bi for v in src_rp]
        cv = PetscInt[v - Bi for v in src_cv]
    end

    nz = Tv == PetscScalar ? copy(src_nz) : PetscScalar.(src_nz)
    return SparseMatrixCSR{0}(m, n, rp, cv, nz)
end

function to_petsc_mat(petsclib, A::SparseMatrixCSR{Bi}, pcache = nothing) where {Bi}
    PetscInt = PETSc.inttype(petsclib)
    PetscScalar = PETSc.scalartype(petsclib)

    canonical = _to_petsc_canonical(A, PetscInt, PetscScalar)
    m, n = size(canonical)

    mat = LibPETSc.MatCreateSeqAIJWithArrays(
        petsclib, MPI.COMM_SELF,
        PetscInt(m), PetscInt(n),
        getrowptr(canonical), getcolval(canonical), canonical.nzval
    )

    PETSc._MATSEQAIJ_WITHARRAYS_STORAGE[mat.ptr] = canonical
    return mat
end

function store_sparse_pattern!(pcache, A::SparseMatrixCSR)
    pcache.sparse_perm = nothing
    pcache.sparse_scratch = nothing
    pcache.prev_colptr = getrowptr(A)
    return pcache.prev_rowval = getcolval(A)
end

function check_pattern_changed(pcache, A::SparseMatrixCSR)
    pcache.prev_colptr === nothing && return true
    old_rp, old_cv = pcache.prev_colptr, pcache.prev_rowval
    new_rp, new_cv = getrowptr(A), getcolval(A)
    old_rp === new_rp && old_cv === new_cv && return false
    (length(old_rp) != length(new_rp) || length(old_cv) != length(new_cv)) && return true
    return old_rp != new_rp || old_cv != new_cv
end

function update_sparse_values!(
        petsclib, PA, pcache, A::SparseMatrixCSR; assemble::Bool = true
    )
    vals = LibPETSc.MatSeqAIJGetArray(petsclib, PA)
    try
        pointer(vals) != pointer(A.nzval) && copyto!(vals, A.nzval)
    finally
        LibPETSc.MatSeqAIJRestoreArray(petsclib, PA, vals)
    end
    assemble && PETSc.assemble!(PA)
    return nothing
end

# Convert a Julia matrix to a PETSc matrix.
#
# Sparse (SparseMatrixCSC) → MatSeqAIJWithArrays :
#   Internally converts to CSR format.
# Dense (AbstractMatrix) → MatSeqDense:
#   PETSc allocates its own buffer and copies the values in on construction.
#   This protects the Julia array even when a factorising preconditioner is
#   used without a separate `prec_matrix`.
# PSparseMatrix  → handled by LinearSolvePETScMPIExt  via to_petsc_mat overload.
#
# The optional `pcache` argument is used by MPI-aware overloads to stash
# GC anchors and per-matrix metadata into the cache during matrix construction.
# Serial matrix types ignore it.
function to_petsc_mat(petsclib, A, pcache = nothing)
    if A isa SparseMatrixCSC
        return PETSc.MatSeqAIJWithArrays(petsclib, MPI.COMM_SELF, A)
    else
        return PETSc.MatSeqDense(petsclib, A)
    end
end

# ── CSC → CSR permutation ─────────────────────────────────────────────────────
#
# PETSc stores sparse matrix values in CSR (row-major) order internally, while
# Julia's SparseMatrixCSC uses CSC (column-major) order.  To bulk-copy nzval
# into PETSc's internal buffer via MatSeqAIJGetArray, we need a permutation
# perm such that:
#
#   petsc_vals[csr_idx] = A.nzval[perm[csr_idx]]
#
# scratch must have length 2*(m+1) and is used as two consecutive views:
#   scratch[1 : m+1]       → row_ptr   (CSR row start positions, 1-based)
#   scratch[m+2 : 2*(m+1)] → fill_pos  (current write cursor per row)
#
# Walking the CSC columns in ascending order guarantees that within each row,
# columns are visited in ascending order — matching PETSc's sorted-column
# requirement for CSR storage.
#
# This is mimicking MatSeqAIJWithArrays without rebuilding the matrix itself.
# This is currently the limiting factor for computing large system.
function _csc_to_csr_perm!(perm::Vector{Int}, scratch::Vector{Int}, A::SparseMatrixCSC)
    m, n = size(A)
    row_ptr = view(scratch, 1:(m + 1))
    fill_pos = view(scratch, (m + 2):(2 * (m + 1)))

    fill!(row_ptr, 0)
    # Pass 1: Count
    @inbounds for i in 1:length(A.rowval)
        row_ptr[A.rowval[i] + 1] += 1
    end

    # Pass 2: Cumulative sum (The "Compact" way)
    count = 1
    @inbounds for r in 1:m
        tmp = row_ptr[r + 1]
        row_ptr[r] = count
        count += tmp
    end
    row_ptr[m + 1] = count

    copyto!(fill_pos, row_ptr)

    # Pass 3: Map
    @inbounds for col in 1:n
        for csc_idx in nzrange(A, col)
            r = A.rowval[csc_idx]
            perm[fill_pos[r]] = csc_idx
            fill_pos[r] += 1
        end
    end
    return perm
end

# ── In-place matrix value updates (Case 3) ────────────────────────────────────

# Sparse: bulk-copy nzval into PETSc's internal CSR buffer in a single C call.
# The pre-computed permutation maps each CSR index to the corresponding CSC
# nzval index, so the inner loop is a plain indexed array read — no FFI
# overhead per element, no allocation.
# assemble! bumps PETSc's modification counter, triggering preconditioner
# recomputation on the next KSPSolve without an explicit KSPSetOperators call.
function update_mat_values!(petsclib, PA, A::SparseMatrixCSC, perm::Vector{Int}; assemble::Bool = true)
    vals = LibPETSc.MatSeqAIJGetArray(petsclib, PA)
    nzval = A.nzval
    try
        @inbounds for i in eachindex(vals)
            vals[i] = nzval[perm[i]]
        end
    finally
        LibPETSc.MatSeqAIJRestoreArray(petsclib, PA, vals)
    end
    assemble && PETSc.assemble!(PA)
    return nothing
end

# Dense: MatDenseGetArray wraps PETSc's column-major buffer as a Julia Array.
# copyto! is equivalent to a single memcpy — O(n²) with negligible constant.
function update_mat_values!(petsclib, PA, A::AbstractMatrix)
    ptr = LibPETSc.MatDenseGetArray(petsclib, PA)
    try
        copyto!(ptr, A)
    finally
        LibPETSc.MatDenseRestoreArray(petsclib, PA, ptr)
    end
    return PETSc.assemble!(PA)
end

# ── Extensible sparse dispatch ───────────────────────────────────────────────
#
# These functions are overloaded by LinearSolvePETScCSRExt to add CSR support.
# They abstract over the sparse format so build_ksp! and solve! don't need
# hardcoded `isa SparseMatrixCSC` checks.

function store_sparse_pattern!(pcache, A::SparseMatrixCSC)
    nnz_val = length(A.nzval)
    m = size(A, 1)
    if pcache.sparse_perm === nothing || length(pcache.sparse_perm) != nnz_val
        pcache.sparse_perm = Vector{Int}(undef, nnz_val)
        pcache.sparse_scratch = Vector{Int}(undef, 2 * (m + 1))
    end
    _csc_to_csr_perm!(pcache.sparse_perm, pcache.sparse_scratch, A)
    pcache.prev_colptr = A.colptr
    return pcache.prev_rowval = A.rowval
end

function check_pattern_changed(pcache, A::SparseMatrixCSC)
    pcache.prev_colptr === nothing && return true
    return sparsity_pattern_changed(pcache.prev_colptr, pcache.prev_rowval, A)
end

function update_sparse_values!(petsclib, PA, pcache, A::SparseMatrixCSC; assemble::Bool = true)
    return update_mat_values!(petsclib, PA, A, pcache.sparse_perm; assemble = assemble)
end

# ── Vector I/O (VecPlaceArray pattern) ───────────────────────────────────────
#
# Use VecPlaceArray / VecResetArray to temporarily redirect PETSc's internal
# Vec buffer to point at Julia arrays — zero copy in both directions.
#
#   VecPlaceArray(petsc_b, cache.b)   # redirect pointer — no copy
#   VecPlaceArray(petsc_x, cache.u)   # redirect pointer — initial guess in-place
#   KSPSolve(...)                     # writes solution directly into cache.u
#   VecResetArray(petsc_x/petsc_b)    # restore PETSc's own buffer
#
# This avoids 3 × O(n) memcpy per solve (write b, write x, read x back).
# VecPlaceArray requires eltype == PetscScalar, which is guaranteed because
# petsclib is chosen by eltype(cache.A) and u/b share that type.
# VecResetArray is always called in a finally block.

# Allocate a PETSc sequential Vec of length n.
# Contents are uninitialised; VecPlaceArray redirects its buffer before each use.
function create_seq_vec(petsclib, n::Int)
    return LibPETSc.VecCreateSeq(petsclib, MPI.COMM_SELF, petsclib.PetscInt(n))
end

# Allocate and fill a Vec from a Julia vector (used for nullspace basis vectors).
function create_seq_vec(petsclib, src::AbstractVector)
    pv = create_seq_vec(petsclib, length(src))
    ptr = LibPETSc.VecGetArray(petsclib, pv)
    try
        copyto!(ptr, src)
    finally
        LibPETSc.VecRestoreArray(petsclib, pv, ptr)
    end
    return pv
end

# Recreate petsc_x and petsc_b only when the problem size changes.
# `comm` is MPI.COMM_SELF for serial solves; MPI extensions pass the real comm.
# Overloaded by LinearSolvePETScMPIExt for PVector.
function ensure_vecs!(pcache, petsclib, comm, b)
    n = length(b)
    if pcache.vec_n != n || pcache.petsc_x === nothing || pcache.petsc_b === nothing
        pcache.petsc_x !== nothing && PETSc.destroy(pcache.petsc_x)
        pcache.petsc_b !== nothing && PETSc.destroy(pcache.petsc_b)
        pcache.petsc_x = create_seq_vec(petsclib, n)
        pcache.petsc_b = create_seq_vec(petsclib, n)
        pcache.vec_n = n
    end
    return pcache.petsc_x, pcache.petsc_b
end

# ── KSP solve dispatch ────────────────────────────────────────────────────────
#
# run_ksp! abstracts the VecPlaceArray / KSPSolve / VecResetArray pattern.
# For serial AbstractVector: zero-copy via VecPlaceArray.
# Overloaded by LinearSolvePETScMPIExt for PVector (uses VecGetArray copy path).
function run_ksp!(pcache, petsclib, alg, b::AbstractVector, u::AbstractVector)
    petsc_x = pcache.petsc_x
    petsc_b = pcache.petsc_b
    LibPETSc.VecPlaceArray(petsclib, petsc_b, b)
    LibPETSc.VecPlaceArray(petsclib, petsc_x, u)
    return try
        if alg.transposed
            LibPETSc.KSPSolveTranspose(petsclib, pcache.ksp, petsc_b, petsc_x)
        else
            LibPETSc.KSPSolve(petsclib, pcache.ksp, petsc_b, petsc_x)
        end
        # Solution is written directly into cache.u — no read-back needed.
    finally
        LibPETSc.VecResetArray(petsclib, petsc_x)
        LibPETSc.VecResetArray(petsclib, petsc_b)
    end
end

# ── Null space ────────────────────────────────────────────────────────────────

# Build a PETSc MatNullSpace from alg.nullspace:
#
#   :none     — no null-space handling; returns nothing.
#   :constant — the constant vector spans the null space (e.g. pressure in
#               pure-Neumann incompressible flow). PETSc constructs it internally.
#   :custom   — caller supplies an explicit orthonormal basis via alg.nullspace_vecs.
#               PETSc copies the vectors on creation, so the temporary wrappers
#               can be destroyed immediately after MatNullSpaceCreate returns.
function build_nullspace(petsclib, alg::PETScAlgorithm)
    alg.nullspace === :none     && return nothing
    alg.nullspace === :constant && return LibPETSc.MatNullSpaceCreate(
        petsclib, MPI.COMM_SELF, LibPETSc.PetscBool(true), 0, LibPETSc.PetscVec[]
    )
    # :custom
    PScalar = petsclib.PetscScalar
    petsc_vecs = LibPETSc.PetscVec[
        create_seq_vec(petsclib, eltype(v) == PScalar ? v : PScalar.(v))
            for v in alg.nullspace_vecs
    ]
    ns = LibPETSc.MatNullSpaceCreate(
        petsclib, MPI.COMM_SELF, LibPETSc.PetscBool(false),
        length(petsc_vecs), petsc_vecs
    )
    foreach(PETSc.destroy, petsc_vecs)
    return ns
end

function attach_nullspace!(petsclib, petsc_A, ns)
    ns === nothing && return
    return LibPETSc.MatSetNullSpace(petsclib, petsc_A, ns)
end

# ── Full KSP construction ─────────────────────────────────────────────────────

# Build PETSc matrices, configure the KSP solver and preconditioner, and
# optionally attach a null space.  Called on Case 1 (first solve) and Case 2
# (structure changed).
#
# For sparse matrices, the sparsity pattern and any permutation data are stored
# in pcache for reuse on all subsequent Case 3 value-only updates.
#
# vec_n is reset to 0 so ensure_vecs! unconditionally recreates petsc_x and
# petsc_b after every rebuild — required when the problem size changes.
function build_ksp!(pcache, petsclib, cache, alg)
    pcache.vec_n = 0
    # Pass pcache so MPI-aware overloads can stash GC anchors / metadata.
    pcache.petsc_A = to_petsc_mat(petsclib, cache.A, pcache)

    if is_sparse_petsc(cache.A)
        store_sparse_pattern!(pcache, cache.A)
    else
        pcache.sparse_perm = nothing
        pcache.sparse_scratch = nothing
        pcache.prev_size = size(cache.A)
    end

    # petsc_P aliases petsc_A when no separate preconditioner matrix is given.
    # prec_matrix is passed with pcache=nothing so MPI overloads do not clobber
    # the system-matrix mpi_data stored above.
    pcache.petsc_P = alg.prec_matrix === nothing ?
        pcache.petsc_A : to_petsc_mat(petsclib, alg.prec_matrix, nothing)

    pcache.ksp = PETSc.KSP(
        pcache.petsc_A, pcache.petsc_P;
        ksp_type = string(alg.solver_type),
        pc_type = string(alg.pc_type),
        ksp_rtol = cache.reltol,
        ksp_atol = cache.abstol,
        ksp_max_it = cache.maxiters,
        alg.ksp_options...
    )

    alg.initial_guess_nonzero &&
        LibPETSc.KSPSetInitialGuessNonzero(petsclib, pcache.ksp, LibPETSc.PetscBool(true))

    pcache.nullspace_obj = build_nullspace(petsclib, alg)
    return attach_nullspace!(petsclib, pcache.petsc_A, pcache.nullspace_obj)
end

# ── solve! ────────────────────────────────────────────────────────────────────

"""
    solve!(cache::LinearCache, alg::PETScAlgorithm; kwargs...)

Solve the linear system stored in `cache` using PETSc.

Three execution paths are chosen based on `cache.isfresh` and the matrix's
structural fingerprint:

  **Case 1** — first solve (`ksp === nothing`):
    Build all PETSc objects from scratch.

  **Case 2** — `isfresh`, structure changed (sparse pattern or dense size):
    Destroy existing PETSc objects and rebuild.

  **Case 3** — `isfresh`, same structure, values changed:
    Update matrix values in place and reuse the existing KSP.
    - Sparse: bulk-copies `nzval` into PETSc's internal CSR buffer using a
      pre-computed CSC→CSR permutation.  Zero allocations on the hot path;
      one C call in, one O(nnz) indexed loop, one C call out.
    - Dense: `copyto!` into PETSc's column-major buffer — equivalent to a
      single `memcpy`.

  **No change** (`isfresh = false`):
    Only the RHS is updated; all PETSc objects are reused as-is.
"""
function SciMLBase.solve!(cache::LinearCache, alg::PETScAlgorithm; kwargs...)

    pcache = cache.cacheval
    comm = get_comm(alg, cache.A)

    petsclib = pcache.petsclib === nothing ? get_petsclib(eltype(cache.A)) : pcache.petsclib
    PETSc.initialized(petsclib) || PETSc.initialize(petsclib)
    pcache.petsclib = petsclib
    pcache.comm = comm
    pcache.initialized = true

    # ── Decide: rebuild, update in-place, or reuse ────────────────────────────
    rebuild_ksp = pcache.ksp === nothing   # Case 1
    if !rebuild_ksp && cache.isfresh
        if is_sparse_petsc(cache.A)
            rebuild_ksp = check_pattern_changed(pcache, cache.A)
        else
            # For dense matrices, check if the size has changed
            rebuild_ksp = pcache.prev_size !== nothing && size(cache.A) != pcache.prev_size
        end
    end

    if rebuild_ksp
        build_ksp!(pcache, petsclib, cache, alg)
    elseif cache.isfresh
        # Case 3: same structure — update values without touching the KSP object.
        if is_sparse_petsc(cache.A)
            update_sparse_values!(
                petsclib, pcache.petsc_A, pcache, cache.A;
                assemble = cache.precsisfresh
            )
        else
            update_mat_values!(petsclib, pcache.petsc_A, cache.A)
        end
        if alg.prec_matrix !== nothing
            if is_sparse_petsc(alg.prec_matrix)
                update_sparse_values!(petsclib, pcache.petsc_P, pcache, alg.prec_matrix)
            else
                update_mat_values!(petsclib, pcache.petsc_P, alg.prec_matrix)
            end
        end

        # Follow the LinearSolve.jl preconditioner-reuse convention:
        #   reinit!(cache)                   → recompute preconditioner (default)
        #   reinit!(cache; reuse_precs=true) → skip recomputation
        if !cache.precsisfresh
            LibPETSc.KSPSetReusePreconditioner(petsclib, pcache.ksp, LibPETSc.PetscBool(true))
        end
    end
    cache.isfresh = false

    # ── Vectors + Solve ───────────────────────────────────────────────────────
    # ensure_vecs! creates (or reuses) petsc_x and petsc_b, dispatching on the
    # vector type so that MPI extensions can create distributed Vecs.
    # run_ksp! performs I/O + KSPSolve, also dispatching on vector type.
    ensure_vecs!(pcache, petsclib, comm, cache.b)
    run_ksp!(pcache, petsclib, alg, cache.b, cache.u)

    # ── Convergence metadata ──────────────────────────────────────────────────
    iters = Int(LibPETSc.KSPGetIterationNumber(petsclib, pcache.ksp))
    reason = Int(LibPETSc.KSPGetConvergedReason(petsclib, pcache.ksp))
    resid = Float64(LibPETSc.KSPGetResidualNorm(petsclib, pcache.ksp))
    # reason > 0 → converged; reason == 0 → iteration limit not yet hit;
    # reason < 0 → diverged or other failure.
    retcode = reason > 0 ? ReturnCode.Success :
        reason == 0 ? ReturnCode.Default : ReturnCode.Failure

    return build_linear_solution(alg, cache.u, resid, cache; retcode, iters)
end

end
