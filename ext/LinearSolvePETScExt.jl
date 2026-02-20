module LinearSolvePETScExt

using PETSc
import PETSc: MPI, LibPETSc
using SparseArrays: SparseMatrixCSC, nzrange, sparse
using LinearSolve: PETScAlgorithm, LinearCache, LinearSolve,
                   OperatorAssumptions, LinearVerbosity
using SciMLBase: LinearSolution, build_linear_solution, ReturnCode, SciMLBase

# ── Helpers ───────────────────────────────────────────────────────────────────

"""
    resolve_comm(alg::PETScAlgorithm)

Return the MPI communicator used by `alg`. Defaults to `MPI.COMM_SELF` if none specified.
"""
resolve_comm(alg::PETScAlgorithm) = alg.comm === nothing ? MPI.COMM_SELF : alg.comm

"""
    is_parallel(comm)

Return `true` if the communicator has more than one rank.
"""
is_parallel(comm) = MPI.Comm_size(comm) > 1

"""
    get_petsclib(::Type{T}=Float64)

Return the PETSc library handle for scalar type `T`.
"""
get_petsclib(::Type{T}=Float64) where T = PETSc.getlib(; PetscScalar=T)

# ── Sparsity fingerprint ──────────────────────────────────────────────────────

sparsity_fingerprint(A::SparseMatrixCSC) = hash(A.rowval, hash(A.colptr, hash(size(A))))
sparsity_fingerprint(::AbstractMatrix)   = UInt(0)

# ── Cache ─────────────────────────────────────────────────────────────────────

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

# Internal helper to clear all fields
function _nullify_all!(pcache::PETScCache)
    pcache.ksp = pcache.petsc_A = pcache.petsc_P = pcache.nullspace_obj = nothing
    pcache.petsc_x = pcache.petsc_b = nothing
    pcache.vec_n = pcache.rstart = pcache.rend = 0
    pcache.sparsity_hash = UInt(0)
    pcache.initialized = false
    pcache.mpi_counts = pcache.mpi_displs = pcache.local_buf = nothing
end

"""
    cleanup_petsc_cache!(pcache::PETScCache)

Destroy all PETSc objects associated with `pcache` and reset its state.
Safe to call multiple times.
"""
function cleanup_petsc_cache!(pcache::PETScCache)
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
        pcache.petsc_P !== nothing && pcache.petsc_P !== pcache.petsc_A &&
            PETSc.destroy(pcache.petsc_P)
        pcache.petsc_A !== nothing && PETSc.destroy(pcache.petsc_A)
    catch
    end
    _nullify_all!(pcache)
end

cleanup_petsc_cache!(cache::LinearCache)  = cleanup_petsc_cache!(cache.cacheval)
cleanup_petsc_cache!(sol::LinearSolution) = cleanup_petsc_cache!(sol.cache.cacheval)

# ── Cache init ────────────────────────────────────────────────────────────────

function LinearSolve.init_cacheval(alg::PETScAlgorithm, A, b, u, Pl, Pr, maxiters::Int,
                                   abstol, reltol, verbose::Union{LinearVerbosity,Bool},
                                   assumptions::OperatorAssumptions)
    T = eltype(A)
    pcache = PETScCache{T}(nothing, nothing, nothing, nothing, nothing, nothing,
                           nothing, nothing, 0, 0, 0, UInt(0), false,
                           nothing, nothing, nothing)
    finalizer(cleanup_petsc_cache!, pcache)
    return pcache
end

# ── Matrix Helpers ────────────────────────────────────────────────────────────

function to_petsc_mat_seq(petsclib, A)
    if A isa SparseMatrixCSC
        return PETSc.MatSeqAIJWithArrays(petsclib, MPI.COMM_SELF, A)
    else
        return PETSc.MatSeqDense(petsclib, copy(A))
    end
end

function to_petsc_mat_mpi(petsclib, A, comm)
    A isa SparseMatrixCSC || (A = sparse(A))
    M, N = size(A)
    PA = LibPETSc.MatCreate(petsclib, comm)
    LibPETSc.MatSetSizes(petsclib, PA,
        petsclib.PetscInt(-1), petsclib.PetscInt(-1),
        petsclib.PetscInt(M),  petsclib.PetscInt(N))
    LibPETSc.MatSetFromOptions(petsclib, PA)
    LibPETSc.MatSetUp(petsclib, PA)

    rstart, rend = Int.(LibPETSc.MatGetOwnershipRange(petsclib, PA))

    @inbounds for col in 1:N
        for idx in nzrange(A, col)
            row = A.rowval[idx]
            (rstart <= row - 1 < rend) && (PA[row, col] = A.nzval[idx])
        end
    end
    PETSc.assemble!(PA)
    return PA, rstart, rend
end

function to_petsc_mat(petsclib, A, comm)
    is_parallel(comm) ? to_petsc_mat_mpi(petsclib, A, comm) :
                       (to_petsc_mat_seq(petsclib, A), 0, size(A, 1))
end


function update_mat_values!(petsclib, PA, A::SparseMatrixCSC, rstart, rend)
    N = size(A, 2)
    @inbounds for col in 1:N
        for idx in nzrange(A, col)
            row = A.rowval[idx]
            (rstart <= row - 1 < rend) && (PA[row, col] = A.nzval[idx])
        end
    end
    PETSc.assemble!(PA)
end

# ── Vec read/write ────────────────────────────────────────────────────────────

function write_local_values!(pv, src::AbstractVector, petsclib, rstart::Int, rend::Int)
    local_n = rend - rstart
    ptr = LibPETSc.VecGetArray(petsclib, pv)
    @inbounds for i in 1:local_n
        ptr[i] = src[rstart + i]
    end
    LibPETSc.VecRestoreArray(petsclib, pv, ptr)
end

function read_local_values!(dst::AbstractVector, pv, petsclib, rstart::Int, rend::Int)
    local_n = rend - rstart
    ptr = LibPETSc.VecGetArrayRead(petsclib, pv)
    @inbounds for i in 1:local_n
        dst[rstart + i] = ptr[i]
    end
    LibPETSc.VecRestoreArrayRead(petsclib, pv, ptr)
end

# ── Distributed vectors ───────────────────────────────────────────────────────

function create_distributed_vec(petsclib, v::AbstractVector, rstart, rend, comm)
    local_n = rend - rstart
    pv = LibPETSc.VecCreateMPI(petsclib, comm,
             petsclib.PetscInt(local_n), petsclib.PetscInt(length(v)))
    write_local_values!(pv, v, petsclib, rstart, rend)
    return pv
end

function ensure_distributed_vecs!(pcache, petsclib, n, u, b, rstart, rend, comm)
    if pcache.vec_n != n || pcache.rstart != rstart || pcache.rend != rend ||
       pcache.petsc_x === nothing || pcache.petsc_b === nothing
        pcache.petsc_x !== nothing && PETSc.destroy(pcache.petsc_x)
        pcache.petsc_b !== nothing && PETSc.destroy(pcache.petsc_b)
        pcache.petsc_x = create_distributed_vec(petsclib, u, rstart, rend, comm)
        pcache.petsc_b = create_distributed_vec(petsclib, b, rstart, rend, comm)
        pcache.vec_n   = n
        pcache.rstart  = rstart
        pcache.rend    = rend
        pcache.mpi_counts = pcache.mpi_displs = pcache.local_buf = nothing
    end
    return pcache.petsc_x, pcache.petsc_b
end


# ── Gather solution ───────────────────────────────────────────────────────────

function setup_mpi_gather!(pcache::PETScCache{T}) where T
    pcache.mpi_counts !== nothing && return
    local_n = pcache.rend - pcache.rstart
    counts = MPI.Allgather(Int32(local_n), pcache.comm)
    displs = cumsum([Int32(0); counts[1:end-1]])
    pcache.mpi_counts = counts
    pcache.mpi_displs = displs
    pcache.local_buf  = Vector{T}(undef, local_n)
end

function gather_solution!(dst::AbstractVector, petsc_v, pcache::PETScCache{T}) where T
    rstart, rend = pcache.rstart, pcache.rend
    petsclib = pcache.petsclib

    if !is_parallel(pcache.comm)
        read_local_values!(dst, petsc_v, petsclib, rstart, rend)
        return dst
    end

    setup_mpi_gather!(pcache)
    local_buf = pcache.local_buf::Vector{T}
    local_n   = rend - rstart

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

function build_nullspace(petsclib, alg::PETScAlgorithm, comm)
    if alg.nullspace === :none
        return nothing
    elseif alg.nullspace === :constant
        return LibPETSc.MatNullSpaceCreate(petsclib, comm, LibPETSc.PetscBool(true), 0, LibPETSc.PetscVec[])
    else
        PScalar = petsclib.PetscScalar
        petsc_vecs = LibPETSc.PetscVec[
            create_distributed_vec(petsclib, PScalar.(v), 0, length(v), comm)
            for v in alg.nullspace_vecs
        ]
        ns = LibPETSc.MatNullSpaceCreate(petsclib, comm, LibPETSc.PetscBool(false), length(petsc_vecs), petsc_vecs)
        foreach(PETSc.destroy, petsc_vecs)
        return ns
    end
end

function attach_nullspace!(petsclib, ksp, petsc_A, ns)
    ns === nothing && return
    LibPETSc.MatSetNullSpace(petsclib, petsc_A, ns)
end

# ── KSP build ─────────────────────────────────────────────────────────────────

function build_ksp!(pcache, petsclib, cache, alg, comm)

    pcache.petsc_A, pcache.rstart, pcache.rend = to_petsc_mat(petsclib, cache.A, comm)
    pcache.sparsity_hash = sparsity_fingerprint(cache.A)

    pcache.petsc_P = alg.prec_matrix === nothing ? pcache.petsc_A :
                     to_petsc_mat(petsclib, alg.prec_matrix, comm)[1]

    pcache.ksp = PETSc.KSP(pcache.petsc_A, pcache.petsc_P;
                           ksp_type   = string(alg.solver_type),
                           pc_type    = string(alg.pc_type),
                           ksp_rtol   = cache.reltol,
                           ksp_atol   = cache.abstol,
                           ksp_max_it = cache.maxiters,
                           alg.ksp_options...)

    if alg.initial_guess_nonzero
        LibPETSc.KSPSetInitialGuessNonzero(petsclib, pcache.ksp, LibPETSc.PetscBool(true))
    end

    pcache.nullspace_obj = alg.nullspace !== :none ?
        build_nullspace(petsclib, alg, comm) : nothing
    attach_nullspace!(petsclib, pcache.ksp, pcache.petsc_A, pcache.nullspace_obj)
end

# ── Solve ─────────────────────────────────────────────────────────────────────

function SciMLBase.solve!(cache::LinearCache, alg::PETScAlgorithm; kwargs...)

    pcache = cache.cacheval
    comm   = resolve_comm(alg)
    petsclib = pcache.petsclib === nothing ? get_petsclib(eltype(cache.A)) : pcache.petsclib
    PETSc.initialized(petsclib) || PETSc.initialize(petsclib)
    pcache.petsclib    = petsclib
    pcache.comm        = comm
    pcache.initialized = true

    # ── Decide whether to rebuild KSP or just update values ──────────────────────
    rebuild_ksp = false
    if pcache.ksp === nothing
        rebuild_ksp = true
    elseif cache.isfresh
        # Only rebuild if sparsity changed
        new_hash = sparsity_fingerprint(cache.A)
        sparse_same = (cache.A isa SparseMatrixCSC) && (new_hash == pcache.sparsity_hash)
        rebuild_ksp = !sparse_same
    end

    # ── Build KSP if needed ─────────────────────────────────────────────────────
    if rebuild_ksp
        build_ksp!(pcache, petsclib, cache, alg, comm)
        # Update sparsity hash immediately after KSP creation
        pcache.sparsity_hash = sparsity_fingerprint(cache.A)
    else
        # Same sparsity → update values only
        update_mat_values!(petsclib, pcache.petsc_A, cache.A, pcache.rstart, pcache.rend)
        if alg.prec_matrix !== nothing
            update_mat_values!(petsclib, pcache.petsc_P, alg.prec_matrix, pcache.rstart, pcache.rend)
        end
        if !cache.precsisfresh
            LibPETSc.KSPSetReusePreconditioner(petsclib, pcache.ksp, LibPETSc.PetscBool(true))
        end
    end
    cache.isfresh = false

    # ── Ensure distributed vectors ─────────────────────────────────────────────
    petsc_x, petsc_b = ensure_distributed_vecs!(pcache, petsclib, length(cache.b),
                                                cache.u, cache.b, pcache.rstart,
                                                pcache.rend, comm)

    write_local_values!(petsc_x, cache.u, petsclib, pcache.rstart, pcache.rend)
    write_local_values!(petsc_b, cache.b, petsclib, pcache.rstart, pcache.rend)

    # ── Solve ──────────────────────────────────────────────────────────────────
    if alg.transposed
        LibPETSc.KSPSolveTranspose(petsclib, pcache.ksp, petsc_b, petsc_x)
    else
        LibPETSc.KSPSolve(petsclib, pcache.ksp, petsc_b, petsc_x)
    end

    gather_solution!(cache.u, petsc_x, pcache)

    # ── Collect solver info ────────────────────────────────────────────────────
    iters  = Int(LibPETSc.KSPGetIterationNumber(petsclib, pcache.ksp))
    reason = Int(LibPETSc.KSPGetConvergedReason(petsclib, pcache.ksp))
    resid  = Float64(LibPETSc.KSPGetResidualNorm(petsclib, pcache.ksp))
    retcode = reason > 0 ? ReturnCode.Success :
              reason == 0 ? ReturnCode.Default : ReturnCode.Failure

    return build_linear_solution(alg, cache.u, resid, cache; retcode, iters)
end

end