module LinearSolveSuperLUDISTExt

using LinearAlgebra
using LinearSolve
using LinearSolve: LinearVerbosity, OperatorAssumptions
using SparseArrays
using SuperLUDIST
import SciMLBase
import SciMLBase: ReturnCode, LinearSolution
using SciMLLogging: @SciMLMessage

const MPI = SuperLUDIST.MPI
const SparseBase = SuperLUDIST.SparseBase
const CIndex = SuperLUDIST.CIndex

mutable struct SuperLUDISTCache
    factor::Any

    function SuperLUDISTCache()
        return new(nothing)
    end
end

cleanup_superludist_cache!(cache::SuperLUDISTCache) = (cache.factor = nothing; cache)
cleanup_superludist_cache!(cache::LinearSolve.LinearCache) = cleanup_superludist_cache!(cache.cacheval)

LinearSolve.needs_concrete_A(::LinearSolve.SuperLUDISTFactorization) = true

function LinearSolve.init_cacheval(
        alg::LinearSolve.SuperLUDISTFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return SuperLUDISTCache()
end

_superlu_eltype(::Type{Float32}) = Float32
_superlu_eltype(::Type{Float64}) = Float64
function _superlu_eltype(::Type{T}) where {T}
    throw(
        ArgumentError(
            "SuperLUDISTFactorization only supports Float32 and Float64 inputs; got element type $T"
        )
    )
end

function _superlu_index_type(A::SparseMatrixCSC)
    limit = max(size(A, 1), size(A, 2), nnz(A) + 1)
    return limit <= typemax(Int32) ? Int32 : Int64
end

function _grid_dims(nprocs::Integer, nprow::Integer, npcol::Integer)
    if nprow == 0 && npcol == 0
        root = floor(Int, sqrt(nprocs))
        for r in root:-1:1
            if nprocs % r == 0
                return r, nprocs ÷ r
            end
        end
    elseif nprow == 0
        nprocs % npcol == 0 || error("npcol=$npcol does not divide communicator size $nprocs")
        return nprocs ÷ npcol, npcol
    elseif npcol == 0
        nprocs % nprow == 0 || error("nprow=$nprow does not divide communicator size $nprocs")
        return nprow, nprocs ÷ nprow
    else
        nprow * npcol == nprocs || error(
            "nprow*npcol must equal communicator size ($nprow * $npcol != $nprocs)"
        )
        return nprow, npcol
    end
    error("could not determine a valid SuperLU_DIST process grid")
end

function _build_options(opt)
    options = SuperLUDIST.Options()
    if opt === nothing
        return options
    elseif opt isa NamedTuple
        for (name, value) in pairs(opt)
            setproperty!(options, name, value)
        end
        return options
    elseif opt isa SuperLUDIST.Options
        return deepcopy(opt)
    else
        throw(
            ArgumentError(
                "options must be nothing, a NamedTuple, or a SuperLUDIST.Options object; got $(typeof(opt))"
            )
        )
    end
end

function _to_superlu_store(A::SparseMatrixCSC{T}, ::Type{I}) where {T, I}
    ptr = CIndex{I}.(A.colptr)
    idx = CIndex{I}.(A.rowval)
    vals = Vector{T}(A.nzval)
    return SparseBase.CSCStore(ptr, idx, vals, size(A))
end

function _to_superlu_matrix(A::SparseMatrixCSC)
    T = _superlu_eltype(eltype(A))
    I = _superlu_index_type(A)
    Ac = SparseMatrixCSC{T, Int}(A)
    store = _to_superlu_store(Ac, I)
    return Ac, store, I
end

function _rhs_matrix(b, ::Type{T}) where {T}
    if b isa AbstractVector
        return reshape(T.(collect(b)), :, 1), true
    else
        return Matrix{T}(b), false
    end
end

function _copy_solution!(u::AbstractVector, x::AbstractMatrix)
    copyto!(u, vec(x))
    return u
end

function _copy_solution!(u::AbstractMatrix, x::AbstractMatrix)
    copyto!(u, x)
    return u
end

function _solve_failed_solution(
        alg::LinearSolve.SuperLUDISTFactorization,
        cache::LinearSolve.LinearCache,
        msg::AbstractString
    )
    @SciMLMessage(msg, cache.verbose, :solver_failure)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
    )
end

function _build_factorization(
        alg::LinearSolve.SuperLUDISTFactorization,
        A::SparseMatrixCSC,
        bmat,
    )
    Ac, store, I = _to_superlu_matrix(A)
    comm = alg.comm === nothing ? MPI.COMM_SELF : alg.comm
    nprocs = MPI.Comm_size(comm)
    nprow, npcol = _grid_dims(nprocs, alg.nprow, alg.npcol)
    alg.threads === nothing || SuperLUDIST.superlu_set_num_threads(I, alg.threads)
    grid = SuperLUDIST.Grid{I}(nprow, npcol, comm)
    Aslu = SuperLUDIST.ReplicatedSuperMatrix(store, grid)
    options = _build_options(alg.options)
    x, factor = SuperLUDIST.pgssvx!(Aslu, copy(bmat); options = options)
    return x, factor, Ac
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::LinearSolve.SuperLUDISTFactorization;
        kwargs...
    )
    A = convert(AbstractMatrix, cache.A)
    A_sparse = A isa SparseMatrixCSC ? A : sparse(A)
    T = try
        _superlu_eltype(promote_type(eltype(A_sparse), eltype(cache.b)))
    catch err
        return _solve_failed_solution(alg, cache, sprint(showerror, err))
    end
    bmat, _ = _rhs_matrix(cache.b, T)
    scache = LinearSolve.@get_cacheval(cache, :SuperLUDISTFactorization)

    x = nothing
    if cache.isfresh || scache.factor === nothing
        cleanup_superludist_cache!(scache)
        result = try
            _build_factorization(alg, SparseMatrixCSC{T, Int}(A_sparse), bmat)
        catch err
            return _solve_failed_solution(alg, cache, sprint(showerror, err))
        end
        x, scache.factor = result[1], result[2]
        cache.isfresh = false
    else
        x = try
            y = copy(bmat)
            ldiv!(scache.factor, y)
            y
        catch err
            return _solve_failed_solution(alg, cache, sprint(showerror, err))
        end
    end

    _copy_solution!(cache.u, x)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Success
    )
end

end
