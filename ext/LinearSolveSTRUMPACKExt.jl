module LinearSolveSTRUMPACKExt

using LinearSolve: LinearSolve, LinearVerbosity, OperatorAssumptions
using SparseArrays: SparseArrays, AbstractSparseMatrixCSC, getcolptr, rowvals, nonzeros
using SciMLBase: SciMLBase, ReturnCode
using SciMLLogging: @SciMLMessage
using Libdl: Libdl

const STRUMPACK_SUCCESS = Cint(0)
const STRUMPACK_MATRIX_NOT_SET = Cint(1)
const STRUMPACK_REORDERING_ERROR = Cint(2)
const STRUMPACK_ZERO_PIVOT = Cint(3)
const STRUMPACK_NO_CONVERGENCE = Cint(4)
const STRUMPACK_INACCURATE_INERTIA = Cint(5)

const STRUMPACK_DOUBLE = Cint(1)
const STRUMPACK_MT = Cint(0)

const _libstrumpack = Ref{Ptr{Cvoid}}(C_NULL)

function _load_libstrumpack()
    for name in (
            "libstrumpack.so",
            "libstrumpack.so.8",
            "libstrumpack.so.7",
            "libstrumpack.dylib",
            "strumpack",
        )
        handle = Libdl.dlopen_e(name)
        handle != C_NULL && return handle
    end
    return C_NULL
end

function __init__()
    return _libstrumpack[] = _load_libstrumpack()
end

strumpack_isavailable() = _libstrumpack[] != C_NULL

mutable struct STRUMPACKCache
    solver::Ref{Ptr{Cvoid}}
    rowptr::Vector{Int32}
    colind::Vector{Int32}
    nzval::Vector{Float64}

    function STRUMPACKCache()
        cache = new(Ref{Ptr{Cvoid}}(C_NULL), Int32[], Int32[], Float64[])
        finalizer(_strumpack_destroy!, cache)
        return cache
    end
end

function _strumpack_destroy!(cache::STRUMPACKCache)
    _libstrumpack[] == C_NULL && return
    cache.solver[] == C_NULL && return
    ccall((:STRUMPACK_destroy, _libstrumpack[]), Cvoid, (Ref{Ptr{Cvoid}},), cache.solver)
    cache.solver[] = C_NULL
    return
end

function _ensure_initialized!(cache::STRUMPACKCache)
    cache.solver[] != C_NULL && return
    ccall(
        (:STRUMPACK_init_mt, _libstrumpack[]),
        Cvoid,
        (Ref{Ptr{Cvoid}}, Cint, Cint, Cint, Ptr{Ptr{UInt8}}, Cint),
        cache.solver,
        STRUMPACK_DOUBLE,
        STRUMPACK_MT,
        Cint(0),
        Ptr{Ptr{UInt8}}(C_NULL),
        Cint(0)
    )
    return
end

function _csc_to_csr_0based(A::AbstractSparseMatrixCSC)
    n = size(A, 1)
    colptr = getcolptr(A)
    rowval = rowvals(A)
    vals = nonzeros(A)

    nnz = length(vals)
    rowptr = zeros(Int32, n + 1)

    @inbounds for idx in eachindex(rowval)
        rowptr[Int(rowval[idx]) + 1] += 1
    end

    @inbounds for i in 1:n
        rowptr[i + 1] += rowptr[i]
    end

    nextidx = copy(rowptr)
    colind = Vector{Int32}(undef, nnz)
    outvals = Vector{Float64}(undef, nnz)

    @inbounds for j in 1:size(A, 2)
        for p in colptr[j]:(colptr[j + 1] - 1)
            row = Int(rowval[p])
            pos = Int(nextidx[row] + 1)
            nextidx[row] += 1
            colind[pos] = Int32(j - 1)
            outvals[pos] = Float64(vals[p])
        end
    end

    return rowptr, colind, outvals
end

function _retcode_from_strumpack(info::Cint)
    return if info == STRUMPACK_SUCCESS
        ReturnCode.Success
    elseif info == STRUMPACK_ZERO_PIVOT
        ReturnCode.Infeasible
    elseif info == STRUMPACK_NO_CONVERGENCE
        ReturnCode.ConvergenceFailure
    elseif info == STRUMPACK_INACCURATE_INERTIA
        ReturnCode.Unstable
    elseif info == STRUMPACK_MATRIX_NOT_SET || info == STRUMPACK_REORDERING_ERROR
        ReturnCode.Failure
    else
        ReturnCode.Failure
    end
end

function LinearSolve.init_cacheval(
        ::LinearSolve.STRUMPACKFactorization,
        A::AbstractSparseMatrixCSC{<:AbstractFloat}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return STRUMPACKCache()
end

function LinearSolve.init_cacheval(
        ::LinearSolve.STRUMPACKFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache,
        alg::LinearSolve.STRUMPACKFactorization;
        kwargs...
    )
    if _libstrumpack[] == C_NULL
        error("STRUMPACKFactorization requires a discoverable STRUMPACK shared library (`libstrumpack`)")
    end

    A = convert(AbstractMatrix, cache.A)
    if !(A isa AbstractSparseMatrixCSC)
        error("STRUMPACKFactorization currently supports only sparse CSC matrices")
    end
    size(A, 1) == size(A, 2) || error("STRUMPACKFactorization requires a square matrix")

    scache = LinearSolve.@get_cacheval(cache, :STRUMPACKFactorization)
    if scache === nothing
        error("STRUMPACKFactorization currently supports `AbstractSparseMatrixCSC{<:AbstractFloat}`")
    end

    _ensure_initialized!(scache)

    if cache.isfresh
        scache.rowptr, scache.colind, scache.nzval = _csc_to_csr_0based(A)
        ccall(
            (:STRUMPACK_set_csr_matrix, _libstrumpack[]),
            Cvoid,
            (Ptr{Cvoid}, Cint, Ref{Cint}, Ref{Cint}, Ref{Cdouble}, Cint),
            scache.solver[],
            Cint(size(A, 1)),
            scache.rowptr,
            scache.colind,
            scache.nzval,
            Cint(0)
        )

        info = ccall((:STRUMPACK_factor, _libstrumpack[]), Cint, (Ptr{Cvoid},), scache.solver[])
        if info != STRUMPACK_SUCCESS
            @SciMLMessage(
                "STRUMPACK factorization failed (code $(Int(info)))",
                cache.verbose,
                :solver_failure
            )
            cache.isfresh = false
            return SciMLBase.build_linear_solution(
                alg,
                cache.u,
                nothing,
                cache;
                retcode = _retcode_from_strumpack(info)
            )
        end
        cache.isfresh = false
    end

    bvec = Float64.(cache.b)
    xvec = Float64.(cache.u)

    info = ccall(
        (:STRUMPACK_solve, _libstrumpack[]),
        Cint,
        (Ptr{Cvoid}, Ref{Cdouble}, Ref{Cdouble}, Cint),
        scache.solver[],
        bvec,
        xvec,
        Cint(alg.use_initial_guess)
    )

    if info != STRUMPACK_SUCCESS
        @SciMLMessage(
            "STRUMPACK solve failed (code $(Int(info)))",
            cache.verbose,
            :solver_failure
        )
        return SciMLBase.build_linear_solution(
            alg,
            cache.u,
            nothing,
            cache;
            retcode = _retcode_from_strumpack(info)
        )
    end

    copyto!(cache.u, xvec)
    return SciMLBase.build_linear_solution(
        alg,
        cache.u,
        nothing,
        cache;
        retcode = ReturnCode.Success
    )
end

end
