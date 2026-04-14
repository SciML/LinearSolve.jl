module LinearSolveSTRUMPACKExt

using LinearSolve: LinearSolve, LinearVerbosity, OperatorAssumptions
using SparseArrays: SparseArrays, AbstractSparseMatrixCSC, getcolptr, rowvals, nonzeros
using SciMLBase: SciMLBase, ReturnCode
using SciMLLogging: @SciMLMessage
using STRUMPACK_jll: libstrumpack
using Libdl: Libdl

const STRUMPACK_SUCCESS = Cint(0)
const STRUMPACK_MATRIX_NOT_SET = Cint(1)
const STRUMPACK_REORDERING_ERROR = Cint(2)
const STRUMPACK_ZERO_PIVOT = Cint(3)
const STRUMPACK_NO_CONVERGENCE = Cint(4)
const STRUMPACK_INACCURATE_INERTIA = Cint(5)

const STRUMPACK_DOUBLE = Cint(1)
const STRUMPACK_MT = Cint(0)

struct STRUMPACKSparseSolver
    solver::Ptr{Cvoid}
    precision::Cint
    interface::Cint
end

const _libstrumpack = Ref{Union{Nothing, String}}(nothing)

function _load_libstrumpack()
    if libstrumpack isa AbstractString
        handle = Libdl.dlopen_e(libstrumpack)
        handle != C_NULL && return String(libstrumpack)
    elseif libstrumpack isa Ptr
        libpath = try
            Libdl.dlpath(libstrumpack)
        catch
            nothing
        end
        if libpath !== nothing
            handle = Libdl.dlopen_e(libpath)
            handle != C_NULL && return String(libpath)
        end
    end

    for name in (
            "libstrumpack.so",
            "libstrumpack.so.8",
            "libstrumpack.so.7",
            "libstrumpack.dylib",
            "strumpack",
        )
        handle = Libdl.dlopen_e(name)
        handle != C_NULL && return String(name)
    end
    return nothing
end

function __init__()
    return _libstrumpack[] = _load_libstrumpack()
end

strumpack_isavailable() = _libstrumpack[] !== nothing

mutable struct STRUMPACKCache
    solver::Ref{STRUMPACKSparseSolver}
    rowptr::Vector{Int32}
    colind::Vector{Int32}
    nzval::Vector{Float64}
    option_storage::Vector{Vector{UInt8}}
    option_ptrs::Vector{Ptr{UInt8}}

    function STRUMPACKCache()
        cache = new(
            Ref(STRUMPACKSparseSolver(C_NULL, 0, 0)),
            Int32[],
            Int32[],
            Float64[],
            Vector{Vector{UInt8}}(),
            Ptr{UInt8}[]
        )
        finalizer(_strumpack_destroy!, cache)
        return cache
    end
end

function _strumpack_destroy!(cache::STRUMPACKCache)
    _libstrumpack[] === nothing && return
    cache.solver[].solver == C_NULL && return
    ccall((:STRUMPACK_destroy, _libstrumpack[]), Cvoid, (Ref{STRUMPACKSparseSolver},), cache.solver)
    cache.solver[] = STRUMPACKSparseSolver(C_NULL, 0, 0)
    return
end

function _set_runtime_options!(cache::STRUMPACKCache, alg::LinearSolve.STRUMPACKFactorization)
    empty!(cache.option_storage)
    empty!(cache.option_ptrs)

    for opt in alg.options
        bytes = Vector{UInt8}(codeunits(opt))
        push!(bytes, 0x00)
        push!(cache.option_storage, bytes)
        push!(cache.option_ptrs, pointer(bytes))
    end

    return
end

function _ensure_initialized!(cache::STRUMPACKCache, alg::LinearSolve.STRUMPACKFactorization)
    cache.solver[].solver != C_NULL && return
    _set_runtime_options!(cache, alg)

    argc = Cint(length(cache.option_ptrs))
    argv = isempty(cache.option_ptrs) ? Ptr{Ptr{UInt8}}(C_NULL) : Ptr{Ptr{UInt8}}(pointer(cache.option_ptrs))

    ccall(
        (:STRUMPACK_init_mt, _libstrumpack[]),
        Cvoid,
        (Ref{STRUMPACKSparseSolver}, Cint, Cint, Cint, Ptr{Ptr{UInt8}}, Cint),
        cache.solver,
        STRUMPACK_DOUBLE,
        STRUMPACK_MT,
        argc,
        argv,
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
    if _libstrumpack[] === nothing
        error("STRUMPACKFactorization requires `using SparseArrays` and loading `STRUMPACK_jll` (for example `import STRUMPACK_jll`)")
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

    _ensure_initialized!(scache, alg)

    if cache.isfresh
        scache.rowptr, scache.colind, scache.nzval = _csc_to_csr_0based(A)
        nref = Ref{Cint}(Cint(size(A, 1)))
        ccall(
            (:STRUMPACK_set_csr_matrix, _libstrumpack[]),
            Cvoid,
            (STRUMPACKSparseSolver, Ref{Cint}, Ref{Cint}, Ref{Cint}, Ref{Cdouble}, Cint),
            scache.solver[],
            nref,
            scache.rowptr,
            scache.colind,
            scache.nzval,
            Cint(0)
        )

        info = ccall((:STRUMPACK_factor, _libstrumpack[]), Cint, (STRUMPACKSparseSolver,), scache.solver[])
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
        (STRUMPACKSparseSolver, Ref{Cdouble}, Ref{Cdouble}, Cint),
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
