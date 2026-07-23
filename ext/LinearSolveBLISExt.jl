module LinearSolveBLISExt

using Libdl
using blis_jll
using LAPACK_jll
using LinearAlgebra
using LinearSolve

using LinearAlgebra: BlasInt
using LinearAlgebra.LAPACK: require_one_based_indexing, chkfinite, chkstride1,
    @blasfunc, chkargsok
using LinearSolve: BLISLUFactorization, @get_cacheval, LinearCache, SciMLBase, LinearVerbosity, get_blas_operation_info, blas_info_msg
using SciMLLogging: SciMLLogging, @SciMLMessage
using SciMLBase: ReturnCode

const global libblis = blis_jll.blis
const global liblapack = LAPACK_jll.liblapack

# Resolve Julia 1.13 lazy JLL products once so solves call fixed function pointers.
const _lapack_handle = Ref{Ptr{Cvoid}}(C_NULL)
const _lapack_zgetrf = Ref{Ptr{Cvoid}}(C_NULL)
const _lapack_cgetrf = Ref{Ptr{Cvoid}}(C_NULL)
const _lapack_dgetrf = Ref{Ptr{Cvoid}}(C_NULL)
const _lapack_sgetrf = Ref{Ptr{Cvoid}}(C_NULL)
const _lapack_zgetrs = Ref{Ptr{Cvoid}}(C_NULL)
const _lapack_cgetrs = Ref{Ptr{Cvoid}}(C_NULL)
const _lapack_dgetrs = Ref{Ptr{Cvoid}}(C_NULL)
const _lapack_sgetrs = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    @static if VERSION >= v"1.13.0-DEV.0"
        handle = Libdl.dlopen(liblapack)
        _lapack_handle[] = handle
        _lapack_zgetrf[] = Libdl.dlsym(handle, @blasfunc(zgetrf_))
        _lapack_cgetrf[] = Libdl.dlsym(handle, @blasfunc(cgetrf_))
        _lapack_dgetrf[] = Libdl.dlsym(handle, @blasfunc(dgetrf_))
        _lapack_sgetrf[] = Libdl.dlsym(handle, @blasfunc(sgetrf_))
        _lapack_zgetrs[] = Libdl.dlsym(handle, @blasfunc(zgetrs_))
        _lapack_cgetrs[] = Libdl.dlsym(handle, @blasfunc(cgetrs_))
        _lapack_dgetrs[] = Libdl.dlsym(handle, @blasfunc(dgetrs_))
        _lapack_sgetrs[] = Libdl.dlsym(handle, @blasfunc(sgetrs_))
    end
    return nothing
end

macro _lapack_function(symbol, pointer)
    if VERSION >= v"1.13.0-DEV.0"
        return :($(esc(pointer))[])
    end
    return :(($(esc(symbol)), liblapack))
end

LinearSolve.useblis(x::Nothing) = true

@inline function getrf!(
        A::AbstractMatrix{<:ComplexF64}, ipiv::AbstractVector{BlasInt},
        info::Ref{BlasInt}, check::Bool
    )
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        @_lapack_function(@blasfunc(zgetrf_), _lapack_zgetrf), Cvoid,
        (
            Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
        ),
        m, n, A, lda, ipiv, info
    )
    chkargsok(info[])
    return info[]
end

@inline function getrf!(
        A::AbstractMatrix{<:ComplexF32}, ipiv::AbstractVector{BlasInt},
        info::Ref{BlasInt}, check::Bool
    )
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        @_lapack_function(@blasfunc(cgetrf_), _lapack_cgetrf), Cvoid,
        (
            Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
        ),
        m, n, A, lda, ipiv, info
    )
    chkargsok(info[])
    return info[]
end

@inline function getrf!(
        A::AbstractMatrix{<:Float64}, ipiv::AbstractVector{BlasInt},
        info::Ref{BlasInt}, check::Bool
    )
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        @_lapack_function(@blasfunc(dgetrf_), _lapack_dgetrf), Cvoid,
        (
            Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
        ),
        m, n, A, lda, ipiv, info
    )
    chkargsok(info[])
    return info[]
end

@inline function getrf!(
        A::AbstractMatrix{<:Float32}, ipiv::AbstractVector{BlasInt},
        info::Ref{BlasInt}, check::Bool
    )
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        @_lapack_function(@blasfunc(sgetrf_), _lapack_sgetrf), Cvoid,
        (
            Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
        ),
        m, n, A, lda, ipiv, info
    )
    chkargsok(info[])
    return info[]
end

function getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF64},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:ComplexF64},
        info::Ref{BlasInt}
    )
    require_one_based_indexing(A, ipiv, B)
    LinearAlgebra.LAPACK.chktrans(trans)
    chkstride1(A, B, ipiv)
    n = LinearAlgebra.checksquare(A)
    if n != size(B, 1)
        throw(DimensionMismatch("B has leading dimension $(size(B, 1)), but needs $n"))
    end
    if n != length(ipiv)
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs to be $n"))
    end
    nrhs = size(B, 2)
    ccall(
        @_lapack_function(@blasfunc(zgetrs_), _lapack_zgetrs), Cvoid,
        (
            Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt}, Ptr{BlasInt}, Clong,
        ),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1
    )
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    return B
end

function getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF32},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:ComplexF32},
        info::Ref{BlasInt}
    )
    require_one_based_indexing(A, ipiv, B)
    LinearAlgebra.LAPACK.chktrans(trans)
    chkstride1(A, B, ipiv)
    n = LinearAlgebra.checksquare(A)
    if n != size(B, 1)
        throw(DimensionMismatch("B has leading dimension $(size(B, 1)), but needs $n"))
    end
    if n != length(ipiv)
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs to be $n"))
    end
    nrhs = size(B, 2)
    ccall(
        @_lapack_function(@blasfunc(cgetrs_), _lapack_cgetrs), Cvoid,
        (
            Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF32}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{ComplexF32}, Ref{BlasInt}, Ptr{BlasInt}, Clong,
        ),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1
    )
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    return B
end

function getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:Float64},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:Float64},
        info::Ref{BlasInt}
    )
    require_one_based_indexing(A, ipiv, B)
    LinearAlgebra.LAPACK.chktrans(trans)
    chkstride1(A, B, ipiv)
    n = LinearAlgebra.checksquare(A)
    if n != size(B, 1)
        throw(DimensionMismatch("B has leading dimension $(size(B, 1)), but needs $n"))
    end
    if n != length(ipiv)
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs to be $n"))
    end
    nrhs = size(B, 2)
    ccall(
        @_lapack_function(@blasfunc(dgetrs_), _lapack_dgetrs), Cvoid,
        (
            Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Clong,
        ),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1
    )
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    return B
end

function getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:Float32},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:Float32},
        info::Ref{BlasInt}
    )
    require_one_based_indexing(A, ipiv, B)
    LinearAlgebra.LAPACK.chktrans(trans)
    chkstride1(A, B, ipiv)
    n = LinearAlgebra.checksquare(A)
    if n != size(B, 1)
        throw(DimensionMismatch("B has leading dimension $(size(B, 1)), but needs $n"))
    end
    if n != length(ipiv)
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs to be $n"))
    end
    nrhs = size(B, 2)
    ccall(
        @_lapack_function(@blasfunc(sgetrs_), _lapack_sgetrs), Cvoid,
        (
            Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{Float32}, Ref{BlasInt}, Ptr{BlasInt}, Clong,
        ),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1
    )
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    return B
end

default_alias_A(::BLISLUFactorization, ::Any, ::Any) = false
default_alias_b(::BLISLUFactorization, ::Any, ::Any) = false

mutable struct BLISLUCache{F, P, I}
    factors::F
    ipiv::P
    info::I
end

LinearSolve._custom_cache_factorization(::BLISLUFactorization, cacheval::BLISLUCache) =
    LinearAlgebra.LU(cacheval.factors, cacheval.ipiv, Int(cacheval.info[]))

@inline function LinearSolve._direct_lu_factorize!(
        cacheval::BLISLUCache, A_work, ::BLISLUFactorization
    )
    cacheval.factors = A_work
    return getrf!(A_work, cacheval.ipiv, cacheval.info, false)
end

@inline function LinearSolve._direct_lu_solve!(
        cacheval::BLISLUCache, u, b, ::BLISLUFactorization
    )
    copyto!(u, b)
    getrs!('N', cacheval.factors, cacheval.ipiv, u, cacheval.info)
    return u
end

function LinearSolve.init_cacheval(
        alg::BLISLUFactorization,
        A::DenseMatrix{<:Union{Float32, Float64, ComplexF32, ComplexF64}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return BLISLUCache(
        A, similar(A, BlasInt, min(size(A, 1), size(A, 2))), Ref{BlasInt}()
    )
end

function SciMLBase.solve!(
        cache::LinearCache, alg::BLISLUFactorization;
        kwargs...
    )
    A_work = convert(AbstractMatrix, cache.A)
    verbose = cache.verbose
    if cache.isfresh
        cacheval = @get_cacheval(cache, :BLISLUFactorization)
        if length(cacheval.ipiv) != min(size(A_work, 1), size(A_work, 2))
            cacheval.ipiv = similar(
                A_work, BlasInt, min(size(A_work, 1), size(A_work, 2))
            )
        end
        info_value = LinearSolve._direct_lu_factorize!(cacheval, A_work, alg)

        if info_value != 0
            if verbose.blas_info != SciMLLogging.Silent() || verbose.blas_errors != SciMLLogging.Silent() ||
                    verbose.blas_invalid_args != SciMLLogging.Silent()
                failure_op_info = get_blas_operation_info(
                    :dgetrf, A_work, cache.b,
                    condition = verbose.condition_number != SciMLLogging.Silent()
                )
                let op_info = failure_op_info
                    @SciMLMessage(cache.verbose, :condition_number) do
                        if isinf(op_info.condition_number)
                            return "Matrix condition number calculation failed."
                        else
                            return "Matrix condition number: $(round(op_info.condition_number, sigdigits = 4)) for $(size(A_work, 1))×$(size(A_work, 2)) matrix in dgetrf"
                        end
                    end
                end
                verb_option, message = blas_info_msg(
                    :dgetrf, info_value; extra_context = failure_op_info
                )
                @SciMLMessage(message, verbose, verb_option)
            end
        else
            @SciMLMessage(cache.verbose, :blas_success) do
                success_op_info = get_blas_operation_info(
                    :dgetrf, A_work, cache.b,
                    condition = verbose.condition_number != SciMLLogging.Silent()
                )
                let op_info = success_op_info
                    @SciMLMessage(cache.verbose, :condition_number) do
                        if isinf(op_info.condition_number)
                            return "Matrix condition number calculation failed."
                        else
                            return "Matrix condition number: $(round(op_info.condition_number, sigdigits = 4)) for $(size(A_work, 1))×$(size(A_work, 2)) matrix in dgetrf"
                        end
                    end
                end
                return "BLAS LU factorization (dgetrf) completed successfully for $(success_op_info.matrix_size) matrix"
            end
        end

        if info_value != 0
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end
        cache.isfresh = false
    end

    cacheval = @get_cacheval(cache, :BLISLUFactorization)
    factors = cacheval.factors
    info = cacheval.info
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(factors, 1), size(factors, 2)
    if m > n
        Bc = copy(cache.b)
        getrs!('N', factors, cacheval.ipiv, Bc, info)
        copyto!(cache.u, 1, Bc, 1, n)
    else
        LinearSolve._direct_lu_solve!(cacheval, cache.u, cache.b, alg)
    end

    return SciMLBase.build_linear_solution(alg, cache.u, nothing, nothing; retcode = ReturnCode.Success)
end

end
