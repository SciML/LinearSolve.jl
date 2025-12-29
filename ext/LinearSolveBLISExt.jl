module LinearSolveBLISExt

using Libdl
using blis_jll
using LAPACK_jll
using LinearAlgebra
using LinearSolve

using LinearAlgebra: BlasInt, LU
using LinearAlgebra.LAPACK: require_one_based_indexing, chkfinite, chkstride1, 
                            @blasfunc, chkargsok
using LinearSolve: ArrayInterface, BLISLUFactorization, @get_cacheval, LinearCache, SciMLBase, LinearVerbosity, get_blas_operation_info, blas_info_msg
using SciMLLogging: SciMLLogging, @SciMLMessage
using SciMLBase: ReturnCode

const global libblis = blis_jll.blis
const global liblapack = LAPACK_jll.liblapack

LinearSolve.useblis(x::Nothing) = true

function getrf!(A::AbstractMatrix{<:ComplexF64};
    ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2))),
    info = Ref{BlasInt}(),
    check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(zgetrf_), liblapack), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info #Error code is stored in LU factorization type
end

function getrf!(A::AbstractMatrix{<:ComplexF32};
    ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2))),
    info = Ref{BlasInt}(),
    check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(cgetrf_), liblapack), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info #Error code is stored in LU factorization type
end

function getrf!(A::AbstractMatrix{<:Float64};
    ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2))),
    info = Ref{BlasInt}(),
    check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(dgetrf_), liblapack), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info #Error code is stored in LU factorization type
end

function getrf!(A::AbstractMatrix{<:Float32};
    ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2))),
    info = Ref{BlasInt}(),
    check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(sgetrf_), liblapack), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info #Error code is stored in LU factorization type
end

function getrs!(trans::AbstractChar,
    A::AbstractMatrix{<:ComplexF64},
    ipiv::AbstractVector{BlasInt},
    B::AbstractVecOrMat{<:ComplexF64};
    info = Ref{BlasInt}())
    require_one_based_indexing(A, ipiv, B)
    LinearAlgebra.LAPACK.chktrans(trans)
    chkstride1(A, B, ipiv)
    n = LinearAlgebra.checksquare(A)
    if n != size(B, 1)
        throw(DimensionMismatch("B has leading dimension $(size(B,1)), but needs $n"))
    end
    if n != length(ipiv)
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs to be $n"))
    end
    nrhs = size(B, 2)
    ccall((@blasfunc(zgetrs_), liblapack), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

function getrs!(trans::AbstractChar,
    A::AbstractMatrix{<:ComplexF32},
    ipiv::AbstractVector{BlasInt},
    B::AbstractVecOrMat{<:ComplexF32};
    info = Ref{BlasInt}())
    require_one_based_indexing(A, ipiv, B)
    LinearAlgebra.LAPACK.chktrans(trans)
    chkstride1(A, B, ipiv)
    n = LinearAlgebra.checksquare(A)
    if n != size(B, 1)
        throw(DimensionMismatch("B has leading dimension $(size(B,1)), but needs $n"))
    end
    if n != length(ipiv)
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs to be $n"))
    end
    nrhs = size(B, 2)
    ccall((@blasfunc(cgetrs_), liblapack), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF32}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{ComplexF32}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

function getrs!(trans::AbstractChar,
    A::AbstractMatrix{<:Float64},
    ipiv::AbstractVector{BlasInt},
    B::AbstractVecOrMat{<:Float64};
    info = Ref{BlasInt}())
    require_one_based_indexing(A, ipiv, B)
    LinearAlgebra.LAPACK.chktrans(trans)
    chkstride1(A, B, ipiv)
    n = LinearAlgebra.checksquare(A)
    if n != size(B, 1)
        throw(DimensionMismatch("B has leading dimension $(size(B,1)), but needs $n"))
    end
    if n != length(ipiv)
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs to be $n"))
    end
    nrhs = size(B, 2)
    ccall((@blasfunc(dgetrs_), liblapack), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

function getrs!(trans::AbstractChar,
    A::AbstractMatrix{<:Float32},
    ipiv::AbstractVector{BlasInt},
    B::AbstractVecOrMat{<:Float32};
    info = Ref{BlasInt}())
    require_one_based_indexing(A, ipiv, B)
    LinearAlgebra.LAPACK.chktrans(trans)
    chkstride1(A, B, ipiv)
    n = LinearAlgebra.checksquare(A)
    if n != size(B, 1)
        throw(DimensionMismatch("B has leading dimension $(size(B,1)), but needs $n"))
    end
    if n != length(ipiv)
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs to be $n"))
    end
    nrhs = size(B, 2)
    ccall((@blasfunc(sgetrs_), liblapack), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{Float32}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

default_alias_A(::BLISLUFactorization, ::Any, ::Any) = false
default_alias_b(::BLISLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_BLIS_LU = begin
    A = rand(0, 0)
    luinst = ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function LinearSolve.init_cacheval(alg::BLISLUFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
    assumptions::OperatorAssumptions)
    PREALLOCATED_BLIS_LU
end

function LinearSolve.init_cacheval(alg::BLISLUFactorization, A::AbstractMatrix{<:Union{Float32,ComplexF32,ComplexF64}}, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
    assumptions::OperatorAssumptions)
    A = rand(eltype(A), 0, 0)
    ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function SciMLBase.solve!(cache::LinearCache, alg::BLISLUFactorization;
    kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    verbose = cache.verbose
    if cache.isfresh
        cacheval = @get_cacheval(cache, :BLISLUFactorization)
        res = getrf!(A; ipiv = cacheval[1].ipiv, info = cacheval[2])
        fact = LU(res[1:3]...), res[4]
        cache.cacheval = fact

        info_value = res[3]

        if info_value != 0
            if !isa(verbose.blas_info, SciMLLogging.Silent) || !isa(verbose.blas_errors, SciMLLogging.Silent) ||
                !isa(verbose.blas_invalid_args, SciMLLogging.Silent) 
                op_info = get_blas_operation_info(:dgetrf, A, cache.b, condition = !isa(verbose.condition_number, SciMLLogging.Silent))
                @SciMLMessage(cache.verbose, :condition_number) do 
                    if op_info[:condition_number] === nothing
                        return "Matrix condition number calculation failed."
                    else 
                        return "Matrix condition number: $(round(op_info[:condition_number], sigdigits=4)) for $(size(A, 1))×$(size(A, 2)) matrix in dgetrf"
                    end
                end
                verb_option, message = blas_info_msg(
                    :dgetrf, info_value; extra_context = op_info)
                @SciMLMessage(message, verbose, verb_option)
            end
        else
            @SciMLMessage(cache.verbose, :blas_success) do 
                op_info = get_blas_operation_info(:dgetrf, A, cache.b,
                    condition = !isa(verbose.condition_number, SciMLLogging.Silent))
                @SciMLMessage(cache.verbose, :condition_number) do
                    if op_info[:condition_number] === nothing
                        return "Matrix condition number calculation failed."
                    else
                        return "Matrix condition number: $(round(op_info[:condition_number], sigdigits=4)) for $(size(A, 1))×$(size(A, 2)) matrix in dgetrf"
                    end
                end
                return "BLAS LU factorization (dgetrf) completed successfully for $(op_info[:matrix_size]) matrix"
            end 
        end

        if !LinearAlgebra.issuccess(fact[1])
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end
        cache.isfresh = false
    end

    A, info = @get_cacheval(cache, :BLISLUFactorization)
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(A, 1), size(A, 2)
    if m > n
        Bc = copy(cache.b)
        getrs!('N', A.factors, A.ipiv, Bc; info)
        copyto!(cache.u, 1, Bc, 1, n)
    else
        copyto!(cache.u, cache.b)
        getrs!('N', A.factors, A.ipiv, cache.u; info)
    end

    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end

end
