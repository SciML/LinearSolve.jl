using LinearAlgebra
using Libdl

# For now, only use BLAS from Accelerate (that is to say, vecLib)
const global libacc = "/System/Library/Frameworks/Accelerate.framework/Accelerate"

"""
```julia
AppleAccelerateLUFactorization()
```

A wrapper over Apple's Accelerate Library. Direct calls to Acceelrate in a way that pre-allocates workspace
to avoid allocations and does not require libblastrampoline.
"""
struct AppleAccelerateLUFactorization <: AbstractFactorization end

@static if !Sys.isapple()
    __appleaccelerate_isavailable() = false
else
    function __appleaccelerate_isavailable()
        libacc_hdl = Libdl.dlopen_e(libacc)
        if libacc_hdl == C_NULL
            return false
        end

        if dlsym_e(libacc_hdl, "dgetrf_") == C_NULL
            return false
        end
        return true
    end
end

function aa_getrf!(A::AbstractMatrix{<:ComplexF64};
        ipiv = similar(A, Cint, min(size(A, 1), size(A, 2))),
        info = Ref{Cint}(),
        check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, Cint, min(size(A, 1), size(A, 2)))
    end
    ccall(("zgetrf_", libacc), Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{ComplexF64},
            Ref{Cint}, Ptr{Cint}, Ptr{Cint}),
        m, n, A, lda, ipiv, info)
    info[] < 0 && throw(ArgumentError("Invalid arguments sent to LAPACK dgetrf_"))
    A, ipiv, BlasInt(info[]), info #Error code is stored in LU factorization type
end

function aa_getrf!(A::AbstractMatrix{<:ComplexF32};
        ipiv = similar(A, Cint, min(size(A, 1), size(A, 2))),
        info = Ref{Cint}(),
        check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, Cint, min(size(A, 1), size(A, 2)))
    end
    ccall(("cgetrf_", libacc), Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{ComplexF32},
            Ref{Cint}, Ptr{Cint}, Ptr{Cint}),
        m, n, A, lda, ipiv, info)
    info[] < 0 && throw(ArgumentError("Invalid arguments sent to LAPACK dgetrf_"))
    A, ipiv, BlasInt(info[]), info #Error code is stored in LU factorization type
end

function aa_getrf!(A::AbstractMatrix{<:Float64};
        ipiv = similar(A, Cint, min(size(A, 1), size(A, 2))),
        info = Ref{Cint}(),
        check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, Cint, min(size(A, 1), size(A, 2)))
    end
    ccall(("dgetrf_", libacc), Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{Float64},
            Ref{Cint}, Ptr{Cint}, Ptr{Cint}),
        m, n, A, lda, ipiv, info)
    info[] < 0 && throw(ArgumentError("Invalid arguments sent to LAPACK dgetrf_"))
    A, ipiv, BlasInt(info[]), info #Error code is stored in LU factorization type
end

function aa_getrf!(A::AbstractMatrix{<:Float32};
        ipiv = similar(A, Cint, min(size(A, 1), size(A, 2))),
        info = Ref{Cint}(),
        check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, Cint, min(size(A, 1), size(A, 2)))
    end

    ccall(("sgetrf_", libacc), Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{Float32},
            Ref{Cint}, Ptr{Cint}, Ptr{Cint}),
        m, n, A, lda, ipiv, info)
    info[] < 0 && throw(ArgumentError("Invalid arguments sent to LAPACK dgetrf_"))
    A, ipiv, BlasInt(info[]), info #Error code is stored in LU factorization type
end

function aa_getrs!(trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF64},
        ipiv::AbstractVector{Cint},
        B::AbstractVecOrMat{<:ComplexF64};
        info = Ref{Cint}())
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
    ccall(("zgetrs_", libacc), Cvoid,
        (Ref{UInt8}, Ref{Cint}, Ref{Cint}, Ptr{ComplexF64}, Ref{Cint},
            Ptr{Cint}, Ptr{ComplexF64}, Ref{Cint}, Ptr{Cint}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
end

function aa_getrs!(trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF32},
        ipiv::AbstractVector{Cint},
        B::AbstractVecOrMat{<:ComplexF32};
        info = Ref{Cint}())
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
    ccall(("cgetrs_", libacc), Cvoid,
        (Ref{UInt8}, Ref{Cint}, Ref{Cint}, Ptr{ComplexF32}, Ref{Cint},
            Ptr{Cint}, Ptr{ComplexF32}, Ref{Cint}, Ptr{Cint}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

function aa_getrs!(trans::AbstractChar,
        A::AbstractMatrix{<:Float64},
        ipiv::AbstractVector{Cint},
        B::AbstractVecOrMat{<:Float64};
        info = Ref{Cint}())
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
    ccall(("dgetrs_", libacc), Cvoid,
        (Ref{UInt8}, Ref{Cint}, Ref{Cint}, Ptr{Float64}, Ref{Cint},
            Ptr{Cint}, Ptr{Float64}, Ref{Cint}, Ptr{Cint}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

function aa_getrs!(trans::AbstractChar,
        A::AbstractMatrix{<:Float32},
        ipiv::AbstractVector{Cint},
        B::AbstractVecOrMat{<:Float32};
        info = Ref{Cint}())
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
    ccall(("sgetrs_", libacc), Cvoid,
        (Ref{UInt8}, Ref{Cint}, Ref{Cint}, Ptr{Float32}, Ref{Cint},
            Ptr{Cint}, Ptr{Float32}, Ref{Cint}, Ptr{Cint}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

default_alias_A(::AppleAccelerateLUFactorization, ::Any, ::Any) = false
default_alias_b(::AppleAccelerateLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_APPLE_LU = begin
    A = rand(0, 0)
    luinst = ArrayInterface.lu_instance(A)
    LU(luinst.factors, similar(A, Cint, 0), luinst.info), Ref{Cint}()
end

function LinearSolve.init_cacheval(alg::AppleAccelerateLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    PREALLOCATED_APPLE_LU
end

function LinearSolve.init_cacheval(alg::AppleAccelerateLUFactorization,
        A::AbstractMatrix{<:Union{Float32, ComplexF32, ComplexF64}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    A = rand(eltype(A), 0, 0)
    luinst = ArrayInterface.lu_instance(A)
    LU(luinst.factors, similar(A, Cint, 0), luinst.info), Ref{Cint}()
end

function SciMLBase.solve!(cache::LinearCache, alg::AppleAccelerateLUFactorization;
        kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :AppleAccelerateLUFactorization)
        res = aa_getrf!(A; ipiv = cacheval[1].ipiv, info = cacheval[2])
        fact = LU(res[1:3]...), res[4]
        cache.cacheval = fact
        cache.isfresh = false
    end

    A, info = @get_cacheval(cache, :AppleAccelerateLUFactorization)
    LinearAlgebra.require_one_based_indexing(cache.u, cache.b)
    m, n = size(A, 1), size(A, 2)
    if m > n
        Bc = copy(cache.b)
        aa_getrs!('N', A.factors, A.ipiv, Bc; info)
        return copyto!(cache.u, 1, Bc, 1, n)
    else
        copyto!(cache.u, cache.b)
        aa_getrs!('N', A.factors, A.ipiv, cache.u; info)
    end

    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end
