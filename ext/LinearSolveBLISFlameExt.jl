"""
LinearSolveBLISFlameExt

Extension module that provides BLISFlameLUFactorization for LinearSolve.jl.
This extension combines BLIS for optimized BLAS operations with libflame for 
optimized LAPACK operations.

WORK IN PROGRESS: There are currently compatibility issues with libflame_jll:
- libflame_jll uses 32-bit integers while Julia's LinearAlgebra uses 64-bit integers (ILP64)
- This causes "undefined symbol" errors when calling libflame functions
- Need to resolve integer size compatibility for full integration

Technical approach:
- Uses BLIS for underlying BLAS operations via libblastrampoline
- Uses libflame for LU factorization (getrf functions) - currently blocked by ILP64 issue
- Uses libflame for solve operations (getrs functions) - libflame doesn't provide these
- Follows the same API patterns as other LinearSolve extensions

This is the foundation for the intended BLIS + libflame integration from PR #660.
"""
module LinearSolveBLISFlameExt

using Libdl
using blis_jll
using libflame_jll
using LAPACK_jll
using LinearAlgebra
using LinearSolve

using LinearAlgebra: BlasInt, LU
using LinearAlgebra.LAPACK: require_one_based_indexing, chkfinite, chkstride1, 
                            @blasfunc, chkargsok
using LinearSolve: ArrayInterface, BLISFlameLUFactorization, @get_cacheval, LinearCache, SciMLBase

# Library handles  
const global libblis = blis_jll.blis
const global libflame = libflame_jll.libflame

# NOTE: libflame integration currently blocked by ILP64/32-bit integer compatibility issue
# libflame expects 32-bit integers but Julia uses 64-bit integers in ILP64 mode

"""
LU factorization implementations using libflame for LAPACK operations.
WORK IN PROGRESS: Currently blocked by ILP64/32-bit integer compatibility.
"""

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
    
    # Call libflame's zgetrf - currently fails due to ILP64 issue
    ccall((@blasfunc(zgetrf_), libflame), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info
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
    
    ccall((@blasfunc(cgetrf_), libflame), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info
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
    
    ccall((@blasfunc(dgetrf_), libflame), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info
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
    
    ccall((@blasfunc(sgetrf_), libflame), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info
end

"""
Linear system solve implementations.
WORK IN PROGRESS: libflame doesn't provide getrs functions, so we need to use
BLAS triangular solve operations or fall back to reference LAPACK.
This is part of the integration challenge.
"""

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
    
    # WORK IN PROGRESS: libflame doesn't provide getrs, need alternative approach
    # For now, fall back to reference LAPACK for solve step
    ccall((@blasfunc(zgetrs_), LAPACK_jll.liblapack), Cvoid,
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
    # WORK IN PROGRESS: libflame doesn't provide getrs, fall back to reference LAPACK
    ccall((@blasfunc(cgetrs_), LAPACK_jll.liblapack), Cvoid,
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
    # WORK IN PROGRESS: libflame doesn't provide getrs, fall back to reference LAPACK
    ccall((@blasfunc(dgetrs_), LAPACK_jll.liblapack), Cvoid,
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
    # WORK IN PROGRESS: libflame doesn't provide getrs, fall back to reference LAPACK
    ccall((@blasfunc(sgetrs_), LAPACK_jll.liblapack), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{Float32}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

# LinearSolve integration
default_alias_A(::BLISFlameLUFactorization, ::Any, ::Any) = false
default_alias_b(::BLISFlameLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_BLIS_FLAME_LU = begin
    A = rand(0, 0)
    luinst = ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function LinearSolve.init_cacheval(alg::BLISFlameLUFactorization, A, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Bool,
    assumptions::OperatorAssumptions)
    PREALLOCATED_BLIS_FLAME_LU
end

function LinearSolve.init_cacheval(alg::BLISFlameLUFactorization, A::AbstractMatrix{<:Union{Float32,ComplexF32,ComplexF64}}, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Bool,
    assumptions::OperatorAssumptions)
    A = rand(eltype(A), 0, 0)
    ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function SciMLBase.solve!(cache::LinearCache, alg::BLISFlameLUFactorization;
    kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :BLISFlameLUFactorization)
        res = getrf!(A; ipiv = cacheval[1].ipiv, info = cacheval[2])
        fact = LU(res[1:3]...), res[4]
        cache.cacheval = fact
        cache.isfresh = false
    end

    y = ldiv!(cache.u, @get_cacheval(cache, :BLISFlameLUFactorization)[1], cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

end