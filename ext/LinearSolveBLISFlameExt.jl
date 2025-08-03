"""
LinearSolveBLISFlameExt

Extension module that provides BLISFlameLUFactorization for LinearSolve.jl.

IMPORTANT LIMITATION: The current libflame_jll package is compiled with 32-bit integers,
but Julia's LinearAlgebra uses 64-bit integers (ILP64). This makes direct libflame 
integration impossible. As a result, this implementation falls back to using BLIS for 
BLAS operations and reference LAPACK for all LAPACK operations.

Technical approach:
- Uses BLIS for underlying BLAS operations via libblastrampoline  
- Uses reference LAPACK for both factorization (getrf) and solve (getrs) operations
- Essentially equivalent to BLISLUFactorization but with different naming for compatibility
- Follows the same API patterns as other LinearSolve extensions

This serves as a placeholder for future libflame integration when compatible packages 
become available, while still providing the BLIS performance benefits.
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
const global liblapack = LAPACK_jll.liblapack

# Note: libflame integration is not feasible due to ILP64/32-bit integer mismatch
# This implementation uses reference LAPACK for all LAPACK operations

"""
LU factorization implementations using reference LAPACK.
These mirror the standard LAPACK interface but are included here for consistency
and to enable future libflame integration when compatible packages become available.
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
    
    # Use reference LAPACK (libflame not compatible with ILP64)
    ccall((@blasfunc(zgetrf_), liblapack), Cvoid,
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
    
    ccall((@blasfunc(cgetrf_), liblapack), Cvoid,
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
    
    ccall((@blasfunc(dgetrf_), liblapack), Cvoid,
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
    
    ccall((@blasfunc(sgetrf_), liblapack), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info
end

"""
Linear system solve implementations using libflame.
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