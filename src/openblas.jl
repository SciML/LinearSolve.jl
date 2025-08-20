"""
```julia
OpenBLASLUFactorization()
```

A direct wrapper over OpenBLAS's LU factorization (`getrf!` and `getrs!`).
This solver makes direct calls to OpenBLAS_jll without going through Julia's
libblastrampoline, which can provide performance benefits in certain configurations.

## Performance Characteristics

  - Pre-allocates workspace to avoid allocations during solving
  - Makes direct `ccall`s to OpenBLAS routines
  - Can be faster than `LUFactorization` when OpenBLAS is well-optimized for the hardware
  - Supports `Float32`, `Float64`, `ComplexF32`, and `ComplexF64` element types

## When to Use

  - When you want to ensure OpenBLAS is used regardless of the system BLAS configuration
  - When benchmarking shows better performance than `LUFactorization` on your specific hardware
  - When you need consistent behavior across different systems (always uses OpenBLAS)

## Example

```julia
using LinearSolve, LinearAlgebra

A = rand(100, 100)
b = rand(100)
prob = LinearProblem(A, b)
sol = solve(prob, OpenBLASLUFactorization())
```
"""
struct OpenBLASLUFactorization <: AbstractFactorization end

# Check if OpenBLAS is available
@static if !@isdefined(OpenBLAS_jll)
    __openblas_isavailable() = false
else
    __openblas_isavailable() = OpenBLAS_jll.is_available()
end

function openblas_getrf!(A::AbstractMatrix{<:ComplexF64};
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2))),
        info = Ref{BlasInt}(),
        check = false)
    __openblas_isavailable() || 
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(zgetrf_), OpenBLAS_jll.libopenblas), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info #Error code is stored in LU factorization type
end

function openblas_getrf!(A::AbstractMatrix{<:ComplexF32};
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2))),
        info = Ref{BlasInt}(),
        check = false)
    __openblas_isavailable() || 
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(cgetrf_), OpenBLAS_jll.libopenblas), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info #Error code is stored in LU factorization type
end

function openblas_getrf!(A::AbstractMatrix{<:Float64};
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2))),
        info = Ref{BlasInt}(),
        check = false)
    __openblas_isavailable() || 
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(dgetrf_), OpenBLAS_jll.libopenblas), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info #Error code is stored in LU factorization type
end

function openblas_getrf!(A::AbstractMatrix{<:Float32};
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2))),
        info = Ref{BlasInt}(),
        check = false)
    __openblas_isavailable() || 
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(sgetrf_), OpenBLAS_jll.libopenblas), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
        m, n, A, lda, ipiv, info)
    chkargsok(info[])
    A, ipiv, info[], info #Error code is stored in LU factorization type
end

function openblas_getrs!(trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF64},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:ComplexF64};
        info = Ref{BlasInt}())
    __openblas_isavailable() || 
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
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
    ccall((@blasfunc(zgetrs_), OpenBLAS_jll.libopenblas), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

function openblas_getrs!(trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF32},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:ComplexF32};
        info = Ref{BlasInt}())
    __openblas_isavailable() || 
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
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
    ccall((@blasfunc(cgetrs_), OpenBLAS_jll.libopenblas), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF32}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{ComplexF32}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

function openblas_getrs!(trans::AbstractChar,
        A::AbstractMatrix{<:Float64},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:Float64};
        info = Ref{BlasInt}())
    __openblas_isavailable() || 
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
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
    ccall((@blasfunc(dgetrs_), OpenBLAS_jll.libopenblas), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

function openblas_getrs!(trans::AbstractChar,
        A::AbstractMatrix{<:Float32},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:Float32};
        info = Ref{BlasInt}())
    __openblas_isavailable() || 
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
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
    ccall((@blasfunc(sgetrs_), OpenBLAS_jll.libopenblas), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{Float32}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

default_alias_A(::OpenBLASLUFactorization, ::Any, ::Any) = false
default_alias_b(::OpenBLASLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_OPENBLAS_LU = begin
    A = rand(0, 0)
    luinst = ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function LinearSolve.init_cacheval(alg::OpenBLASLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity,
        assumptions::OperatorAssumptions)
    PREALLOCATED_OPENBLAS_LU
end

function LinearSolve.init_cacheval(alg::OpenBLASLUFactorization,
        A::AbstractMatrix{<:Union{Float32, ComplexF32, ComplexF64}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity,
        assumptions::OperatorAssumptions)
    A = rand(eltype(A), 0, 0)
    ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function SciMLBase.solve!(cache::LinearCache, alg::OpenBLASLUFactorization;
        kwargs...)
    __openblas_isavailable() || 
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :OpenBLASLUFactorization)
        res = openblas_getrf!(A; ipiv = cacheval[1].ipiv, info = cacheval[2])
        fact = LU(res[1:3]...), res[4]
        cache.cacheval = fact

        if !LinearAlgebra.issuccess(fact[1])
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end
        cache.isfresh = false
    end

    A, info = @get_cacheval(cache, :OpenBLASLUFactorization)
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(A, 1), size(A, 2)
    if m > n
        Bc = copy(cache.b)
        openblas_getrs!('N', A.factors, A.ipiv, Bc; info)
        copyto!(cache.u, 1, Bc, 1, n)
    else
        copyto!(cache.u, cache.b)
        openblas_getrs!('N', A.factors, A.ipiv, cache.u; info)
    end

    SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end
