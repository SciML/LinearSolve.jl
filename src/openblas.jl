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
struct OpenBLASLUFactorization <: AbstractFactorization
    residualsafety::Bool
end

OpenBLASLUFactorization(; residualsafety::Bool = false) = OpenBLASLUFactorization(residualsafety)

__openblas_isavailable() = useopenblas

@inline function openblas_getrf!(
        A::AbstractMatrix{<:ComplexF64}, ipiv::AbstractVector{BlasInt},
        info::Ref{BlasInt}, check::Bool
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        (@blasfunc(zgetrf_), libopenblas), Cvoid,
        (
            Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
        ),
        m, n, A, lda, ipiv, info
    )
    chkargsok(info[])
    return info[]
end

@inline function openblas_getrf!(
        A::AbstractMatrix{<:ComplexF32}, ipiv::AbstractVector{BlasInt},
        info::Ref{BlasInt}, check::Bool
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        (@blasfunc(cgetrf_), libopenblas), Cvoid,
        (
            Ref{BlasInt}, Ref{BlasInt}, Ptr{ComplexF32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
        ),
        m, n, A, lda, ipiv, info
    )
    chkargsok(info[])
    return info[]
end

@inline function openblas_getrf!(
        A::AbstractMatrix{<:Float64}, ipiv::AbstractVector{BlasInt},
        info::Ref{BlasInt}, check::Bool
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        (@blasfunc(dgetrf_), libopenblas), Cvoid,
        (
            Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
        ),
        m, n, A, lda, ipiv, info
    )
    chkargsok(info[])
    return info[]
end

@inline function openblas_getrf!(
        A::AbstractMatrix{<:Float32}, ipiv::AbstractVector{BlasInt},
        info::Ref{BlasInt}, check::Bool
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) ||
        throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        (@blasfunc(sgetrf_), libopenblas), Cvoid,
        (
            Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32},
            Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
        ),
        m, n, A, lda, ipiv, info
    )
    chkargsok(info[])
    return info[]
end

function openblas_getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF64},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:ComplexF64},
        info::Ref{BlasInt}
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
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
        (@blasfunc(zgetrs_), libopenblas), Cvoid,
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

function openblas_getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF32},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:ComplexF32},
        info::Ref{BlasInt}
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
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
        (@blasfunc(cgetrs_), libopenblas), Cvoid,
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

function openblas_getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:Float64},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:Float64},
        info::Ref{BlasInt}
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
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
        (@blasfunc(dgetrs_), libopenblas), Cvoid,
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

function openblas_getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:Float32},
        ipiv::AbstractVector{BlasInt},
        B::AbstractVecOrMat{<:Float32},
        info::Ref{BlasInt}
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
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
        (@blasfunc(sgetrs_), libopenblas), Cvoid,
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

_get_residualsafety(alg::OpenBLASLUFactorization) = alg.residualsafety

default_alias_A(::OpenBLASLUFactorization, ::Any, ::Any) = false
default_alias_b(::OpenBLASLUFactorization, ::Any, ::Any) = false

mutable struct OpenBLASLUCache{F, P, I}
    factors::F
    ipiv::P
    info::I
end

_custom_cache_factorization(::OpenBLASLUFactorization, cacheval::OpenBLASLUCache) =
    LU(cacheval.factors, cacheval.ipiv, Int(cacheval.info[]))

@inline function _direct_lu_factorize!(
        cacheval::OpenBLASLUCache, A_work, ::OpenBLASLUFactorization
    )
    cacheval.factors = A_work
    return openblas_getrf!(A_work, cacheval.ipiv, cacheval.info, false)
end

@inline function _direct_lu_solve!(
        cacheval::OpenBLASLUCache, u, b, ::OpenBLASLUFactorization
    )
    copyto!(u, b)
    openblas_getrs!('N', cacheval.factors, cacheval.ipiv, u, cacheval.info)
    return u
end

function LinearSolve.init_cacheval(
        alg::OpenBLASLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    A0 = Matrix{Float64}(undef, 0, 0)
    return OpenBLASLUCache(A0, Vector{BlasInt}(undef, 0), Ref{BlasInt}())
end

function LinearSolve.init_cacheval(
        alg::OpenBLASLUFactorization,
        A::DenseMatrix{<:BLASELTYPES}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return OpenBLASLUCache(
        A, similar(A, BlasInt, min(size(A, 1), size(A, 2))), Ref{BlasInt}()
    )
end

function SciMLBase.solve!(
        cache::LinearCache, alg::OpenBLASLUFactorization;
        kwargs...
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    A_work = convert(AbstractMatrix, cache.A)
    check_safety = alg.residualsafety && cache.isfresh
    needs_backup = check_safety ||
        (cache.alg isa DefaultLinearSolver && cache.alg.safetyfallback && cache.isfresh)
    A_original = needs_backup ? _copy_A_for_safety(cache) : A_work
    verbose = cache.verbose
    if cache.isfresh
        cacheval = @get_cacheval(cache, :OpenBLASLUFactorization)
        if length(cacheval.ipiv) != min(size(A_work, 1), size(A_work, 2))
            cacheval.ipiv = similar(
                A_work, BlasInt, min(size(A_work, 1), size(A_work, 2))
            )
        end
        info_value = _direct_lu_factorize!(cacheval, A_work, alg)

        if info_value != 0
            if verbose.blas_info != SciMLLogging.Silent() ||
                    verbose.blas_errors != SciMLLogging.Silent() ||
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
                verb_option,
                    message = blas_info_msg(
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

    cacheval = @get_cacheval(cache, :OpenBLASLUFactorization)
    factors = cacheval.factors
    info = cacheval.info
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(factors, 1), size(factors, 2)
    if m > n
        Bc = copy(cache.b)
        openblas_getrs!('N', factors, cacheval.ipiv, Bc, info)
        if cache.b isa AbstractMatrix
            copyto!(cache.u, @view(Bc[1:n, :]))
        else
            copyto!(cache.u, 1, Bc, 1, n)
        end
    else
        _direct_lu_solve!(cacheval, cache.u, cache.b, alg)
    end

    if check_safety
        failed = _check_residual_safety(cache, alg, A_original, cache.u)
        failed !== nothing && return failed
    end

    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Success
    )
end

# Mixed precision OpenBLAS implementation
default_alias_A(::OpenBLAS32MixedLUFactorization, ::Any, ::Any) = false
default_alias_b(::OpenBLAS32MixedLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_OPENBLAS32_LU = begin
    A = rand(Float32, 0, 0)
    luinst = ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function LinearSolve.init_cacheval(
        alg::OpenBLAS32MixedLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    # Pre-allocate appropriate 32-bit arrays based on input type
    m, n = size(A)
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    A_32 = similar(A, T32)
    b_32 = similar(b, T32)
    u_32 = similar(u, T32)
    ipiv = similar(A_32, BlasInt, min(size(A_32, 1), size(A_32, 2)))
    luinst = LU(A_32, ipiv, zero(BlasInt))
    # Return tuple with pre-allocated arrays
    return (luinst, Ref{BlasInt}(), A_32, b_32, u_32)
end

function SciMLBase.solve!(
        cache::LinearCache, alg::OpenBLAS32MixedLUFactorization;
        kwargs...
    )
    __openblas_isavailable() ||
        error("Error, OpenBLAS binary is missing but solve is being called. Report this issue")
    A = cache.A
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        # Get pre-allocated arrays from cacheval
        luinst, info, A_32, b_32,
            u_32 = @get_cacheval(cache, :OpenBLAS32MixedLUFactorization)
        # Compute 32-bit type on demand and copy A
        T32 = eltype(A) <: Complex ? ComplexF32 : Float32
        A_32 .= T32.(A)
        info_value = openblas_getrf!(A_32, luinst.ipiv, info, false)

        if info_value != 0
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end
        cache.isfresh = false
    end

    A_lu, info, A_32, b_32, u_32 = @get_cacheval(cache, :OpenBLAS32MixedLUFactorization)
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(A_lu, 1), size(A_lu, 2)

    # Compute types on demand for conversions
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    Torig = eltype(cache.u)

    # Copy b to pre-allocated 32-bit array
    b_32 .= T32.(cache.b)

    if m > n
        openblas_getrs!('N', A_lu.factors, A_lu.ipiv, b_32, info)
        # Convert back to original precision
        if cache.b isa AbstractMatrix
            cache.u .= Torig.(@view(b_32[1:n, :]))
        else
            cache.u[1:n] .= Torig.(@view(b_32[1:n]))
        end
    else
        copyto!(u_32, b_32)
        openblas_getrs!('N', A_lu.factors, A_lu.ipiv, u_32, info)
        # Convert back to original precision
        cache.u .= Torig.(u_32)
    end

    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Success
    )
end
