using LinearAlgebra

# For now, only use BLAS from Accelerate (that is to say, vecLib)
const global libacc = "/System/Library/Frameworks/Accelerate.framework/Accelerate"

"""
```julia
AppleAccelerateLUFactorization()
```

A wrapper over Apple's Accelerate Library. Direct calls to Acceelrate in a way that pre-allocates workspace
to avoid allocations and does not require libblastrampoline.
"""
struct AppleAccelerateLUFactorization <: AbstractFactorization
    residualsafety::Bool
end

AppleAccelerateLUFactorization(; residualsafety::Bool = false) = AppleAccelerateLUFactorization(residualsafety)

# To make Enzyme happy, this has to be static
@static if !Sys.isapple()
    const AA_IS_AVAILABLE = false
    __appleaccelerate_isavailable() = false
else
    @static if Libdl.dlopen(libacc; throw_error = false) === nothing
        __appleaccelerate_isavailable() = false
    elseif Libdl.dlsym(Libdl.dlopen(libacc), "dgetrf_"; throw_error = false) === nothing
        __appleaccelerate_isavailable() = false
    else
        __appleaccelerate_isavailable() = true
    end
end

@inline function aa_getrf!(
        A::AbstractMatrix{<:ComplexF64}, ipiv::AbstractVector{Cint},
        info::Ref{Cint}, check::Bool
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) || throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        ("zgetrf_", libacc), Cvoid,
        (
            Ref{Cint}, Ref{Cint}, Ptr{ComplexF64},
            Ref{Cint}, Ptr{Cint}, Ptr{Cint},
        ),
        m, n, A, lda, ipiv, info
    )
    info[] < 0 && throw(ArgumentError("Invalid arguments sent to LAPACK dgetrf_"))
    return BlasInt(info[])
end

@inline function aa_getrf!(
        A::AbstractMatrix{<:ComplexF32}, ipiv::AbstractVector{Cint},
        info::Ref{Cint}, check::Bool
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) || throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        ("cgetrf_", libacc), Cvoid,
        (
            Ref{Cint}, Ref{Cint}, Ptr{ComplexF32},
            Ref{Cint}, Ptr{Cint}, Ptr{Cint},
        ),
        m, n, A, lda, ipiv, info
    )
    info[] < 0 && throw(ArgumentError("Invalid arguments sent to LAPACK dgetrf_"))
    return BlasInt(info[])
end

@inline function aa_getrf!(
        A::AbstractMatrix{<:Float64}, ipiv::AbstractVector{Cint},
        info::Ref{Cint}, check::Bool
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) || throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))
    ccall(
        ("dgetrf_", libacc), Cvoid,
        (
            Ref{Cint}, Ref{Cint}, Ptr{Float64},
            Ref{Cint}, Ptr{Cint}, Ptr{Cint},
        ),
        m, n, A, lda, ipiv, info
    )
    info[] < 0 && throw(ArgumentError("Invalid arguments sent to LAPACK dgetrf_"))
    return BlasInt(info[])
end

@inline function aa_getrf!(
        A::AbstractMatrix{<:Float32}, ipiv::AbstractVector{Cint},
        info::Ref{Cint}, check::Bool
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    length(ipiv) == min(m, n) || throw(DimensionMismatch("ipiv has length $(length(ipiv)), but needs $(min(m, n))"))

    ccall(
        ("sgetrf_", libacc), Cvoid,
        (
            Ref{Cint}, Ref{Cint}, Ptr{Float32},
            Ref{Cint}, Ptr{Cint}, Ptr{Cint},
        ),
        m, n, A, lda, ipiv, info
    )
    info[] < 0 && throw(ArgumentError("Invalid arguments sent to LAPACK dgetrf_"))
    return BlasInt(info[])
end

function aa_getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF64},
        ipiv::AbstractVector{Cint},
        B::AbstractVecOrMat{<:ComplexF64},
        info::Ref{Cint}
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
        ("zgetrs_", libacc), Cvoid,
        (
            Ref{UInt8}, Ref{Cint}, Ref{Cint}, Ptr{ComplexF64}, Ref{Cint},
            Ptr{Cint}, Ptr{ComplexF64}, Ref{Cint}, Ptr{Cint}, Clong,
        ),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1
    )
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
end

function aa_getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:ComplexF32},
        ipiv::AbstractVector{Cint},
        B::AbstractVecOrMat{<:ComplexF32},
        info::Ref{Cint}
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
        ("cgetrs_", libacc), Cvoid,
        (
            Ref{UInt8}, Ref{Cint}, Ref{Cint}, Ptr{ComplexF32}, Ref{Cint},
            Ptr{Cint}, Ptr{ComplexF32}, Ref{Cint}, Ptr{Cint}, Clong,
        ),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1
    )
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    return B
end

function aa_getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:Float64},
        ipiv::AbstractVector{Cint},
        B::AbstractVecOrMat{<:Float64},
        info::Ref{Cint}
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
        ("dgetrs_", libacc), Cvoid,
        (
            Ref{UInt8}, Ref{Cint}, Ref{Cint}, Ptr{Float64}, Ref{Cint},
            Ptr{Cint}, Ptr{Float64}, Ref{Cint}, Ptr{Cint}, Clong,
        ),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1
    )
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    return B
end

function aa_getrs!(
        trans::AbstractChar,
        A::AbstractMatrix{<:Float32},
        ipiv::AbstractVector{Cint},
        B::AbstractVecOrMat{<:Float32},
        info::Ref{Cint}
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
        ("sgetrs_", libacc), Cvoid,
        (
            Ref{UInt8}, Ref{Cint}, Ref{Cint}, Ptr{Float32}, Ref{Cint},
            Ptr{Cint}, Ptr{Float32}, Ref{Cint}, Ptr{Cint}, Clong,
        ),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1
    )
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    return B
end

_get_residualsafety(alg::AppleAccelerateLUFactorization) = alg.residualsafety

default_alias_A(::AppleAccelerateLUFactorization, ::Any, ::Any) = false
default_alias_b(::AppleAccelerateLUFactorization, ::Any, ::Any) = false

mutable struct AppleAccelerateLUCache{F, P, I}
    factors::F
    ipiv::P
    info::I
end

function LinearSolve.init_cacheval(
        alg::AppleAccelerateLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    A0 = Matrix{Float64}(undef, 0, 0)
    return AppleAccelerateLUCache(A0, Vector{Cint}(undef, 0), Ref{Cint}())
end

function LinearSolve.init_cacheval(
        alg::AppleAccelerateLUFactorization,
        A::DenseMatrix{<:BLASELTYPES}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return AppleAccelerateLUCache(
        A, similar(A, Cint, min(size(A, 1), size(A, 2))), Ref{Cint}()
    )
end

function SciMLBase.solve!(
        cache::LinearCache, alg::AppleAccelerateLUFactorization;
        kwargs...
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
    A_work = convert(AbstractMatrix, cache.A)
    check_safety = alg.residualsafety && cache.isfresh
    needs_backup = check_safety ||
        (cache.alg isa DefaultLinearSolver && cache.alg.safetyfallback && cache.isfresh)
    A_original = needs_backup ? _copy_A_for_safety(cache) : A_work
    verbose = cache.verbose
    if cache.isfresh
        cacheval = @get_cacheval(cache, :AppleAccelerateLUFactorization)
        if length(cacheval.ipiv) != min(size(A_work, 1), size(A_work, 2))
            cacheval.ipiv = similar(
                A_work, Cint, min(size(A_work, 1), size(A_work, 2))
            )
        end
        cacheval.factors = A_work
        info_value = aa_getrf!(A_work, cacheval.ipiv, cacheval.info, false)

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

    cacheval = @get_cacheval(cache, :AppleAccelerateLUFactorization)
    factors = cacheval.factors
    info = cacheval.info
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(factors, 1), size(factors, 2)
    if m > n
        Bc = copy(cache.b)
        aa_getrs!('N', factors, cacheval.ipiv, Bc, info)
        if cache.b isa AbstractMatrix
            copyto!(cache.u, @view(Bc[1:n, :]))
        else
            copyto!(cache.u, 1, Bc, 1, n)
        end
    else
        copyto!(cache.u, cache.b)
        aa_getrs!('N', factors, cacheval.ipiv, cache.u, info)
    end

    if check_safety
        failed = _check_residual_safety(cache, alg, A_original, cache.u)
        failed !== nothing && return failed
    end

    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Success
    )
end

# Mixed precision AppleAccelerate implementation
default_alias_A(::AppleAccelerate32MixedLUFactorization, ::Any, ::Any) = false
default_alias_b(::AppleAccelerate32MixedLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_APPLE32_LU = begin
    A = rand(Float32, 0, 0)
    luinst = ArrayInterface.lu_instance(A)
    LU(luinst.factors, similar(A, Cint, 0), luinst.info), Ref{Cint}()
end

function LinearSolve.init_cacheval(
        alg::AppleAccelerate32MixedLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    # Pre-allocate appropriate 32-bit arrays based on input type
    m, n = size(A)
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    A_32 = similar(A, T32)
    b_32 = similar(b, T32)
    u_32 = similar(u, T32)
    ipiv = similar(A_32, Cint, min(size(A_32, 1), size(A_32, 2)))
    luinst = LU(A_32, ipiv, zero(BlasInt))
    # Return tuple with pre-allocated arrays
    return (luinst, Ref{Cint}(), A_32, b_32, u_32)
end

function SciMLBase.solve!(
        cache::LinearCache, alg::AppleAccelerate32MixedLUFactorization;
        kwargs...
    )
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
    A = cache.A
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        # Get pre-allocated arrays from cacheval
        luinst, info, A_32,
            b_32, u_32 = @get_cacheval(cache, :AppleAccelerate32MixedLUFactorization)
        # Compute 32-bit type on demand and copy A
        T32 = eltype(A) <: Complex ? ComplexF32 : Float32
        A_32 .= T32.(A)
        info_value = aa_getrf!(A_32, luinst.ipiv, info, false)

        if info_value != 0
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end
        cache.isfresh = false
    end

    A_lu, info, A_32, b_32,
        u_32 = @get_cacheval(cache, :AppleAccelerate32MixedLUFactorization)
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(A_lu, 1), size(A_lu, 2)

    # Compute types on demand for conversions
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    Torig = eltype(cache.u)

    # Copy b to pre-allocated 32-bit array
    b_32 .= T32.(cache.b)

    if m > n
        aa_getrs!('N', A_lu.factors, A_lu.ipiv, b_32, info)
        # Convert back to original precision
        if cache.b isa AbstractMatrix
            cache.u .= Torig.(@view(b_32[1:n, :]))
        else
            cache.u[1:n] .= Torig.(@view(b_32[1:n]))
        end
    else
        copyto!(u_32, b_32)
        aa_getrs!('N', A_lu.factors, A_lu.ipiv, u_32, info)
        # Convert back to original precision
        cache.u .= Torig.(u_32)
    end

    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Success
    )
end
