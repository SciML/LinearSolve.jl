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
struct AppleAccelerateLUFactorization <: AbstractFactorization end

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

function aa_getrf!(A::AbstractMatrix{<:ComplexF64};
        ipiv = similar(A, Cint, min(size(A, 1), size(A, 2))),
        info = Ref{Cint}(),
        check = false)
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
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
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions)
    PREALLOCATED_APPLE_LU
end

function LinearSolve.init_cacheval(alg::AppleAccelerateLUFactorization,
        A::AbstractMatrix{<:Union{Float32, ComplexF32, ComplexF64}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions)
    A = rand(eltype(A), 0, 0)
    luinst = ArrayInterface.lu_instance(A)
    LU(luinst.factors, similar(A, Cint, 0), luinst.info), Ref{Cint}()
end

function SciMLBase.solve!(cache::LinearCache, alg::AppleAccelerateLUFactorization;
        kwargs...)
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
    A = cache.A
    A = convert(AbstractMatrix, A)
    verbose = cache.verbose
    if cache.isfresh
        cacheval = @get_cacheval(cache, :AppleAccelerateLUFactorization)
        res = aa_getrf!(A; ipiv = cacheval[1].ipiv, info = cacheval[2])
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

    A, info = @get_cacheval(cache, :AppleAccelerateLUFactorization)
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(A, 1), size(A, 2)
    if m > n
        Bc = copy(cache.b)
        aa_getrs!('N', A.factors, A.ipiv, Bc; info)
        return copyto!(cache.u, 1, Bc, 1, n)
    else
        copyto!(cache.u, cache.b)
        aa_getrs!('N', A.factors, A.ipiv, cache.u; info)
    end

    SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end

# Mixed precision AppleAccelerate implementation
default_alias_A(::AppleAccelerate32MixedLUFactorization, ::Any, ::Any) = false
default_alias_b(::AppleAccelerate32MixedLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_APPLE32_LU = begin
    A = rand(Float32, 0, 0)
    luinst = ArrayInterface.lu_instance(A)
    LU(luinst.factors, similar(A, Cint, 0), luinst.info), Ref{Cint}()
end

function LinearSolve.init_cacheval(alg::AppleAccelerate32MixedLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions)
    # Pre-allocate appropriate 32-bit arrays based on input type
    m, n = size(A)
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    A_32 = similar(A, T32)
    b_32 = similar(b, T32)
    u_32 = similar(u, T32)
    luinst = ArrayInterface.lu_instance(rand(T32, 0, 0))
    # Return tuple with pre-allocated arrays
    (LU(luinst.factors, similar(A_32, Cint, 0), luinst.info), Ref{Cint}(), A_32, b_32, u_32)
end

function SciMLBase.solve!(cache::LinearCache, alg::AppleAccelerate32MixedLUFactorization;
        kwargs...)
    __appleaccelerate_isavailable() ||
        error("Error, AppleAccelerate binary is missing but solve is being called. Report this issue")
    A = cache.A
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        # Get pre-allocated arrays from cacheval
        luinst, info, A_32, b_32, u_32 = @get_cacheval(cache, :AppleAccelerate32MixedLUFactorization)
        # Compute 32-bit type on demand and copy A
        T32 = eltype(A) <: Complex ? ComplexF32 : Float32
        A_32 .= T32.(A)
        res = aa_getrf!(A_32; ipiv = luinst.ipiv, info = info)
        fact = (LU(res[1:3]...), res[4], A_32, b_32, u_32)
        cache.cacheval = fact

        if !LinearAlgebra.issuccess(fact[1])
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end
        cache.isfresh = false
    end

    A_lu, info, A_32, b_32, u_32 = @get_cacheval(cache, :AppleAccelerate32MixedLUFactorization)
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(A_lu, 1), size(A_lu, 2)

    # Compute types on demand for conversions
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    Torig = eltype(cache.u)
    
    # Copy b to pre-allocated 32-bit array
    b_32 .= T32.(cache.b)

    if m > n
        aa_getrs!('N', A_lu.factors, A_lu.ipiv, b_32; info)
        # Convert back to original precision
        cache.u[1:n] .= Torig.(b_32[1:n])
    else
        copyto!(u_32, b_32)
        aa_getrs!('N', A_lu.factors, A_lu.ipiv, u_32; info)
        # Convert back to original precision
        cache.u .= Torig.(u_32)
    end

    SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end
