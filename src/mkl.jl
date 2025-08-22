"""
```julia
MKLLUFactorization()
```

A wrapper over Intel's Math Kernel Library (MKL). Direct calls to MKL in a way that pre-allocates workspace
to avoid allocations and does not require libblastrampoline.
"""
struct MKLLUFactorization <: AbstractFactorization end

# Check if MKL is available
@static if !@isdefined(MKL_jll)
    __mkl_isavailable() = false
else
    __mkl_isavailable() = MKL_jll.is_available()
end

function getrf!(A::AbstractMatrix{<:ComplexF64};
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2))),
        info = Ref{BlasInt}(),
        check = false)
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(zgetrf_), MKL_jll.libmkl_rt), Cvoid,
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
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(cgetrf_), MKL_jll.libmkl_rt), Cvoid,
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
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(dgetrf_), MKL_jll.libmkl_rt), Cvoid,
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
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, BlasInt, min(size(A, 1), size(A, 2)))
    end
    ccall((@blasfunc(sgetrf_), MKL_jll.libmkl_rt), Cvoid,
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
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
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
    ccall((@blasfunc(zgetrs_), MKL_jll.libmkl_rt), Cvoid,
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
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
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
    ccall((@blasfunc(cgetrs_), MKL_jll.libmkl_rt), Cvoid,
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
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
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
    ccall((@blasfunc(dgetrs_), MKL_jll.libmkl_rt), Cvoid,
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
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
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
    ccall((@blasfunc(sgetrs_), MKL_jll.libmkl_rt), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{Float32}, Ref{BlasInt},
            Ptr{BlasInt}, Ptr{Float32}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
        trans, n, size(B, 2), A, max(1, stride(A, 2)), ipiv, B, max(1, stride(B, 2)), info,
        1)
    LinearAlgebra.LAPACK.chklapackerror(BlasInt(info[]))
    B
end

default_alias_A(::MKLLUFactorization, ::Any, ::Any) = false
default_alias_b(::MKLLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_MKL_LU = begin
    A = rand(0, 0)
    luinst = ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function LinearSolve.init_cacheval(alg::MKLLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    PREALLOCATED_MKL_LU
end

function LinearSolve.init_cacheval(alg::MKLLUFactorization,
        A::AbstractMatrix{<:Union{Float32, ComplexF32, ComplexF64}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    A = rand(eltype(A), 0, 0)
    ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function SciMLBase.solve!(cache::LinearCache, alg::MKLLUFactorization;
        kwargs...)
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :MKLLUFactorization)
        res = getrf!(A; ipiv = cacheval[1].ipiv, info = cacheval[2])
        fact = LU(res[1:3]...), res[4]
        cache.cacheval = fact

        if !LinearAlgebra.issuccess(fact[1])
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end
        cache.isfresh = false
    end

    A, info = @get_cacheval(cache, :MKLLUFactorization)
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

    SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end

# Mixed precision MKL implementation
default_alias_A(::MKL32MixedLUFactorization, ::Any, ::Any) = false
default_alias_b(::MKL32MixedLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_MKL32_LU = begin
    A = rand(Float32, 0, 0)
    luinst = ArrayInterface.lu_instance(A), Ref{BlasInt}()
end

function LinearSolve.init_cacheval(alg::MKL32MixedLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    # Pre-allocate appropriate 32-bit arrays based on input type
    if eltype(A) <: Complex
        A_32 = rand(ComplexF32, 0, 0)
    else
        A_32 = rand(Float32, 0, 0)
    end
    ArrayInterface.lu_instance(A_32), Ref{BlasInt}()
end

function SciMLBase.solve!(cache::LinearCache, alg::MKL32MixedLUFactorization;
        kwargs...)
    __mkl_isavailable() ||
        error("Error, MKL binary is missing but solve is being called. Report this issue")
    A = cache.A
    A = convert(AbstractMatrix, A)

    # Check if we have complex numbers
    iscomplex = eltype(A) <: Complex

    if cache.isfresh
        cacheval = @get_cacheval(cache, :MKL32MixedLUFactorization)
        # Convert to appropriate 32-bit type for factorization
        if iscomplex
            A_f32 = ComplexF32.(A)
        else
            A_f32 = Float32.(A)
        end
        res = getrf!(A_f32; ipiv = cacheval[1].ipiv, info = cacheval[2])
        fact = LU(res[1:3]...), res[4]
        cache.cacheval = fact

        if !LinearAlgebra.issuccess(fact[1])
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end
        cache.isfresh = false
    end

    A_lu, info = @get_cacheval(cache, :MKL32MixedLUFactorization)
    require_one_based_indexing(cache.u, cache.b)
    m, n = size(A_lu, 1), size(A_lu, 2)

    # Convert b to appropriate 32-bit type for solving
    if iscomplex
        b_f32 = ComplexF32.(cache.b)
    else
        b_f32 = Float32.(cache.b)
    end

    if m > n
        Bc = copy(b_f32)
        getrs!('N', A_lu.factors, A_lu.ipiv, Bc; info)
        # Convert back to original precision
        T = eltype(cache.u)
        cache.u .= T.(Bc[1:n])
    else
        u_f32 = copy(b_f32)
        getrs!('N', A_lu.factors, A_lu.ipiv, u_f32; info)
        # Convert back to original precision
        T = eltype(cache.u)
        cache.u .= T.(u_f32)
    end

    SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end
