using LinearAlgebra
using Libdl

# For now, only use BLAS from Accelerate (that is to say, vecLib)
global const libacc = "/System/Library/Frameworks/Accelerate.framework/Accelerate"

"""
```julia
AppleAccelerateLUFactorization()
```

A wrapper over Apple's Accelerate Library. Direct calls to Acceelrate in a way that pre-allocates workspace
to avoid allocations and does not require libblastrampoline.
"""
struct AppleAccelerateLUFactorization <: AbstractFactorization end

function appleaccelerate_isavailable()
    libacc_hdl = Libdl.dlopen_e(libacc)
    if libacc_hdl == C_NULL
        return false
    end

    if dlsym_e(libacc_hdl, "dgetrf_") == C_NULL
        return false
    end
    return true
end

function aa_getrf!(A::AbstractMatrix{<:Float64}; ipiv = similar(A, Cint, min(size(A,1),size(A,2))), info = Ref{Cint}(), check = false)
    require_one_based_indexing(A)
    check && chkfinite(A)
    chkstride1(A)
    m, n = size(A)
    lda  = max(1,stride(A, 2))
    if isempty(ipiv)
        ipiv = similar(A, Cint, min(size(A,1),size(A,2)))
    end

    ccall(("dgetrf_", libacc), Cvoid,
            (Ref{Cint}, Ref{Cint}, Ptr{Float64},
            Ref{Cint}, Ptr{Cint}, Ptr{Cint}),
            m, n, A, lda, ipiv, info)
    info[] < 0 && throw(ArgumentError("Invalid arguments sent to LAPACK dgetrf_"))
    A, Vector{BlasInt}(ipiv), BlasInt(info[]) #Error code is stored in LU factorization type
end

default_alias_A(::AppleAccelerateLUFactorization, ::Any, ::Any) = false
default_alias_b(::AppleAccelerateLUFactorization, ::Any, ::Any) = false

function LinearSolve.init_cacheval(alg::AppleAccelerateLUFactorization, A, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Bool,
    assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(cache::LinearCache, alg::AppleAccelerateLUFactorization;
    kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :AppleAccelerateLUFactorization)
        fact = LU(aa_getrf!(A; ipiv = cacheval.ipiv)...)
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, @get_cacheval(cache, :AppleAccelerateLUFactorization), cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end
