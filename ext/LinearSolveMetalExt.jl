module LinearSolveMetalExt

using Metal, LinearSolve
using LinearAlgebra, SciMLBase
using SciMLBase: AbstractSciMLOperator
using LinearSolve: ArrayInterface, MKLLUFactorization, MetalOffload32MixedLUFactorization,
    @get_cacheval, LinearCache, LinearCacheType, SciMLBase, OperatorAssumptions, LinearVerbosity

@static if Sys.isapple()

    LinearSolve.usemetal(x::Nothing) = true

end

default_alias_A(::MetalLUFactorization, ::Any, ::Any) = false
default_alias_b(::MetalLUFactorization, ::Any, ::Any) = false

function LinearSolve.init_cacheval(
        alg::MetalLUFactorization, A::AbstractArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(
        cache::LinearCacheType, alg::MetalLUFactorization;
        kwargs...
    )
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :MetalLUFactorization)
        res = lu(MtlArray(A))
        cache.cacheval = LU(Array(res.factors), Array{Int}(res.ipiv), res.info)
        cache.isfresh = false
    end
    y = ldiv!(cache.u, @get_cacheval(cache, :MetalLUFactorization), cache.b)
    return SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

# Mixed precision Metal LU implementation
default_alias_A(::MetalOffload32MixedLUFactorization, ::Any, ::Any) = false
default_alias_b(::MetalOffload32MixedLUFactorization, ::Any, ::Any) = false

function LinearSolve.init_cacheval(
        alg::MetalOffload32MixedLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    # Pre-allocate with Float32 arrays
    m, n = size(A)
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    A_f32 = similar(A, T32)
    b_f32 = similar(b, T32)
    u_f32 = similar(u, T32)
    luinst = ArrayInterface.lu_instance(rand(T32, 0, 0))
    # Pre-allocate Metal arrays
    A_mtl = MtlArray{T32}(undef, m, n)
    b_mtl = MtlVector{T32}(undef, size(b, 1))
    u_mtl = MtlVector{T32}(undef, size(u, 1))
    return (luinst, A_f32, b_f32, u_f32, A_mtl, b_mtl, u_mtl)
end

function SciMLBase.solve!(
        cache::LinearCacheType, alg::MetalOffload32MixedLUFactorization;
        kwargs...
    )
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        luinst, A_f32, b_f32, u_f32, A_mtl, b_mtl, u_mtl = @get_cacheval(cache, :MetalOffload32MixedLUFactorization)
        # Compute 32-bit type on demand and convert
        T32 = eltype(A) <: Complex ? ComplexF32 : Float32
        A_f32 .= T32.(A)
        copyto!(A_mtl, A_f32)
        res = lu(A_mtl)
        # Store factorization and pre-allocated arrays
        fact = LU(Array(res.factors), Array{Int}(res.ipiv), res.info)
        cache.cacheval = (fact, A_f32, b_f32, u_f32, A_mtl, b_mtl, u_mtl)
        cache.isfresh = false
    end

    fact, A_f32, b_f32, u_f32, A_mtl, b_mtl, u_mtl = @get_cacheval(cache, :MetalOffload32MixedLUFactorization)

    # Compute types on demand for conversions
    T32 = eltype(cache.A) <: Complex ? ComplexF32 : Float32
    Torig = eltype(cache.u)

    # Convert b to 32-bit for solving
    b_f32 .= T32.(cache.b)

    # Create a temporary Float32 LU factorization for solving
    fact_f32 = LU(T32.(fact.factors), fact.ipiv, fact.info)
    ldiv!(u_f32, fact_f32, b_f32)

    # Convert back to original precision
    cache.u .= Torig.(u_f32)
    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

end
