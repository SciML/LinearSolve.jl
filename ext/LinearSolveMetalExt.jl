module LinearSolveMetalExt

using Metal, LinearSolve
using LinearAlgebra, SciMLBase
using SciMLBase: AbstractSciMLOperator
using LinearSolve: ArrayInterface, MKLLUFactorization, MetalOffload32MixedLUFactorization, 
                   @get_cacheval, LinearCache, SciMLBase, OperatorAssumptions, LinearVerbosity

default_alias_A(::MetalLUFactorization, ::Any, ::Any) = false
default_alias_b(::MetalLUFactorization, ::Any, ::Any) = false

function LinearSolve.init_cacheval(alg::MetalLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(cache::LinearCache, alg::MetalLUFactorization;
        kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :MetalLUFactorization)
        res = lu(MtlArray(A))
        cache.cacheval = LU(Array(res.factors), Array{Int}(res.ipiv), res.info)
        cache.isfresh = false
    end
    y = ldiv!(cache.u, @get_cacheval(cache, :MetalLUFactorization), cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

# Mixed precision Metal LU implementation
default_alias_A(::MetalOffload32MixedLUFactorization, ::Any, ::Any) = false
default_alias_b(::MetalOffload32MixedLUFactorization, ::Any, ::Any) = false

function LinearSolve.init_cacheval(alg::MetalOffload32MixedLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity,
        assumptions::OperatorAssumptions)
    # Pre-allocate with Float32 arrays
    A_f32 = Float32.(convert(AbstractMatrix, A))
    ArrayInterface.lu_instance(A_f32)
end

function SciMLBase.solve!(cache::LinearCache, alg::MetalOffload32MixedLUFactorization;
        kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :MetalOffload32MixedLUFactorization)
        # Convert to Float32 for factorization
        A_f32 = Float32.(A)
        res = lu(MtlArray(A_f32))
        # Store factorization on CPU with converted types
        cache.cacheval = LU(Array(res.factors), Array{Int}(res.ipiv), res.info)
        cache.isfresh = false
    end
    
    fact = @get_cacheval(cache, :MetalOffload32MixedLUFactorization)
    # Convert b to Float32 for solving
    b_f32 = Float32.(cache.b)
    u_f32 = similar(b_f32)
    
    # Create a temporary Float32 LU factorization for solving
    fact_f32 = LU(Float32.(fact.factors), fact.ipiv, fact.info)
    ldiv!(u_f32, fact_f32, b_f32)
    
    # Convert back to original precision
    T = eltype(cache.u)
    cache.u .= T.(u_f32)
    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

end
