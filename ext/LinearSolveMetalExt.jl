module LinearSolveMetalExt

using Metal, LinearSolve
using LinearAlgebra, SciMLBase
using SciMLBase: AbstractSciMLOperator
using LinearSolve: ArrayInterface, MKLLUFactorization, @get_cacheval, LinearCache, SciMLBase

default_alias_A(::MetalLUFactorization, ::Any, ::Any) = false
default_alias_b(::MetalLUFactorization, ::Any, ::Any) = false

function LinearSolve.init_cacheval(alg::MetalLUFactorization, A, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Bool,
    assumptions::OperatorAssumptions)
    ArrayInterface.lu_instance(convert(AbstractMatrix, MtlArray(A)))
end

function SciMLBase.solve!(cache::LinearCache, alg::MetalLUFactorization;
    kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :MetalLUFactorization)
        res = lu(MtlArray(A))
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = Array(ldiv!(MtlArray(cache.u), @get_cacheval(cache, :MetalLUFactorization), MtlArray(cache.b)))
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

end