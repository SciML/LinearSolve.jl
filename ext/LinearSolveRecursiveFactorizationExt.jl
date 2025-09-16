module LinearSolveRecursiveFactorizationExt

using LinearSolve
using LinearSolve.LinearAlgebra, LinearSolve.ArrayInterface, RecursiveFactorization

LinearSolve.userecursivefactorization(A::Union{Nothing, AbstractMatrix}) = true

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::RFLUFactorization{P, T};
        kwargs...) where {P, T}
    A = cache.A
    A = convert(AbstractMatrix, A)
    fact, ipiv = LinearSolve.@get_cacheval(cache, :RFLUFactorization)
    if cache.isfresh
        if length(ipiv) != min(size(A)...)
            ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
        end
        fact = RecursiveFactorization.lu!(A, ipiv, Val(P), Val(T), check = false)
        cache.cacheval = (fact, ipiv)
        if !LinearAlgebra.issuccess(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end

        cache.isfresh = false
    end
    y = ldiv!(cache.u, LinearSolve.@get_cacheval(cache, :RFLUFactorization)[1], cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::ButterflyFactorization;
        kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    b = cache.b
    M, N = size(A)
    B, U, V = cache.cacheval[1], cache.cacheval[2], cache.cacheval[3]
    if cache.isfresh
        @assert M==N "A must be square"
        U, V, F = RecursiveFactorization.🦋workspace(A, B, U, V)
        cache.cacheval = (B, U, V, F)
        cache.isfresh = false
        b = [b; rand(4 - M % 4)]
    end
    B, U, V, F = cache.cacheval
    sol = V * (F \ (U * b))    
   SciMLBase.build_linear_solution(alg, sol[1:M], nothing, cache)
end

function LinearSolve.init_cacheval(alg::ButterflyFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    A, A', A, RecursiveFactorization.lu!(rand(1, 1), Val(false))
end

end

