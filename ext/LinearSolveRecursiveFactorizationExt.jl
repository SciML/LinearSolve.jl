module LinearSolveRecursiveFactorizationExt

using LinearSolve
using LinearSolve.LinearAlgebra, LinearSolve.ArrayInterface, RecursiveFactorization

function LinearSolve.init_cacheval(alg::RFLUFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
    ArrayInterface.lu_instance(convert(AbstractMatrix, A)), ipiv
end

function LinearSolve.init_cacheval(
        alg::RFLUFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, 0)
    LinearSolve.PREALLOCATED_LU, ipiv
end

function LinearSolve.init_cacheval(alg::RFLUFactorization,
        A::Union{LinearSolve.SparseArrays.AbstractSparseArray, LinearSolve.SciMLOperators.AbstractSciMLOperator}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing, nothing
end

function LinearSolve.init_cacheval(alg::RFLUFactorization,
        A::Union{Diagonal, SymTridiagonal, Tridiagonal}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing, nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::RFLUFactorization{P, T};
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
    y = ldiv!(cache.u, @get_cacheval(cache, :RFLUFactorization)[1], cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

end
