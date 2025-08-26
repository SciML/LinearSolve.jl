module LinearSolveRecursiveFactorizationExt

using LinearSolve
using SparseArrays
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
    if cache.isfresh
        @assert M==N "A must be square"
        U, V, F = RecursiveFactorization.🦋workspace(A)
        cache.cacheval = (U, V, F)
        cache.isfresh = false
    end
    U, V, F = cache.cacheval
    #sol = U * b_ext
    #TriangularSolve.rdiv!(sol, A_ext, F.U, Val(false))
    #TriangularSolve.ldiv!(sol, A_ext, F.L, Val(false))
    #sol *= V
    sol = V * (F \ (U * b))    
    #sol = V * (TriangularSolve.ldiv!(UpperTriangular(F.U), TriangularSolve.ldiv!(LowerTriangular(F.L), U * b)))
    SciMLBase.build_linear_solution(alg, sol, nothing, cache)
end

function LinearSolve.init_cacheval(alg::ButterflyFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    #A, b, (RecursiveFactorization.SparseBandedMatrix{typeof(A[1,1])}(undef, 1, 1))', RecursiveFactorization.SparseBandedMatrix{typeof(A[1,1])}(undef, 1, 1), RecursiveFactorization.lu!(rand(1, 1), Val(false))
    #A, b, (spzeros(1, 1))', spzeros(1,1), RecursiveFactorization.lu!(rand(1, 1), Val(false))
    A', A, RecursiveFactorization.lu!(rand(1, 1), Val(false))
end

end

