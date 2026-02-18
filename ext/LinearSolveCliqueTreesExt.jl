module LinearSolveCliqueTreesExt

using CliqueTrees.Multifrontal: cholesky!, ldiv!, ChordalCholesky
using LinearSolve
using SparseArrays

function makefactor(A::AbstractMatrix, alg, snd)
    if isnothing(alg) && isnothing(snd)
        F = ChordalCholesky(A)
    elseif isnothing(alg)
        F = ChordalCholesky(A; snd=alg.snd)
    elseif isnothing(snd)
        F = ChordalCholesky(A; alg=alg.alg)
    else
        F = ChordalCholesky(A; alg=alg.alg, snd=snd.snd)
    end

    return F
end

function LinearSolve.init_cacheval(
        alg::CliqueTreesFactorization{ALG, SND}, A::AbstractMatrix,
        b, u, Pl, Pr, maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {ALG, SND}
    return makefactor(A, alg.alg, alg.snd)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CliqueTreesFactorization; kwargs...)
    A = cache.A
    u = cache.u
    b = cache.b

    if cache.isfresh
        if isnothing(cache.cacheval) || !alg.reuse_symbolic
            cache.cacheval = makefactor(A, alg.alg, alg.snd)
        end

        cholesky!(copy!(cache.cacheval, A))
        cache.isfresh = false
    end

    ldiv!(u, cache.cacheval, b)
    return SciMLBase.build_linear_solution(alg, u, nothing, cache)
end

LinearSolve.PrecompileTools.@compile_workload begin
    A = sparse(
        Float64[
            3 1 0 0 0 0 0 0
            1 3 1 0 0 2 0 0
            0 1 3 1 0 1 2 1
            0 0 1 3 0 0 0 0
            0 0 0 0 3 1 1 0
            0 2 1 0 1 3 0 0
            0 0 2 0 1 0 3 1
            0 0 1 0 0 0 1 3
        ]
    )

    b = rand(8)
    prb = LinearProblem(A, b)
    sol = solve(prb, CliqueTreesFactorization())
end

end
