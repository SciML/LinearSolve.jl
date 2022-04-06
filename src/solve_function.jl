#
function DEFAULT_LINEAR_SOLVE(A,b,u,p,newA,Pl,Pr,solverdata;kwargs...)
    solve(LinearProblem(A, b; u0=u); p=p, kwargs...).u
end

Base.@kwdef struct LinearSolveFunction{F} <: AbstractSolveFunction
    solve_func::F = DEFAULT_LINEAR_SOLVE
end

function SciMLBase.solve(cache::LinearCache, alg::LinearSolveFunction,
                         args...; kwargs...)
    @unpack A,b,u,p,isfresh,Pl,Pr,cacheval = cache
    @unpack solve_func = alg

    u = solve_func(A,b,u,p,isfresh,Pl,Pr,cacheval;kwargs...)
    cache = set_u(cache, u)

    return SciMLBase.build_linear_solution(alg,cache.u,nothing,cache)
end
