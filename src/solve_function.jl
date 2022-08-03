#
struct LinearSolveFunction{F} <: AbstractSolveFunction
    solve_func::F
end

function SciMLBase.solve(cache::LinearCache, alg::LinearSolveFunction, args...; kwargs...)
    @unpack A, b, u, p, isfresh, Pl, Pr, cacheval = cache
    @unpack solve_func = alg

    u = solve_func(A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)
    cache = set_u(cache, u)

    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

struct ApplyLdiv <: AbstractSolveFunction end
function SciMLBase.solve(cache::LinearCache, ::ApplyLdiv, args...; kwargs...)
    @unpack A, b, u = cache

    v = A \ b
    copy!(u, v)

    return SciMLBase.build_linear_solution(ApplyLdiv(), cache.u, nothing, cache)
end

struct ApplyLdiv! <: AbstractSolveFunction end
function SciMLBase.solve(cache::LinearCache, ::ApplyLdiv!, args...; kwargs...)
    @unpack A, b, u = cache

    ldiv!(u, A, b)

    return SciMLBase.build_linear_solution(ApplyLdiv!(), cache.u, nothing, cache)
end

