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

Base.@kwdef struct DirectLdiv <: AbstractSolveFunction
    inplace::Bool = true
end

function SciMLBase.solve(cache::LinearCache, alg::DirectLdiv, args...; kwargs...)
    @unpack A, b, u = cache
    @unpack inplace = alg

    if inplace
        ldiv!(u, A, b)
    else
        v = A \ b
        copy!(u, v)
    end

    return SciMLBase.build_linear_solution(ApplyLdiv(), cache.u, nothing, cache)
end
