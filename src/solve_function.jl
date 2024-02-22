#
struct LinearSolveFunction{F} <: AbstractSolveFunction
    solve_func::F
end

function SciMLBase.solve!(cache::LinearCache, alg::LinearSolveFunction,
        args...; kwargs...)
    @unpack A, b, u, p, isfresh, Pl, Pr, cacheval = cache
    @unpack solve_func = alg

    u = solve_func(A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)
    return SciMLBase.build_linear_solution(alg, u, nothing, cache)
end

struct DirectLdiv! <: AbstractSolveFunction end

function SciMLBase.solve!(cache::LinearCache, alg::DirectLdiv!, args...; kwargs...)
    @unpack A, b, u = cache
    ldiv!(u, A, b)

    return SciMLBase.build_linear_solution(alg, u, nothing, cache)
end
