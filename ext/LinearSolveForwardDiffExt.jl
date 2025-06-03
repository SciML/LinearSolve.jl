module LinearSolveForwardDiffExt

using LinearSolve
using LinearAlgebra
using ForwardDiff
using ForwardDiff: Dual, Partials
using SciMLBase
using RecursiveArrayTools

const DualLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Union{Number, <:AbstractArray, SciMLBase.NullParameters}
} where {iip, T, V, P}

const DualALinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing},
    iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Union{Number, <:AbstractArray},
    <:Union{Number, <:AbstractArray, SciMLBase.NullParameters}
} where {iip, T, V, P}

const DualBLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing},
    iip,
    <:Union{Number, <:AbstractArray},
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Union{Number, <:AbstractArray, SciMLBase.NullParameters}
} where {iip, T, V, P}

const DualAbstractLinearProblem = Union{
    DualLinearProblem, DualALinearProblem, DualBLinearProblem}

LinearSolve.@concrete mutable struct DualLinearCache
    cache
    prob
    alg
    partials_A
    partials_b
end

function linearsolve_forwarddiff_solve(cache::DualLinearCache, alg, args...; kwargs...)
    sol = solve!(cache.cache, alg, args...; kwargs...)
    uu = sol.u

    # Solves Dual partials separately 
    ∂_A = cache.partials_A
    ∂_b = cache.partials_b

    rhs_list = xp_linsolve_rhs(uu, ∂_A, ∂_b)

    partial_prob = LinearProblem(cache.cache.A, rhs_list[1])
    partial_cache = init(partial_prob, alg, args...; kwargs...)

    Main.@infiltrate

    for i in eachindex(rhs_list)
        partial_cache.b = rhs_list[i]
        rhs_list[i] = copy(solve!(partial_cache, alg).u)
    end

    partial_sols = rhs_list

    sol, partial_sols
end

function SciMLBase.solve(prob::DualAbstractLinearProblem, args...; kwargs...)
    return solve(prob, nothing, args...; kwargs...)
end

function SciMLBase.solve(prob::DualAbstractLinearProblem, ::Nothing, args...;
        assump = OperatorAssumptions(issquare(prob.A)), kwargs...)
    return solve(prob, LinearSolve.defaultalg(prob.A, prob.b, assump), args...; kwargs...)
end

function SciMLBase.solve(prob::DualAbstractLinearProblem,
        alg::LinearSolve.SciMLLinearSolveAlgorithm, args...; kwargs...)
    solve!(init(prob, alg, args...; kwargs...))
end

function linearsolve_dual_solution(
        u::Number, partials, dual_type)
    return dual_type(u, partials)
end

function linearsolve_dual_solution(
        u::AbstractArray, partials, dual_type)
    partials_list = RecursiveArrayTools.VectorOfArray(partials)
    return map(((uᵢ, pᵢ),) -> dual_type(uᵢ, Partials(Tuple(pᵢ))),
        zip(u, partials_list[i, :] for i in 1:length(partials_list[1])))
end

get_dual_type(x::Dual) = typeof(x)
get_dual_type(x::AbstractArray{<:Dual}) = eltype(x)
get_dual_type(x) = nothing

partial_vals(x::Dual) = ForwardDiff.partials(x)
partial_vals(x::AbstractArray{<:Dual}) = map(ForwardDiff.partials, x)
partial_vals(x) = nothing

nodual_value(x) = x
nodual_value(x::Dual) = ForwardDiff.value(x)
nodual_value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

function xp_linsolve_rhs(uu, ∂_A::Union{<:Partials, <:AbstractArray{<:Partials}},
        ∂_b::Union{<:Partials, <:AbstractArray{<:Partials}})
    A_list = partials_to_list(∂_A)
    b_list = partials_to_list(∂_b)

    Auu = [A * uu for A in A_list]
    b_list .- Auu
end

function xp_linsolve_rhs(
        uu, ∂_A::Union{<:Partials, <:AbstractArray{<:Partials}}, ∂_b::Nothing)
    A_list = partials_to_list(∂_A)

    Auu = [A * uu for A in A_list]

    -Auu
end

function xp_linsolve_rhs(
        uu, ∂_A::Nothing, ∂_b::Union{<:Partials, <:AbstractArray{<:Partials}})
    b_list = partials_to_list(∂_b)
    Main.@infiltrate
    b_list
end

function partials_to_list(partial_matrix::Vector)
    p = eachindex(first(partial_matrix))
    [[partial[i] for partial in partial_matrix] for i in p]
end

function partials_to_list(partial_matrix)
    p = length(first(partial_matrix))
    m, n = size(partial_matrix)
    res_list = fill(zeros(m, n), p)
    for k in 1:p
        res = zeros(m, n)
        for i in 1:m
            for j in 1:n
                res[i, j] = partial_matrix[i, j][k]
            end
        end
        res_list[k] = res
    end
    return res_list
end

function SciMLBase.init(
        prob::DualAbstractLinearProblem, alg::LinearSolve.SciMLLinearSolveAlgorithm,
        args...;
        alias = LinearAliasSpecifier(),
        abstol = LinearSolve.default_tol(real(eltype(prob.b))),
        reltol = LinearSolve.default_tol(real(eltype(prob.b))),
        maxiters::Int = length(prob.b),
        verbose::Bool = false,
        Pl = nothing,
        Pr = nothing,
        assumptions = OperatorAssumptions(issquare(prob.A)),
        sensealg = LinearSolveAdjoint(),
        kwargs...)
    new_A = nodual_value(prob.A)
    new_b = nodual_value(prob.b)

    ∂_A = partial_vals(prob.A)
    ∂_b = partial_vals(prob.b)

    newprob = remake(prob; A = new_A, b = new_b)

    non_partial_cache = init(
        newprob, alg, args...; alias = alias, abstol = abstol, reltol = reltol,
        maxiters = maxiters, verbose = verbose, Pl = Pl, Pr = Pr, assumptions = assumptions,
        sensealg = sensealg, kwargs...)

    return DualLinearCache(non_partial_cache, prob, alg, ∂_A, ∂_b)
end

function SciMLBase.solve!(cache::DualLinearCache, args...; kwargs...)
    sol,
    partials = linearsolve_forwarddiff_solve(
        cache::DualLinearCache, cache.alg, args...; kwargs...)

    if get_dual_type(cache.prob.A) !== nothing
        dual_type = get_dual_type(cache.prob.A)
    elseif get_dual_type(cache.prob.b) !== nothing
        dual_type = get_dual_type(cache.prob.b)
    end

    dual_sol = linearsolve_dual_solution(sol.u, partials, dual_type)

    return SciMLBase.build_linear_solution(
        cache.alg, dual_sol, sol.resid, sol.cache; sol.retcode, sol.iters, sol.stats
    )
end

end
