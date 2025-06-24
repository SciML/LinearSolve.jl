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
    <:Any
} where {iip, T, V, P}

const DualALinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing},
    iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Union{Number, <:AbstractArray},
    <:Any
} where {iip, T, V, P}

const DualBLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing},
    iip,
    <:Union{Number, <:AbstractArray},
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Any
} where {iip, T, V, P}

const DualAbstractLinearProblem = Union{
    DualLinearProblem, DualALinearProblem, DualBLinearProblem}

LinearSolve.@concrete mutable struct DualLinearCache
    linear_cache
    dual_type
    partials_A
    partials_b
end

function linearsolve_forwarddiff_solve(cache::DualLinearCache, alg, args...; kwargs...)
    # Solve the primal problem
    dual_u0 = copy(cache.linear_cache.u)
    sol = solve!(cache.linear_cache, alg, args...; kwargs...)
    primal_b = copy(cache.linear_cache.b)
    uu = sol.u

    primal_sol = deepcopy(sol)

    # Solves Dual partials separately 
    ∂_A = cache.partials_A
    ∂_b = cache.partials_b

    rhs_list = xp_linsolve_rhs(uu, ∂_A, ∂_b)

    partial_cache = cache.linear_cache
    partial_cache.u = dual_u0

    for i in eachindex(rhs_list)
        partial_cache.b = rhs_list[i]
        rhs_list[i] = copy(solve!(partial_cache, alg, args...; kwargs...).u)
    end

    # Reset to the original `b`, users will expect that `b` doesn't change if they don't tell it to
    partial_cache.b = primal_b

    partial_sols = rhs_list

    primal_sol, partial_sols
end

function xp_linsolve_rhs(uu, ∂_A::Union{<:Partials, <:AbstractArray{<:Partials}},
        ∂_b::Union{<:Partials, <:AbstractArray{<:Partials}})
    A_list = partials_to_list(∂_A)
    b_list = partials_to_list(∂_b)

    Auu = [A * uu for A in A_list]

    return b_list .- Auu
end

function xp_linsolve_rhs(
        uu, ∂_A::Union{<:Partials, <:AbstractArray{<:Partials}}, ∂_b::Nothing)
    A_list = partials_to_list(∂_A)

    Auu = [A * uu for A in A_list]

    return -Auu
end

function xp_linsolve_rhs(
        uu, ∂_A::Nothing, ∂_b::Union{<:Partials, <:AbstractArray{<:Partials}})
    b_list = partials_to_list(∂_b)
    b_list
end

#=
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
=#

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

#=
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

    (; A, b, u0, p) = prob
    new_A = nodual_value(A)
    new_b = nodual_value(b)
    new_u0 = nodual_value(u0)

    ∂_A = partial_vals(A)
    ∂_b = partial_vals(b)

    #primal_prob = LinearProblem(new_A, new_b, u0 = new_u0)
    primal_prob = remake(prob; A = new_A, b = new_b, u0 = new_u0)

    if get_dual_type(prob.A) !== nothing
        dual_type = get_dual_type(prob.A)
    elseif get_dual_type(prob.b) !== nothing
        dual_type = get_dual_type(prob.b)
    end

    non_partial_cache = init(
        primal_prob, alg, args...; alias = alias, abstol = abstol, reltol = reltol,
        maxiters = maxiters, verbose = verbose, Pl = Pl, Pr = Pr, assumptions = assumptions,
        sensealg = sensealg, u0 = new_u0, kwargs...)
    return DualLinearCache(non_partial_cache, dual_type, ∂_A, ∂_b)
end

function SciMLBase.solve!(cache::DualLinearCache, args...; kwargs...)
    sol,
    partials = linearsolve_forwarddiff_solve(
        cache::DualLinearCache, cache.alg, args...; kwargs...)

    dual_sol = linearsolve_dual_solution(sol.u, partials, cache.dual_type)
    return SciMLBase.build_linear_solution(
        cache.alg, dual_sol, sol.resid, cache; sol.retcode, sol.iters, sol.stats
    )
end
=#

# If setting A or b for DualLinearCache, put the Dual-stripped versions in the LinearCache
# Also "forwards" setproperty so that 
function Base.setproperty!(dc::DualLinearCache, sym::Symbol, val)
    # If the property is A or b, also update it in the LinearCache
    if sym === :A || sym === :b || sym === :u
        setproperty!(dc.linear_cache, sym, nodual_value(val))
    elseif hasfield(LinearSolve.LinearCache, sym)
        setproperty!(dc.linear_cache, sym, val)
    end

    # Update the partials if setting A or b
    if sym === :A
        setfield!(dc, :partials_A, partial_vals(val))
    elseif  sym === :b
        setfield!(dc, :partials_b, partial_vals(val))
    else
        setfield!(dc, sym, val)
    end
end

# "Forwards" getproperty to LinearCache if necessary
function Base.getproperty(dc::DualLinearCache, sym::Symbol)
    if hasfield(LinearSolve.LinearCache, sym)
        return getproperty(dc.linear_cache, sym)
    else
        return getfield(dc, sym)
    end
end



# Helper functions for Dual numbers
get_dual_type(x::Dual) = typeof(x)
get_dual_type(x::AbstractArray{<:Dual}) = eltype(x)
get_dual_type(x) = nothing

partial_vals(x::Dual) = ForwardDiff.partials(x)
partial_vals(x::AbstractArray{<:Dual}) = map(ForwardDiff.partials, x)
partial_vals(x) = nothing

nodual_value(x) = x
nodual_value(x::Dual) = ForwardDiff.value(x)
nodual_value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)


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


end
