module LinearSolveForwardDiffExt

using LinearSolve
using LinearSolve: SciMLLinearSolveAlgorithm
using LinearAlgebra
using ForwardDiff
using ForwardDiff: Dual, Partials
using SciMLBase
using RecursiveArrayTools


# Define type for non-nested dual numbers
const SingleDual{T, V, P} = Dual{T, V, P} where {T, V <:AbstractFloat , P}

# Define type for nested dual numbers
const NestedDual{T, V, P} = Dual{T, V, P} where {T, V <: Dual, P}

const SingleDualLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing}, iip,
    <:Union{<:SingleDual, <:AbstractArray{<:SingleDual}},
    <:Union{<:SingleDual, <:AbstractArray{<:SingleDual}},
    <:Any
} where {iip}

const NestedDualLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing}, iip,
    <:Union{<:NestedDual, <:AbstractArray{<:NestedDual}},
    <:Union{<:NestedDual, <:AbstractArray{<:NestedDual}},
    <:Any
} where {iip}

const DualALinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing},
    iip,
    <:Union{<:SingleDual, <:AbstractArray{<:SingleDual}},
    <:Union{Number, <:AbstractArray},
    <:Any
} where {iip}

const DualBLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing},
    iip,
    <:Union{Number, <:AbstractArray},
    <:Union{<:SingleDual, <:AbstractArray{<:SingleDual}},
    <:Any
} where {iip}

const DualAbstractLinearProblem = Union{
    SingleDualLinearProblem, DualALinearProblem, DualBLinearProblem, NestedDualLinearProblem}

LinearSolve.@concrete mutable struct DualLinearCache
    linear_cache
    dual_type

    partials_A
    partials_b
    partials_u

    dual_A
    dual_b
    dual_u
end

# function linearsolve_forwarddiff_solve(cache::DualLinearCache, alg, args...; kwargs...)
#     # Solve the primal problem
#     dual_u0 = copy(cache.linear_cache.u)
#     sol = solve!(cache.linear_cache, alg, args...; kwargs...)
#     primal_b = copy(cache.linear_cache.b)
#     uu = sol.u

#     primal_sol = deepcopy(sol)

#     # Solves Dual partials separately 
#     ∂_A = cache.partials_A
#     ∂_b = cache.partials_b

#     rhs_list = xp_linsolve_rhs(uu, ∂_A, ∂_b)

#     cache.linear_cache.u = dual_u0
#     # We can reuse the linear cache, because the same factorization will work for the partials.
#     for i in eachindex(rhs_list)
#         cache.linear_cache.b = rhs_list[i]
#         rhs_list[i] = copy(solve!(cache.linear_cache, alg, args...; kwargs...).u)
#     end

#     # Reset to the original `b` and `u`, users will expect that `b` doesn't change if they don't tell it to
#     cache.linear_cache.b = primal_b

#     partial_sols = rhs_list

#     primal_sol, partial_sols
# end

function linearsolve_forwarddiff_solve(cache::DualLinearCache, alg, args...; kwargs...)
    # Solve the primal problem
    #Main.@infiltrate
    dual_u0 = copy(cache.linear_cache.u)
    #Main.@infiltrate
    sol = solve!(cache.linear_cache, alg, args...; kwargs...)
    primal_b = copy(cache.linear_cache.b)
    uu = sol.u

    primal_sol = deepcopy(sol)

    # Solves Dual partials separately 
    ∂_A = cache.partials_A
    ∂_b = cache.partials_b

    rhs_list = xp_linsolve_rhs(uu, ∂_A, ∂_b)
    #Main.@infiltrate

    cache.linear_cache.u = dual_u0
    # We can reuse the linear cache, because the same factorization will work for the partials.
    partial_sols = []
    for i in eachindex(rhs_list)
        cache.linear_cache.b = rhs_list[i]
        # For nested duals, the result of this solve might also be a dual number
        # which will be handled recursively by the same mechanism
        push!(partial_sols, copy(solve!(cache.linear_cache, alg, args...; kwargs...).u))
    end

    # Reset to the original `b` and `u`
    cache.linear_cache.b = primal_b

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
        assump = OperatorAssumptions(issquare(nodual_value(prob.A))), kwargs...)
    # Extract primal values
    primal_A = nodual_value(prob.A)
    primal_b = nodual_value(prob.b)

    # Use the default algorithm selection based on primal values
    default_alg = LinearSolve.defaultalg(primal_A, primal_b, assump)

    # Solve with the selected algorithm
    return solve(prob, default_alg, args...; kwargs...)
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

function linearsolve_dual_solution(u::Number, partials,
        dual_type::Type{<:Dual{T, V, P}}) where {T, V <: AbstractFloat, P}
    # Handle single-level duals
    return dual_type(u, partials)
end

# function linearsolve_dual_solution(
#         u::AbstractArray, partials, dual_type)
#     partials_list = RecursiveArrayTools.VectorOfArray(partials)
#     return map(((uᵢ, pᵢ),) -> dual_type(uᵢ, Partials(Tuple(pᵢ))),
#         zip(u, partials_list[i, :] for i in 1:length(partials_list[1])))
# end

function linearsolve_dual_solution(u::AbstractArray, partials,
        dual_type::Type{<:Dual{T, V, P}}) where {T, V <: AbstractFloat, P}
    # Handle single-level duals for arrays
    partials_list = RecursiveArrayTools.VectorOfArray(partials)
    return map(((uᵢ, pᵢ),) -> dual_type(uᵢ, Partials(Tuple(pᵢ))),
        zip(u, partials_list[i, :] for i in 1:length(partials_list[1])))
end


function linearsolve_dual_solution(
        u::Number, partials, dual_type::Type{<:Dual{T, V, P}}) where {T, V <: Dual, P}
    # Handle nested duals - recursive case
    # For nested duals, u itself could be a dual number with its own partials
    inner_dual_type = V
    outer_tag_type = T

    # Reconstruct the nested dual by first building the inner dual, then the outer one
    inner_dual = u  # u is already a dual for the inner level

    # Create outer dual with the inner dual as its value
    return Dual{outer_tag_type, typeof(inner_dual), P}(inner_dual, partials)
end

function linearsolve_dual_solution(u::AbstractArray, partials,
        dual_type::Type{<:Dual{T, V, P}}) where {T, V <: Dual, P}
    # Handle nested duals for arrays - recursive case
    inner_dual_type = V
    outer_tag_type = T

    partials_list = RecursiveArrayTools.VectorOfArray(partials)

    # For nested duals, each element of u could be a dual number with its own partials
    return map(((uᵢ, pᵢ),) -> Dual{outer_tag_type, typeof(uᵢ), P}(uᵢ, Partials(Tuple(pᵢ))),
        zip(u, partials_list[i, :] for i in 1:length(partials_list[1])))
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
        assumptions = nothing,
        sensealg = LinearSolveAdjoint(),
        kwargs...)
    (; A, b, u0, p) = prob
    new_A = nodual_value(A)
    new_b = nodual_value(b)
    new_u0 = nodual_value(u0)

    ∂_A = partial_vals(A)
    ∂_b = partial_vals(b)

    primal_prob = remake(prob; A = new_A, b = new_b, u0 = new_u0)

    assumptions = OperatorAssumptions(issquare(primal_prob.A))

    if get_dual_type(prob.A) !== nothing
        dual_type = get_dual_type(prob.A)
    elseif get_dual_type(prob.b) !== nothing
        dual_type = get_dual_type(prob.b)
    end
    #Main.@infiltrate
    non_partial_cache = init(
        primal_prob, alg, args...; alias = alias, abstol = abstol, reltol = reltol,
        maxiters = maxiters, verbose = verbose, Pl = Pl, Pr = Pr, assumptions = assumptions,
        sensealg = sensealg, u0 = new_u0, kwargs...)
    return DualLinearCache(non_partial_cache, dual_type, ∂_A, ∂_b, !isnothing(∂_b) ? zero.(∂_b) : ∂_b, A, b, zero.(b))
end

function SciMLBase.solve!(cache::DualLinearCache, args...; kwargs...)
   solve!(cache, cache.alg, args...; kwargs...)
end

function SciMLBase.solve!(cache::DualLinearCache, alg::SciMLLinearSolveAlgorithm, args...; kwargs...)
    #Main.@infiltrate
    sol,
    partials = linearsolve_forwarddiff_solve(
        cache::DualLinearCache, cache.alg, args...; kwargs...)
    #Main.@infiltrate
    dual_sol = linearsolve_dual_solution(sol.u, partials, cache.dual_type)

    cache.dual_u = dual_sol

    return SciMLBase.build_linear_solution(
        cache.alg, dual_sol, sol.resid, cache; sol.retcode, sol.iters, sol.stats
    )
end
=#

# If setting A or b for DualLinearCache, put the Dual-stripped versions in the LinearCache
function Base.setproperty!(dc::DualLinearCache, sym::Symbol, val)
    # If the property is A or b, also update it in the LinearCache
    if sym === :A || sym === :b || sym === :u
        setproperty!(dc.linear_cache, sym, nodual_value(val))
    elseif hasfield(DualLinearCache, sym)
        setfield!(dc,sym,val)
    elseif hasfield(LinearSolve.LinearCache, sym)
        setproperty!(dc.linear_cache, sym, val)
    end
    

    # Update the partials if setting A or b
    if sym === :A
        setfield!(dc, :partials_A, partial_vals(val))
    elseif sym === :b
        setfield!(dc, :partials_b, partial_vals(val))
    elseif sym === :u
        setfield!(dc, :partials_u, partial_vals(val))
    end
end

# "Forwards" getproperty to LinearCache if necessary
function Base.getproperty(dc::DualLinearCache, sym::Symbol)
    if sym === :A
        dc.dual_A
    elseif sym === :b
        dc.dual_b
    elseif sym === :u
        dc.dual_u
    elseif hasfield(LinearSolve.LinearCache, sym)
        return getproperty(dc.linear_cache, sym)
    else
        return getfield(dc, sym)
    end
end



# Enhanced helper functions for Dual numbers to handle recursion
get_dual_type(x::Dual{T, V, P}) where {T, V <: AbstractFloat, P} = typeof(x)
get_dual_type(x::Dual{T, V, P}) where {T, V <: Dual, P} = typeof(x)
get_dual_type(x::AbstractArray{<:Dual}) = eltype(x)
get_dual_type(x) = nothing

# Add recursive handling for nested dual partials
partial_vals(x::Dual{T, V, P}) where {T, V <: AbstractFloat, P} = ForwardDiff.partials(x)
partial_vals(x::Dual{T, V, P}) where {T, V <: Dual, P} = ForwardDiff.partials(x)
partial_vals(x::AbstractArray{<:Dual}) = map(ForwardDiff.partials, x)
partial_vals(x) = nothing

# Add recursive handling for nested dual values
nodual_value(x) = x
nodual_value(x::Dual{T, V, P}) where {T, V <: AbstractFloat, P} = ForwardDiff.value(x)
nodual_value(x::Dual{T, V, P}) where {T, V <: Dual, P} = x.value  # Keep the inner dual intact
nodual_value(x::AbstractArray{<:Dual}) = map(nodual_value, x)


function partials_to_list(partial_matrix::Vector)
    p = eachindex(first(partial_matrix))
    [[partial[i] for partial in partial_matrix] for i in p]
end

function partials_to_list(partial_matrix)
    p = length(first(partial_matrix))
    m, n = size(partial_matrix)
    res_list = fill(zeros(typeof(partial_matrix[1, 1][1]), m, n), p)
    for k in 1:p
        res = zeros(typeof(partial_matrix[1, 1][1]), m, n)
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

