module LinearSolveForwardDiffExt

using LinearSolve
using LinearSolve: SciMLLinearSolveAlgorithm, __init, LinearVerbosity, DefaultLinearSolver,
                   DefaultAlgorithmChoice, defaultalg, reinit!
using LinearAlgebra
using ForwardDiff
using ForwardDiff: Dual, Partials
using SciMLBase
using RecursiveArrayTools
using SciMLLogging
using ArrayInterface

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

LinearSolve.@concrete mutable struct DualLinearCache{DT}
    linear_cache

    partials_A
    partials_b
    partials_u

    # Cached lists of partials to avoid repeated allocations
    partials_A_list
    partials_b_list

    # Cached intermediate values for calculations
    rhs_list
    dual_u0_cache
    primal_u_cache
    primal_b_cache

    # Cache validity flag for RHS precalculation optimization
    rhs_cache_valid

    dual_A
    dual_b
    dual_u
end

function linearsolve_forwarddiff_solve!(cache::DualLinearCache, alg, args...; kwargs...)
    # Solve the primal problem
    cache.dual_u0_cache .= cache.linear_cache.u
    sol = solve!(cache.linear_cache, alg, args...; kwargs...)

    cache.primal_u_cache .= cache.linear_cache.u
    cache.primal_b_cache .= cache.linear_cache.b
    uu = sol.u

    # Solves Dual partials separately 
    ∂_A = cache.partials_A
    ∂_b = cache.partials_b

    xp_linsolve_rhs!(uu, ∂_A, ∂_b, cache)

    rhs_list = cache.rhs_list

    cache.linear_cache.u .= cache.dual_u0_cache
    # We can reuse the linear cache, because the same factorization will work for the partials.
    for i in eachindex(rhs_list)
        if cache.linear_cache isa DualLinearCache
            # For nested duals, assign directly to partials_b
            cache.linear_cache.b = copy(rhs_list[i])
        else
            # For regular linear cache, use broadcasting assignment
            cache.linear_cache.b .= rhs_list[i]
        end
        rhs_list[i] .= solve!(cache.linear_cache, alg, args...; kwargs...).u
    end

    # Reset to the original `b` and `u`, users will expect that `b` doesn't change if they don't tell it to
    cache.linear_cache.b .= cache.primal_b_cache
    cache.linear_cache.u .= cache.primal_u_cache

    return sol
end

function xp_linsolve_rhs!(uu, ∂_A::Union{<:Partials, <:AbstractArray{<:Partials}},
        ∂_b::Union{<:Partials, <:AbstractArray{<:Partials}}, cache::DualLinearCache)

    # Update cached partials lists if cache is invalid
    if !cache.rhs_cache_valid
        update_partials_list!(∂_A, cache.partials_A_list)
        update_partials_list!(∂_b, cache.partials_b_list)
        cache.rhs_cache_valid = true
    end

    A_list = cache.partials_A_list
    b_list = cache.partials_b_list

    # Compute rhs = b - A*uu using precalculated b_list and five-argument mul!
    for i in eachindex(b_list)
        cache.rhs_list[i] .= b_list[i]
        mul!(cache.rhs_list[i], A_list[i], uu, -1, 1)
    end

    return cache.rhs_list
end

function xp_linsolve_rhs!(
        uu, ∂_A::Union{<:Partials, <:AbstractArray{<:Partials}},
        ∂_b::Nothing, cache::DualLinearCache)

    # Update cached partials list for A if cache is invalid
    if !cache.rhs_cache_valid
        update_partials_list!(∂_A, cache.partials_A_list)
        cache.rhs_cache_valid = true
    end

    A_list = cache.partials_A_list

    # Compute rhs = -A*uu using five-argument mul!
    for i in eachindex(A_list)
        mul!(cache.rhs_list[i], A_list[i], uu, -1, 0)
    end

    return cache.rhs_list
end

function xp_linsolve_rhs!(
        uu, ∂_A::Nothing, ∂_b::Union{<:Partials, <:AbstractArray{<:Partials}},
        cache::DualLinearCache)

    # Update cached partials list for b if cache is invalid
    if !cache.rhs_cache_valid
        update_partials_list!(∂_b, cache.partials_b_list)
        cache.rhs_cache_valid = true
    end

    b_list = cache.partials_b_list

    # Copy precalculated b_list to rhs_list (no A*uu computation needed)
    for i in eachindex(b_list)
        cache.rhs_list[i] .= b_list[i]
    end

    return cache.rhs_list
end

function linearsolve_dual_solution(
        u::Number, partials, cache::DualLinearCache{DT}) where {DT}
    return DT(u, partials)
end

function linearsolve_dual_solution(u::AbstractArray, partials,
        cache::DualLinearCache{DT}) where {T, V, N, DT <: Dual{T, V, N}}
    # Optimized in-place version that reuses cache.dual_u
    linearsolve_dual_solution!(getfield(cache, :dual_u), u, partials)
    return getfield(cache, :dual_u)
end

function linearsolve_dual_solution!(dual_u::AbstractArray{DT}, u::AbstractArray,
        partials) where {T, V, N, DT <: Dual{T, V, N}}
    # Direct in-place construction of dual numbers without temporary allocations
    n_partials = length(partials)

    for i in eachindex(u, dual_u)
        # Extract partials for this element directly
        partial_vals = ntuple(Val(N)) do j
            V(partials[j][i])
        end

        # Construct dual number in-place
        dual_u[i] = DT(u[i], Partials{N, V}(partial_vals))
    end

    return dual_u
end

function SciMLBase.init(prob::DualAbstractLinearProblem, alg::SciMLLinearSolveAlgorithm, args...; kwargs...)
    return __dual_init(prob, alg, args...; kwargs...)
end

# Opt out for GenericLUFactorization
function SciMLBase.init(prob::DualAbstractLinearProblem, alg::GenericLUFactorization, args...; kwargs...)
    return __init(prob, alg, args...; kwargs...)
end

# Opt out for SparspakFactorization
function SciMLBase.init(prob::DualAbstractLinearProblem, alg::SparspakFactorization, args...; kwargs...)
    return __init(prob, alg, args...; kwargs...)
end

function SciMLBase.init(prob::DualAbstractLinearProblem, alg::DefaultLinearSolver, args...; kwargs...)
    if alg.alg === DefaultAlgorithmChoice.GenericLUFactorization
        return __init(prob, alg, args...; kwargs...)
    else
        return __dual_init(prob, alg, args...; kwargs...)
    end
end

function SciMLBase.init(prob::DualAbstractLinearProblem, alg::Nothing,
        args...;
        assumptions = OperatorAssumptions(issquare(prob.A)),
        kwargs...)
    new_A = nodual_value(prob.A)
    new_b = nodual_value(prob.b)
    SciMLBase.init(
        prob, defaultalg(new_A, new_b, assumptions), args...; assumptions, kwargs...)
end

function __dual_init(
        prob::DualAbstractLinearProblem, alg::SciMLLinearSolveAlgorithm,
        args...;
        alias = LinearAliasSpecifier(),
        abstol = LinearSolve.default_tol(real(eltype(prob.b))),
        reltol = LinearSolve.default_tol(real(eltype(prob.b))),
        maxiters::Int = length(prob.b),
        verbose = LinearVerbosity(SciMLLogging.None()),
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

    primal_prob = remake(prob; A = new_A, b = new_b, u0 = new_u0)

    if get_dual_type(prob.A) !== nothing
        dual_type = get_dual_type(prob.A)
    elseif get_dual_type(prob.b) !== nothing
        dual_type = get_dual_type(prob.b)
    end

    non_partial_cache = init(
        primal_prob, alg, assumptions, args...;
        alias = alias, abstol = abstol, reltol = reltol,
        maxiters = maxiters, verbose = verbose, Pl = Pl, Pr = Pr, assumptions = assumptions,
        sensealg = sensealg, u0 = new_u0, kwargs...)

    # Initialize caches for partials lists and intermediate calculations
    partials_A_list = !isnothing(∂_A) ? partials_to_list(∂_A) : nothing
    partials_b_list = !isnothing(∂_b) ? partials_to_list(∂_b) : nothing

    # Determine size and type for rhs_list
    if !isnothing(partials_A_list)
        n_partials = length(partials_A_list)
        rhs_list = [similar(non_partial_cache.b) for _ in 1:n_partials]
    elseif !isnothing(partials_b_list)
        n_partials = length(partials_b_list)
        rhs_list = [similar(non_partial_cache.b) for _ in 1:n_partials]
    else
        rhs_list = nothing
    end

    return DualLinearCache{dual_type}(
        non_partial_cache,
        ∂_A,
        ∂_b,
        !isnothing(∂_b) ? zero.(∂_b) : ∂_b,
        partials_A_list,
        partials_b_list,
        rhs_list,
        similar(new_b),
        similar(new_b),
        similar(new_b),
        true,  # Cache is initially valid
        A,
        b,
        ArrayInterface.restructure(b, zeros(dual_type, length(b)))
    )
end

function SciMLBase.solve!(cache::DualLinearCache, args...; kwargs...)
    solve!(cache, getfield(cache, :linear_cache).alg, args...; kwargs...)
end

function SciMLBase.solve!(
        cache::DualLinearCache{DT}, alg::SciMLLinearSolveAlgorithm, args...; kwargs...) where {DT <:
                                                                                               ForwardDiff.Dual}
    primal_sol = linearsolve_forwarddiff_solve!(
        cache::DualLinearCache, getfield(cache, :linear_cache).alg, args...; kwargs...)
    dual_sol = linearsolve_dual_solution(getfield(cache, :linear_cache).u, getfield(cache, :rhs_list), cache)

    # For scalars, we still need to assign since cache.dual_u might not be pre-allocated
    if !(getfield(cache, :dual_u) isa AbstractArray)
        setfield!(cache, :dual_u, dual_sol)
    end

    return SciMLBase.build_linear_solution(
        getfield(cache, :linear_cache).alg, getfield(cache, :dual_u), primal_sol.resid, cache;
        primal_sol.retcode, primal_sol.iters, primal_sol.stats
    )
end

function setA!(dc::DualLinearCache, A)
    # Put the Dual-stripped versions in the LinearCache
    prop = nodual_value!(getproperty(dc.linear_cache, :A), A) # Update in-place
    setproperty!(dc.linear_cache, :A, prop) # Does additional invalidation logic etc.

    # Update partials
    setfield!(dc, :dual_A, A)
    partial_vals!(getfield(dc, :partials_A), A) # Update in-place

    # Invalidate cache (if setting A or b)
    setfield!(dc, :rhs_cache_valid, false)
end
function setb!(dc::DualLinearCache, b)
    # Put the Dual-stripped versions in the LinearCache
    prop = nodual_value!(getproperty(dc.linear_cache, :b), b) # Update in-place
    setproperty!(dc.linear_cache, :b, prop) # Does additional invalidation logic etc.

    # Update partials
    setfield!(dc, :dual_b, b)
    partial_vals!(getfield(dc, :partials_b), b) # Update in-place

    # Invalidate cache (if setting A or b)
    setfield!(dc, :rhs_cache_valid, false)
end
function setu!(dc::DualLinearCache, u)
    # Put the Dual-stripped versions in the LinearCache
    prop = nodual_value!(getproperty(dc.linear_cache, :u), u) # Update in-place
    setproperty!(dc.linear_cache, :u, prop) # Does additional invalidation logic etc.

    # Update partials
    setfield!(dc, :dual_u, u)
    partial_vals!(getfield(dc, :partials_u), u) # Update in-place
end

function SciMLBase.reinit!(cache::DualLinearCache;
        A = nothing,
        b = nothing,
        u = nothing,
        p = nothing,
        reuse_precs = false)
    linear_cache = getfield(cache, :linear_cache)

    # Compute freshness flags like in LinearCache reinit!
    isfresh = !isnothing(A)
    precsisfresh = !reuse_precs && (isfresh || !isnothing(p))
    isfresh |= linear_cache.isfresh
    precsisfresh |= linear_cache.precsisfresh

    # Update A if provided
    if !isnothing(A)
        # Update the primal value in linear_cache
        nodual_value!(linear_cache.A, A)
        # Update dual_A
        setfield!(cache, :dual_A, A)
        # Update partials_A
        partial_vals!(getfield(cache, :partials_A), A)
        # Update partials_A_list from new partials
        partials_A = getfield(cache, :partials_A)
        partials_A_list = getfield(cache, :partials_A_list)
        if !isnothing(partials_A) && !isnothing(partials_A_list)
            update_partials_list!(partials_A, partials_A_list)
        end
        # Invalidate RHS cache
        setfield!(cache, :rhs_cache_valid, false)
    end

    # Update b if provided
    if !isnothing(b)
        # Update the primal value in linear_cache
        nodual_value!(linear_cache.b, b)
        # Update dual_b
        setfield!(cache, :dual_b, b)
        # Update partials_b
        partial_vals!(getfield(cache, :partials_b), b)
        # Update partials_b_list from new partials
        partials_b = getfield(cache, :partials_b)
        partials_b_list = getfield(cache, :partials_b_list)
        if !isnothing(partials_b) && !isnothing(partials_b_list)
            update_partials_list!(partials_b, partials_b_list)
        end
        # Invalidate RHS cache
        setfield!(cache, :rhs_cache_valid, false)
    end

    # Update u if provided
    if !isnothing(u)
        # Update the primal value in linear_cache
        nodual_value!(linear_cache.u, u)
        # Update dual_u
        setfield!(cache, :dual_u, u)
        # Update partials_u
        partial_vals!(getfield(cache, :partials_u), u)
    end

    # Update p if provided
    if !isnothing(p)
        linear_cache.p = p
    end

    # Set freshness flags on linear_cache
    linear_cache.isfresh = true
    linear_cache.precsisfresh = precsisfresh

    nothing
end

function Base.setproperty!(dc::DualLinearCache, sym::Symbol, val)
    # If the property is A or b, also update it in the LinearCache
    if sym === :A
        setA!(dc, val)
    elseif sym === :b
        setb!(dc, val)
    elseif sym === :u
        setu!(dc, val)
    elseif hasfield(DualLinearCache, sym)
        setfield!(dc, sym, val)
    elseif hasfield(LinearSolve.LinearCache, sym)
        setproperty!(dc.linear_cache, sym, val)
    end
    nothing
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
partial_vals!(out, x) = map!(partial_vals, out, x) # Update in-place

# Add recursive handling for nested dual values
nodual_value(x) = x
nodual_value(x::Dual{T, V, P}) where {T, V <: AbstractFloat, P} = ForwardDiff.value(x)
nodual_value(x::Dual{T, V, P}) where {T, V <: Dual, P} = x.value  # Keep the inner dual intact
function nodual_value(x::AbstractArray{<:Dual})
    nodual_value!(similar(x, typeof(nodual_value(first(x)))), x)
end
nodual_value!(out, x) = map!(nodual_value, out, x) # Update in-place

function update_partials_list!(partial_matrix::AbstractVector{T}, list_cache) where {T}
    p = eachindex(first(partial_matrix))
    for i in p
        for j in eachindex(partial_matrix)
            @inbounds list_cache[i][j] = partial_matrix[j][i]
        end
    end
    return list_cache
end

function update_partials_list!(partial_matrix, list_cache)
    p = length(first(partial_matrix))
    m, n = size(partial_matrix)

    for k in 1:p
        for i in 1:m
            for j in 1:n
                @inbounds list_cache[k][i, j] = partial_matrix[i, j][k]
            end
        end
    end
    return list_cache
end

function partials_to_list(partial_matrix::AbstractVector{T}) where {T}
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
