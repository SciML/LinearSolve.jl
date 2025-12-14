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
    # Check if A is square - if not, use the non-square system path
    A = cache.linear_cache.A

    if !issquare(A)
        # For overdetermined systems, differentiate the normal equations: A'Ax = A'b
        # Taking d/dθ of both sides:
        # dA'/dθ · Ax + A' · dA/dθ · x + A'A · dx/dθ = dA'/dθ · b + A' · db/dθ
        # Rearranging:
        # A'A · dx/dθ = A' · db/dθ + dA'/dθ · (b  - Ax) - A' · dA/dθ · x

        # Solve the primal problem first
        cache.dual_u0_cache .= cache.linear_cache.u
        sol = solve!(cache.linear_cache, alg, args...; kwargs...)
        cache.primal_u_cache .= cache.linear_cache.u
        cache.primal_b_cache .= cache.linear_cache.b
        u = sol.u

        # Get the partials and primal values
        # After solve!, cache.linear_cache.A may be modified by factorization,
        # so we extract primal A from the original dual_A stored in cache
        ∂_A = cache.partials_A
        ∂_b = cache.partials_b
        A = nodual_value(cache.dual_A)
        A_adj = A'
        b = cache.primal_b_cache
        residual = b - A * u  # residual r = b - Ax

        rhs_list = cache.rhs_list

        # Update cached partials lists if cache is invalid
        if !cache.rhs_cache_valid
            if !isnothing(∂_A)
                update_partials_list!(∂_A, cache.partials_A_list)
            end
            if !isnothing(∂_b)
                update_partials_list!(∂_b, cache.partials_b_list)
            end
            cache.rhs_cache_valid = true
        end

        A_list = cache.partials_A_list
        b_list = cache.partials_b_list

        # Compute RHS: A' · db/dθ + dA'/dθ · (b - Ax) - A' · dA/dθ · x
        for i in eachindex(rhs_list)
            if !isnothing(b_list)
                # A' · db/dθ
                rhs_list[i] .= A_adj * b_list[i]
            else
                fill!(rhs_list[i], 0)
            end

            if !isnothing(A_list)
                # Add dA'/dθ · (b - Ax) = (dA/dθ)' · residual
                rhs_list[i] .+= A_list[i]' * residual
                # Subtract A' · dA/dθ · x
                temp = A_list[i] * u
                rhs_list[i] .-= A_adj * temp
            end
        end

        for i in eachindex(rhs_list)
            cache.linear_cache.b .= A_adj \ rhs_list[i]
            rhs_list[i] .= solve!(cache.linear_cache, alg, args...; kwargs...).u
        end

        cache.linear_cache.b .= cache.primal_b_cache
        cache.linear_cache.u .= cache.primal_u_cache

        return sol
    end

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

# NOTE: Removed GenericLUFactorization opt-out from init to fix type inference.
# The special handling for GenericLUFactorization is now done in solve! instead.
# This ensures init always returns DualLinearCache for type stability.

# Opt out for SparspakFactorization (sparse solvers can't handle Duals in the same way)
function SciMLBase.init(prob::DualAbstractLinearProblem, alg::SparspakFactorization, args...; kwargs...)
    return __init(prob, alg, args...; kwargs...)
end

# NOTE: Removed the runtime conditional for DefaultLinearSolver that checked for
# GenericLUFactorization. Now always use __dual_init for type stability.
function SciMLBase.init(prob::DualAbstractLinearProblem, alg::DefaultLinearSolver, args...; kwargs...)
    return __dual_init(prob, alg, args...; kwargs...)
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
    # For square systems, use b size. For overdetermined, use u size (solution size)
    rhs_template = length(non_partial_cache.u) == length(non_partial_cache.b) ?
                   non_partial_cache.b : non_partial_cache.u

    if !isnothing(partials_A_list)
        n_partials = length(partials_A_list)
        rhs_list = [similar(rhs_template) for _ in 1:n_partials]
    elseif !isnothing(partials_b_list)
        n_partials = length(partials_b_list)
        rhs_list = [similar(rhs_template) for _ in 1:n_partials]
    else
        rhs_list = nothing
    end
    # Use b for restructuring if sizes match (square system), otherwise use u (non-square)
    # This preserves ComponentArray structure from b when possible
    dual_u_init = if length(non_partial_cache.u) == length(b)
        ArrayInterface.restructure(b, zeros(dual_type, length(b)))
    else
        ArrayInterface.restructure(non_partial_cache.u, zeros(dual_type, length(non_partial_cache.u)))
    end

    return DualLinearCache{dual_type}(
        non_partial_cache,
        ∂_A,
        ∂_b,
        !isnothing(∂_b) ? zero.(∂_b) : ∂_b,
        partials_A_list,
        partials_b_list,
        rhs_list,
        similar(non_partial_cache.u),  # Use u's size, not b's size
        similar(non_partial_cache.u),  # primal_u_cache
        similar(new_b),                # primal_b_cache
        true,  # Cache is initially valid
        A,
        b,
        dual_u_init
    )
end

function SciMLBase.solve!(cache::DualLinearCache, args...; kwargs...)
    solve!(cache, getfield(cache, :linear_cache).alg, args...; kwargs...)
end

# Check if the algorithm should use the direct dual solve path
# (algorithms that can work directly with Dual numbers without the primal/partials separation)
function _use_direct_dual_solve(alg)
    return alg isa GenericLUFactorization
end

function _use_direct_dual_solve(alg::DefaultLinearSolver)
    return alg.alg === DefaultAlgorithmChoice.GenericLUFactorization
end

function SciMLBase.solve!(
        cache::DualLinearCache{DT}, alg::SciMLLinearSolveAlgorithm, args...; kwargs...) where {DT <:
                                                                                               ForwardDiff.Dual}
    # Check if this algorithm can work directly with Duals (e.g., GenericLUFactorization)
    # In that case, we solve the dual problem directly without separating primal/partials
    if _use_direct_dual_solve(getfield(cache, :linear_cache).alg)
        return _solve_direct_dual!(cache, alg, args...; kwargs...)
    end

    primal_sol = linearsolve_forwarddiff_solve!(
        cache::DualLinearCache, getfield(cache, :linear_cache).alg, args...; kwargs...)

    # Construct dual solution from primal solution and partials
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

# Direct solve path for algorithms that can work with Dual numbers directly
# This avoids the primal/partials separation overhead
function _solve_direct_dual!(
        cache::DualLinearCache{DT}, alg, args...; kwargs...) where {DT <: ForwardDiff.Dual}
    # Reconstruct the dual A and b
    dual_A = getfield(cache, :dual_A)
    dual_b = getfield(cache, :dual_b)

    # Solve directly with Duals using the generic LU path
    # This works because GenericLUFactorization doesn't use BLAS and can handle any number type
    dual_u = dual_A \ dual_b

    # Update the cache
    if getfield(cache, :dual_u) isa AbstractArray
        getfield(cache, :dual_u) .= dual_u
    else
        setfield!(cache, :dual_u, dual_u)
    end

    # Also update the primal cache for consistency
    primal_u = nodual_value.(dual_u)
    if getfield(cache, :linear_cache).u isa AbstractArray
        getfield(cache, :linear_cache).u .= primal_u
    end

    return SciMLBase.build_linear_solution(
        getfield(cache, :linear_cache).alg, getfield(cache, :dual_u), nothing, cache;
        retcode = ReturnCode.Success, iters = 1, stats = nothing
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
    if !isnothing(A)
        setA!(cache, A)
    end

    if !isnothing(b)
        setb!(cache, b)
    end

    if !isnothing(u)
        setu!(cache, u)
    end

    if !isnothing(p)
        cache.linear_cache.p=p
    end

    isfresh = !isnothing(A)
    precsisfresh = !reuse_precs && (isfresh || !isnothing(p))
    isfresh |= cache.linear_cache.isfresh
    precsisfresh |= cache.linear_cache.precsisfresh
    cache.linear_cache.isfresh = true
    cache.linear_cache.precsisfresh = precsisfresh

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
