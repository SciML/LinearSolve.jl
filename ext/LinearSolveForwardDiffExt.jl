module LinearSolveForwardDiffExt

using LinearSolve: LinearSolve, SciMLLinearSolveAlgorithm, __init, LinearVerbosity,
    DefaultLinearSolver, GenericLUFactorization, LinearSolveAdjoint,
    OperatorAssumptions, PureKLUFactorization, SparspakFactorization, defaultalg,
    default_alias_A
using ConcreteStructs: @concrete
using LinearAlgebra: LinearAlgebra, mul!
using SparseArrays: SparseArrays, SparseMatrixCSC, nonzeros
using ForwardDiff: ForwardDiff, Dual, Partials
using SciMLBase: SciMLBase, LinearAliasSpecifier, LinearProblem, init, solve, solve!
using SciMLOperators: issquare
using RecursiveArrayTools: RecursiveArrayTools
using SciMLLogging: SciMLLogging
using ArrayInterface: ArrayInterface

const DualLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Any,
} where {iip, T, V, P}

const DualALinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing},
    iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Union{Number, <:AbstractArray},
    <:Any,
} where {iip, T, V, P}

const DualBLinearProblem = LinearProblem{
    <:Union{Number, <:AbstractArray, Nothing},
    iip,
    <:Union{Number, <:AbstractArray},
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
    <:Any,
} where {iip, T, V, P}

const DualAbstractLinearProblem = Union{
    DualLinearProblem, DualALinearProblem, DualBLinearProblem,
}

@concrete mutable struct DualLinearCache{DT}
    linear_cache

    partials_A
    partials_b
    partials_u

    # Cached lists of partials to avoid repeated allocations
    partials_A_list
    partials_b_list

    # Cached intermediate values for calculations
    rhs_list
    # p x m gemv output for the fused xp_linsolve_rhs!, laid out to match
    # `reinterpret` of ∂A. Sized from ∂A's rows, which differ from length(b) for a
    # non-square system. `nothing` when there is no dense Dual matrix to sweep.
    partials_scratch
    # `vec` of the above, aliasing the same buffer. Held rather than recomputed
    # because the gemv wants the flat form and the scatter wants the p x m form,
    # and `vec` escapes into BLAS, so it cannot be elided at the call site.
    partials_scratch_flat
    dual_u0_cache
    primal_u_cache
    primal_b_cache

    # Cache validity flags for when partials of A or b changes
    A_partials_valid
    b_partials_valid

    dual_A
    dual_b
    dual_u

    # Cached LinearCache for direct Dual path (nothing for split path algorithms)
    dual_linear_cache
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

        if !cache.A_partials_valid && !isnothing(∂_A)
            update_partials_list!(∂_A, cache.partials_A_list)
            cache.A_partials_valid = true
        end
        if !cache.b_partials_valid && !isnothing(∂_b)
            update_partials_list!(∂_b, cache.partials_b_list)
            cache.b_partials_valid = true
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

# Fused rhs construction for a dense Dual matrix.
#
# The generic method below materialises `p` separate derivative matrices via
# update_partials_list! and then issues `p` mul! calls, i.e. p+1 passes over ∂A.
# That materialisation exists only so each matvec can reach BLAS gemv.
#
# A `Partials{p,V}` stores its p components contiguously, so reinterpreting a
# dense m x n array of them as `V` already gives a strided (p*m) x n matrix whose
# row (i-1)*p + k holds partial k of row i — the transpose is a no-op view rather
# than a copy. All p right-hand sides rhs_k = ∂b_k - (∂A_k) * u are then a single
# gemv against that view: one pass over ∂A, nothing materialised, and the work
# still goes through BLAS (or CUBLAS, for a GPU-resident ∂A).
#
# `partials_scratch` receives the result: it is p x m, matching the reinterpreted
# layout element for element, so `vec` of it is the gemv output vector. rhs_list
# is p separate vectors, which is why the result cannot be accumulated in place.
#
# Dense, not AbstractMatrix: a SparseMatrixCSC{<:Partials} is an
# AbstractMatrix{<:Partials} and would be captured here, but its partials are not
# a contiguous block to reinterpret. Sparse ∂A stays on the list-based method
# below, which has a sparsity-aware update_partials_list!.
#
# Small problems take a hand-written loop instead, see _use_scalar_dual_rhs.
function xp_linsolve_rhs!(
        uu, ∂_A::DenseMatrix{<:Partials},
        ∂_b::DenseVector{<:Partials}, cache::DualLinearCache
    )
    if _use_scalar_dual_rhs(∂_A, ∂_b, uu)
        return _xp_linsolve_rhs_scalar!(uu, ∂_A, ∂_b, cache)
    end
    return _xp_linsolve_rhs_gemv!(uu, ∂_A, ∂_b, cache)
end

# Work below which the loop beats the gemv, counted in scalar multiply-adds
# (m * n * p). The gemv's advantage is asymptotic -- it pays a fixed BLAS call
# overhead the loop does not -- so on the kernel alone the loop is ~3x faster at
# 5x5 and ~1.4x at 10x10, the two are level around 20x20, and the gemv pulls
# ahead from there (to 4x at 200x200 under OpenBLAS).
#
# The exact value is a hedge rather than a sharp optimum, for two reasons. The
# crossover is BLAS-dependent: OpenBLAS turns over near 20x20, MKL's gemv is
# weaker for these tall-skinny shapes and does not turn over until nearer
# 100x100. And end-to-end the choice barely registers -- whole `solve!` differs
# by under 3% either way at every size measured, because the rhs construction is
# a small share of a solve. This sits between the two crossovers, so neither
# backend is badly served.
const DUAL_RHS_GEMV_CUTOFF = 30_000

# Only a plain Array is eligible for the loop: it indexes elements scalarly,
# which is exactly what a GPU array forbids. Anything else -- a CuArray, a
# JLArray -- always takes the gemv, where the work lands in CUBLAS.
@inline function _use_scalar_dual_rhs(∂_A::Array, ∂_b::Array, uu::Array)
    return length(∂_A) * ForwardDiff.npartials(eltype(∂_A)) < DUAL_RHS_GEMV_CUTOFF
end
@inline _use_scalar_dual_rhs(∂_A, ∂_b, uu) = false

function _xp_linsolve_rhs_gemv!(uu, ∂_A, ∂_b, cache::DualLinearCache)
    rhs_list = cache.rhs_list
    scratch = cache.partials_scratch
    rhs_flat = cache.partials_scratch_flat
    V = eltype(scratch)

    rhs_flat .= reinterpret(V, ∂_b)
    mul!(rhs_flat, reinterpret(V, ∂_A), uu, -1, 1)

    for k in eachindex(rhs_list)
        rhs_list[k] .= view(scratch, k, :)
    end

    return rhs_list
end

# Same arithmetic as the gemv path, accumulated by hand into the same p x m
# scratch. The partial index is innermost so the accumulator is walked
# contiguously and ∂A is still read exactly once.
function _xp_linsolve_rhs_scalar!(uu, ∂_A, ∂_b, cache::DualLinearCache)
    rhs_list = cache.rhs_list
    scratch = cache.partials_scratch
    m, n = size(∂_A)
    # From the type, not size(scratch, 1): as a compile-time constant the
    # innermost loop over the partial index unrolls, which is most of the point
    # of this path. A runtime `p` measures several times slower.
    p = ForwardDiff.npartials(eltype(∂_A))

    @inbounds for i in 1:m
        bi = ∂_b[i]
        for k in 1:p
            scratch[k, i] = bi[k]
        end
    end
    @inbounds for j in 1:n
        uj = uu[j]
        for i in 1:m
            aij = ∂_A[i, j]
            for k in 1:p
                scratch[k, i] -= aij[k] * uj
            end
        end
    end
    @inbounds for k in 1:p
        rk = rhs_list[k]
        for i in 1:m
            rk[i] = scratch[k, i]
        end
    end

    return rhs_list
end

function xp_linsolve_rhs!(
        uu, ∂_A::Union{<:Partials, <:AbstractArray{<:Partials}},
        ∂_b::Union{<:Partials, <:AbstractArray{<:Partials}}, cache::DualLinearCache
    )

    if !cache.A_partials_valid
        update_partials_list!(∂_A, cache.partials_A_list)
        cache.A_partials_valid = true
    end
    if !cache.b_partials_valid
        update_partials_list!(∂_b, cache.partials_b_list)
        cache.b_partials_valid = true
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
        ∂_b::Nothing, cache::DualLinearCache
    )

    if !cache.A_partials_valid
        update_partials_list!(∂_A, cache.partials_A_list)
        cache.A_partials_valid = true
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
        cache::DualLinearCache
    )

    if !cache.b_partials_valid
        update_partials_list!(∂_b, cache.partials_b_list)
        cache.b_partials_valid = true
    end

    b_list = cache.partials_b_list

    # Copy precalculated b_list to rhs_list (no A*uu computation needed)
    for i in eachindex(b_list)
        cache.rhs_list[i] .= b_list[i]
    end

    return cache.rhs_list
end

function linearsolve_dual_solution(
        u::Number, partials, cache::DualLinearCache{DT}
    ) where {DT}
    return DT(u, partials)
end

function linearsolve_dual_solution(
        u::AbstractArray, partials,
        cache::DualLinearCache{DT}
    ) where {T, V, N, DT <: Dual{T, V, N}}
    # Optimized in-place version that reuses cache.dual_u
    linearsolve_dual_solution!(getfield(cache, :dual_u), u, partials)
    return getfield(cache, :dual_u)
end

function linearsolve_dual_solution!(
        dual_u::AbstractArray{DT}, u::AbstractArray,
        partials
    ) where {T, V, N, DT <: Dual{T, V, N}}
    # Broadcast rather than an indexed loop so this runs on a GPU-resident u.
    # `partials` is a Vector of N arrays, one per partial: hoisting it into a
    # tuple first keeps the per-element work a plain combine over N arrays,
    # with no indexing into `partials` left inside the kernel.
    partial_arrays = ntuple(j -> partials[j], Val(N))
    dual_u .= ((uu, pp...) -> DT(uu, Partials{N, V}(convert.(V, pp)))).(u, partial_arrays...)

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

# Duals only in b (A is primal): route PureKLU to a plain LinearCache and solve natively.
# PureKLU's mixed-type `ldiv!` (primal KLU factor against a Dual RHS, from PureKLUForwardDiffExt)
# keeps the factorization in Float64 and pushes the duals through the back-substitution, so
# there is no reason to build the split DualLinearCache here. This is type-stable: dispatch is
# purely on the problem
# subtype (b-dual / A-plain) and the alg type, so `init` always returns a `LinearCache` for
# this method. It also gets correct factorization reuse across b-only `reinit!`s for free,
# which the split path's `reinit!` does not (it always marks the inner cache fresh).
function SciMLBase.init(prob::DualBLinearProblem, alg::PureKLUFactorization, args...; kwargs...)
    return __init(prob, alg, args...; kwargs...)
end

# DualBLinearProblem's A slot is `Union{Number, AbstractArray}`, which a Dual-eltype A
# also matches, so the b-only opt-out above would otherwise capture both-Dual problems
# (`DualLinearProblem <: DualBLinearProblem`). Restore the direct dual path for them: a
# Dual A needs the DualLinearCache machinery, the mixed-type ldiv! only covers primal A.
function SciMLBase.init(prob::DualLinearProblem, alg::PureKLUFactorization, args...; kwargs...)
    return __dual_init(prob, alg, args...; kwargs...)
end

# NOTE: Removed the runtime conditional for DefaultLinearSolver that checked for
# GenericLUFactorization. Now always use __dual_init for type stability.
function SciMLBase.init(prob::DualAbstractLinearProblem, alg::DefaultLinearSolver, args...; kwargs...)
    return __dual_init(prob, alg, args...; kwargs...)
end

function SciMLBase.init(
        prob::DualAbstractLinearProblem, alg::Nothing,
        args...;
        assumptions = OperatorAssumptions(issquare(prob.A)),
        kwargs...
    )
    new_A = nodual_value(prob.A)
    new_b = nodual_value(prob.b)
    return SciMLBase.init(
        prob, defaultalg(new_A, new_b, assumptions), args...; assumptions, kwargs...
    )
end

# `solve(prob)` resolves `alg::Nothing` itself (common.jl) before `init` is called,
# which would select from the *dual* A/b types. The algorithm always executes on the
# primal arrays (`nodual_value(A)`), so selection must see those instead: e.g. a
# reinterpret-wrapped Dual A is not a `DenseMatrix` and would select KrylovJL_GMRES,
# while the primal cache is a dense `Matrix` whose default-solver Krylov cacheval
# slots are initialized as `Nothing` (see `_init_default_cacheval`).
function SciMLBase.solve(
        prob::DualAbstractLinearProblem, ::Nothing, args...;
        assump = OperatorAssumptions(issquare(prob.A)), kwargs...
    )
    new_A = nodual_value(prob.A)
    new_b = nodual_value(prob.b)
    return solve(prob, defaultalg(new_A, new_b, assump), args...; kwargs...)
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
        kwargs...
    )
    (; A, b, u0, p) = prob
    new_A = nodual_value(A)
    new_b = nodual_value(b)
    new_u0 = nodual_value(u0)

    ∂_A = partial_vals(A)
    ∂_b = partial_vals(b)

    new_p = nodual_value(p)
    primal_prob = LinearProblem{SciMLBase.isinplace(prob)}(new_A, new_b, new_p; u0 = new_u0)

    if get_dual_type(prob.A) !== nothing
        dual_type = get_dual_type(prob.A)
    elseif get_dual_type(prob.b) !== nothing
        dual_type = get_dual_type(prob.b)
    end

    non_partial_cache = init(
        primal_prob, alg, assumptions, args...;
        alias = alias, abstol = abstol, reltol = reltol,
        maxiters = maxiters, verbose = verbose, Pl = Pl, Pr = Pr, assumptions = assumptions,
        sensealg = sensealg, u0 = new_u0, kwargs...
    )

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
    # Scratch for the fused rhs construction: p rows (partial index) x m columns,
    # matching the layout of `reinterpret(V, ∂_A)`. Allocated here rather than on
    # first use because DualLinearCache is @concrete: a field initialised to
    # `nothing` is typed Nothing and cannot later hold a matrix. The condition
    # mirrors the fused method's signature exactly, so that method can use the
    # field unconditionally; `∂_A`/`∂_b` keep their type and size for the life of
    # the cache, since setA!/setb! update them in place.
    partials_scratch = if ∂_A isa DenseMatrix{<:Partials} && ∂_b isa DenseVector{<:Partials}
        V = eltype(eltype(∂_A))
        similar(∂_A, V, ForwardDiff.npartials(eltype(∂_A)), size(∂_A, 1))
    else
        nothing
    end
    partials_scratch_flat = isnothing(partials_scratch) ? nothing : vec(partials_scratch)

    # Use b for restructuring if sizes match (square system), otherwise use u (non-square)
    # This preserves ComponentArray structure from b when possible
    dual_u_init = if length(non_partial_cache.u) == length(b)
        ArrayInterface.restructure(b, zeros(dual_type, length(b)))
    else
        ArrayInterface.restructure(non_partial_cache.u, zeros(dual_type, length(non_partial_cache.u)))
    end

    # For algorithms taking the direct Dual path, use __init to create a regular LinearCache
    # (bypasses ForwardDiff extension) then solve! on that cache directly with the dual values
    # Promote b to Dual so that dc.b is typed correctly for later dc.b = dual_b
    # assignments in _solve_direct_dual! (b may be plain Float64 when only A is Dual).
    if _use_direct_dual_solve(alg)
        dual_b_init = eltype(b) <: ForwardDiff.Dual ? b : dual_type.(b)
        dual_linear_cache_init = __init(
            LinearProblem(A, dual_b_init), alg;
            alias, abstol, reltol, maxiters, verbose, Pl, Pr, assumptions, sensealg, kwargs...
        )
    else
        dual_linear_cache_init = nothing
    end

    return DualLinearCache{dual_type}(
        non_partial_cache,
        ∂_A,
        ∂_b,
        !isnothing(∂_b) ? zero.(∂_b) : ∂_b,
        partials_A_list,
        partials_b_list,
        rhs_list,
        partials_scratch,
        partials_scratch_flat,
        similar(non_partial_cache.u),  # Use u's size, not b's size
        similar(non_partial_cache.u),  # primal_u_cache
        similar(new_b),                # primal_b_cache
        true,  # Cache is initially valid
        true,
        A,
        b,
        dual_u_init,
        dual_linear_cache_init
    )
end

function SciMLBase.solve!(cache::DualLinearCache, args...; kwargs...)
    return solve!(cache, getfield(cache, :linear_cache).alg, args...; kwargs...)
end

# Check if the algorithm should use the direct dual solve path
# (algorithms that can work directly with Dual numbers without the primal/partials separation)
function _use_direct_dual_solve(alg)
    # NOTE: RFLUFactorization is intentionally *not* on the direct path. Even when
    # A carries duals, its fast Float64 factorization is BLAS/SIMD-grade, and routing
    # the Dual problem through it falls back to generic scalar dual arithmetic, losing
    # that speedup entirely (~40x slower, see issue #1052). The split path keeps the
    # fast primal factorization and reuses it across the partial back-solves.
    return alg isa GenericLUFactorization ||
        alg isa LinearSolve.SpecializedLUFactorization ||
        alg isa LinearSolve.SpecializedQRFactorization ||
        alg isa LinearSolve.PureKLUFactorization
end

function SciMLBase.solve!(
        cache::DualLinearCache{DT}, alg::SciMLLinearSolveAlgorithm, args...; kwargs...
    ) where {
        DT <:
        ForwardDiff.Dual,
    }
    # Check if this algorithm can work directly with Duals (e.g., GenericLUFactorization)
    # In that case, we solve the dual problem directly without separating primal/partials.
    # Only worthwhile when A itself carries duals: with duals just in b, the split
    # path (one primal factorization + partials back-solves) is strictly cheaper
    # than factorizing in dual arithmetic.
    if _use_direct_dual_solve(getfield(cache, :linear_cache).alg) &&
            get_dual_type(getfield(cache, :dual_A)) !== nothing
        return _solve_direct_dual!(cache, alg, args...; kwargs...)
    end

    primal_sol = linearsolve_forwarddiff_solve!(
        cache::DualLinearCache, getfield(cache, :linear_cache).alg, args...; kwargs...
    )

    # Construct dual solution from primal solution and partials
    dual_sol = linearsolve_dual_solution(getfield(cache, :linear_cache).u, getfield(cache, :rhs_list), cache)

    # For scalars, we still need to assign since cache.dual_u might not be pre-allocated
    if !(getfield(cache, :dual_u) isa AbstractArray)
        setfield!(cache, :dual_u, dual_sol)
    end

    return SciMLBase.build_linear_solution(
        getfield(cache, :linear_cache).alg, getfield(cache, :dual_u), primal_sol.resid, nothing;
        primal_sol.retcode, primal_sol.iters, primal_sol.stats
    )
end

# Direct solve path for algorithms that can work with Dual numbers directly
# This avoids the primal/partials separation overhead
# The inner dual LinearCache is created eagerly in __dual_init and reused here,
# mirroring how the split path reuses the primal factorisation across RHS vectors.
function _solve_direct_dual!(
        cache::DualLinearCache{DT}, alg, args...; kwargs...
    ) where {DT <: ForwardDiff.Dual}
    # Get the dual A and b
    dual_A = getfield(cache, :dual_A)
    dual_b = getfield(cache, :dual_b)

    # When only A carries duals, b is a plain array — promote it so the
    # factorisation kernel sees a uniform element type.
    if eltype(dual_b) != DT
        dual_b = DT.(dual_b)
    end

    # Get regular LinearCache prepared in __dual__init
    linear_cache = getfield(cache, :linear_cache)
    dual_cache = getfield(cache, :dual_linear_cache)

    # Update A (and trigger re-factorisation) when the outer primal cache signals
    # that A has changed via its isfresh flag, which is set by setA!.
    if linear_cache.isfresh
        dual_cache.A = default_alias_A(alg, dual_A, dual_b) ? dual_A : copy(dual_A)
    end
    dual_cache.b .= dual_b

    # solve! on the regular LinearCache directly with the dual values (bypasses ForwardDiff extension)
    dual_sol = SciMLBase.solve!(dual_cache)

    setfield!(linear_cache, :isfresh, false)

    # Update the cache
    if getfield(cache, :dual_u) isa AbstractArray
        getfield(cache, :dual_u) .= dual_sol.u
    else
        setfield!(cache, :dual_u, dual_sol.u)
    end

    # Also update the primal cache for consistency
    if linear_cache.u isa AbstractArray
        linear_cache.u .= nodual_value.(dual_sol.u)
    end

    return SciMLBase.build_linear_solution(
        linear_cache.alg, getfield(cache, :dual_u), dual_sol.resid, nothing;
        dual_sol.retcode, dual_sol.iters, dual_sol.stats
    )
end

function setA!(dc::DualLinearCache, A)
    # Put the Dual-stripped versions in the LinearCache
    prop = nodual_value!(getproperty(dc.linear_cache, :A), A) # Update in-place
    setproperty!(dc.linear_cache, :A, prop) # Does additional invalidation logic etc.

    # Update partials only when A actually carries Duals; otherwise there is
    # nothing to extract and the partials slot may be unallocated.
    setfield!(dc, :dual_A, A)
    if get_dual_type(A) !== nothing
        partial_vals!(getfield(dc, :partials_A), A)
    end

    return setfield!(dc, :A_partials_valid, false)
end
function setb!(dc::DualLinearCache, b)
    # Put the Dual-stripped versions in the LinearCache
    prop = nodual_value!(getproperty(dc.linear_cache, :b), b) # Update in-place
    setproperty!(dc.linear_cache, :b, prop) # Does additional invalidation logic etc.

    # Update partials only when b actually carries Duals; otherwise there is
    # nothing to extract and the partials slot may be unallocated.
    setfield!(dc, :dual_b, b)
    if get_dual_type(b) !== nothing
        partial_vals!(getfield(dc, :partials_b), b)
    end

    return setfield!(dc, :b_partials_valid, false)
end
function setu!(dc::DualLinearCache{DT}, u) where {DT}
    # Put the Dual-stripped versions in the LinearCache
    prop = nodual_value!(getproperty(dc.linear_cache, :u), u) # Update in-place
    setproperty!(dc.linear_cache, :u, prop) # Does additional invalidation logic etc.

    if get_dual_type(u) === nothing
        # `u` is primal-only (e.g. the Vector{Float64} iterate handed in by
        # `NonlinearSolveBase.set_lincache_u!` during Newton iterations under
        # an outer Hessian tag), while `dual_u` is statically typed
        # Vector{<:Dual}. Promote element-wise to `DT` with zero partials so
        # the field invariant is preserved without dropping derivatives — the
        # next solve! will rewrite the partials from the Dual A / b via
        # `linearsolve_dual_solution!`.
        dual_u_field = getfield(dc, :dual_u)
        if dual_u_field isa AbstractArray
            dual_u_field .= DT.(u)
        else
            setfield!(dc, :dual_u, DT(u))
        end
        pu = getfield(dc, :partials_u)
        pu === nothing || fill!(pu, zero(eltype(pu)))
        return nothing
    end

    setfield!(dc, :dual_u, u)
    return partial_vals!(getfield(dc, :partials_u), u)
end

function SciMLBase.reinit!(
        cache::DualLinearCache;
        A = nothing,
        b = nothing,
        u = nothing,
        p = nothing,
        reuse_precs = false
    )
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
        cache.linear_cache.p = nodual_value(p)
    end

    isfresh = !isnothing(A)
    precsisfresh = !reuse_precs && (isfresh || !isnothing(p))
    isfresh |= cache.linear_cache.isfresh
    precsisfresh |= cache.linear_cache.precsisfresh
    cache.linear_cache.isfresh = true
    cache.linear_cache.precsisfresh = precsisfresh

    return nothing
end

function Base.setproperty!(dc::DualLinearCache, sym::Symbol, val)
    # If the property is A or b, also update it in the LinearCache
    if sym === :A
        setA!(dc, val)
    elseif sym === :b
        setb!(dc, val)
    elseif sym === :u
        setu!(dc, val)
    elseif sym === :p
        setproperty!(dc.linear_cache, :p, nodual_value(val))
    elseif hasfield(DualLinearCache, sym)
        setfield!(dc, sym, val)
    elseif hasfield(LinearSolve.LinearCache, sym)
        setproperty!(dc.linear_cache, sym, val)
    end
    return nothing
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
nodual_value(x::Tuple) = map(nodual_value, x)
function nodual_value(x::AbstractArray{<:Dual})
    # valtype rather than `typeof(nodual_value(first(x)))`: identical answer for
    # both the AbstractFloat and the nested-Dual case above, without the scalar
    # read of `first(x)`, which a GPU array disallows.
    return nodual_value!(similar(x, ForwardDiff.valtype(eltype(x))), x)
end
nodual_value!(out, x) = map!(nodual_value, out, x) # Update in-place

# Both of these are one broadcast per partial index rather than a scalar loop,
# so a GPU-resident ∂A/∂b works. Kept as separate vector/matrix methods, not one
# AbstractArray method, so the SparseMatrixCSC specialisations below stay
# strictly more specific and no ambiguity is introduced.
function update_partials_list!(partial_matrix::AbstractVector{T}, list_cache) where {T}
    for k in eachindex(list_cache)
        list_cache[k] .= getindex.(partial_matrix, k)
    end
    return list_cache
end

function update_partials_list!(partial_matrix::AbstractMatrix{T}, list_cache) where {T}
    for k in eachindex(list_cache)
        list_cache[k] .= getindex.(partial_matrix, k)
    end
    return list_cache
end

function partials_to_list(partial_matrix::AbstractVector{T}) where {T}
    return [getindex.(partial_matrix, k) for k in 1:ForwardDiff.npartials(T)]
end

function partials_to_list(partial_matrix::AbstractMatrix{T}) where {T}
    return [getindex.(partial_matrix, k) for k in 1:ForwardDiff.npartials(T)]
end

# Specializations for sparse matrices

function partials_to_list(partial_matrix::SparseMatrixCSC)
    nz = nonzeros(partial_matrix)
    m, n = size(partial_matrix)
    T = eltype(partial_matrix)
    p = ForwardDiff.npartials(T)
    V = ForwardDiff.valtype(T) # use type for concrete array below in empty-nz case (e.g. all-zero Jacobian at init)
    return [
        SparseMatrixCSC(
                m, n, partial_matrix.colptr, partial_matrix.rowval,
                V[nz[i][k] for i in eachindex(nz)]
            ) for k in 1:p
    ]
end

function update_partials_list!(partial_matrix::SparseMatrixCSC, list_cache)
    nz = nonzeros(partial_matrix)
    if length(nz) != length(nonzeros(first(list_cache)))
        list_cache .= partials_to_list(partial_matrix) # sparsity pattern changed
    else
        for k in eachindex(list_cache)
            nz_k = nonzeros(list_cache[k])
            @inbounds for i in eachindex(nz, nz_k)
                nz_k[i] = nz[i][k]
            end
        end
    end
    return list_cache
end


# The MC64 matching in the vendored SupernodalLU solver runs its combinatorial
# side (the max-product assignment) in Float64, so it needs a real magnitude
# for a `Dual` entry.  Only the *ordering* of candidate pivots is decided from
# these numbers; the factorization itself stays in Dual arithmetic, so taking
# the primal here loses nothing.  Without this, `snlu` on a Dual matrix with a
# missing or weak structural diagonal - exactly when `matching = :auto`
# engages - throws `MethodError: no method matching Float64(::Dual)`.
@inline LinearSolve.SupernodalLU._costabs(x::Dual) =
    LinearSolve.SupernodalLU._costabs(ForwardDiff.value(x))

end
