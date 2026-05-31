module LinearSolvePartitionedSolversExt

using LinearAlgebra: UniformScaling, norm
using PartitionedArrays: PSparseMatrix, PVector
using PartitionedSolvers: PartitionedSolvers
using LinearSolve: LinearSolve, PartitionedSolversAlgorithm, LinearCache, LinearProblem,
    init_cacheval, __init, OperatorAssumptions, LinearVerbosity, defaultalg
using SciMLBase: ReturnCode, SciMLBase
using SciMLOperators: IdentityOperator

mutable struct PartitionedSolversCache
    solver::Any
end

function LinearSolve.init_cacheval(
        alg::PartitionedSolversAlgorithm, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PartitionedSolversCache(nothing)
end

function validate_partitionedsolvers_problem(A, b, u0)
    A isa PSparseMatrix || throw(
        ArgumentError(
            "PartitionedSolversAlgorithm requires A::PSparseMatrix, got $(typeof(A))"
        )
    )
    b isa PVector || throw(
        ArgumentError("PartitionedSolversAlgorithm requires b::PVector, got $(typeof(b))")
    )
    (u0 === nothing || u0 isa PVector) || throw(
        ArgumentError(
            "PartitionedSolversAlgorithm requires u0::PVector when provided, got $(typeof(u0))"
        )
    )
    return nothing
end

function SciMLBase.init(prob::LinearProblem, alg::PartitionedSolversAlgorithm, args...; kwargs...)
    (; A, b, u0, p) = prob
    validate_partitionedsolvers_problem(A, b, u0)
    u0_ = u0 === nothing ? zero(b) : u0
    prob_ = u0 === u0_ ? prob : LinearProblem(A, b, p; u0 = u0_)
    return __init(prob_, alg, args...; kwargs...)
end

function LinearSolve.defaultalg(
        A::PSparseMatrix, b::PVector, assump::OperatorAssumptions{Bool}
    )
    assump.issq || error(
        "PartitionedSolversAlgorithm currently only supports square PSparseMatrix problems"
    )
    return PartitionedSolversAlgorithm(PartitionedSolvers.cg)
end

# Discover which keyword arguments a PartitionedSolvers solver constructor accepts so that
# auto-derived convergence options are only forwarded to solvers that understand them. A
# constructor that slurps keywords (e.g. `cg(p; kwargs...)`) accepts everything, while a
# fixed-signature solver (e.g. `amg(p; fine_params, coarse_params)`) only takes its declared
# keywords. This keeps the integration solver-agnostic instead of being implicitly cg-only.
function accepted_solver_kwargs(solver)
    names = Symbol[]
    for m in methods(solver)
        for k in Base.kwarg_decl(m)
            endswith(string(k), "...") && return (true, names)
            push!(names, k)
        end
    end
    return (false, unique(names))
end

function filter_solver_kwargs(solver, kwargs::NamedTuple)
    slurps, names = accepted_solver_kwargs(solver)
    slurps && return kwargs
    kept = filter(p -> first(p) in names, pairs(kwargs))
    return NamedTuple(kept)
end

function auto_convergence_kwargs(cache::LinearCache)
    kwargs = (;
        iterations = cache.maxiters,
        abstol = cache.abstol,
        reltol = cache.reltol,
        verbose = LinearSolve.verbosity_to_int(cache.verbose.KrylovJL_verbosity) > 0,
    )
    if !(cache.Pl isa Union{UniformScaling, IdentityOperator})
        kwargs = merge(kwargs, (; Pl = cache.Pl))
    else
        kwargs = merge(kwargs, (; update_Pl = false))
    end
    return kwargs
end

function partitionedsolvers_solver_kwargs(cache::LinearCache, alg::PartitionedSolversAlgorithm)
    # Forward only the auto-derived options the solver actually accepts, then let any
    # explicit user-provided keywords take precedence (and always pass through).
    auto = filter_solver_kwargs(alg.solver, auto_convergence_kwargs(cache))
    return merge(auto, alg.kwargs)
end

partitionedsolvers_problem(cache::LinearCache) = PartitionedSolvers.linear_problem(
    cache.u, cache.A, cache.b
)

function init_partitionedsolvers_solver(
        cache::LinearCache, alg::PartitionedSolversAlgorithm, problem
    )
    solver = alg.solver
    if solver === nothing
        return PartitionedSolvers.default_solver(problem)
    elseif solver isa PartitionedSolvers.AbstractSolver
        return PartitionedSolvers.update(solver; problem)
    else
        return solver(problem; partitionedsolvers_solver_kwargs(cache, alg)...)
    end
end

function update_partitionedsolvers_solver(
        cache::LinearCache, alg::PartitionedSolversAlgorithm, solver, problem
    )
    solver === nothing && return init_partitionedsolvers_solver(cache, alg, problem)
    return PartitionedSolvers.update(
        solver; problem, matrix = cache.A, rhs = cache.b, solution = cache.u
    )
end

function partitionedsolvers_result_metadata(solver)
    workspace = PartitionedSolvers.workspace(solver)
    hasproperty(workspace, :state) || return nothing, ReturnCode.Success, 0
    state = getproperty(workspace, :state)
    resid = hasproperty(state, :current) ? getproperty(state, :current) : nothing
    iters = hasproperty(state, :iteration) ? Int(getproperty(state, :iteration)) : 0
    target = hasproperty(state, :target) ? getproperty(state, :target) : nothing
    if resid === nothing
        return nothing, ReturnCode.Success, iters
    elseif !isfinite(resid)
        return resid, ReturnCode.Failure, iters
    elseif target !== nothing && resid <= target
        return resid, ReturnCode.Success, iters
    else
        return resid, ReturnCode.MaxIters, iters
    end
end

function SciMLBase.solve!(cache::LinearCache, alg::PartitionedSolversAlgorithm, args...; kwargs...)
    problem = partitionedsolvers_problem(cache)
    solver = update_partitionedsolvers_solver(cache, alg, cache.cacheval.solver, problem)
    solver = PartitionedSolvers.solve(solver)
    setfield!(cache.cacheval, :solver, solver)
    setfield!(cache, :u, PartitionedSolvers.solution(solver))

    resid, retcode, iters = partitionedsolvers_result_metadata(solver)
    if resid === nothing
        resid = norm(cache.A * cache.u - cache.b)
        retcode = isfinite(resid) ? ReturnCode.Success : ReturnCode.Failure
    end

    return SciMLBase.build_linear_solution(alg, cache.u, resid, cache; retcode, iters)
end

end # module LinearSolvePartitionedSolversExt
