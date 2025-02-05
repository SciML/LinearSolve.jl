module LinearSolveHYPREExt

using LinearAlgebra
using HYPRE.LibHYPRE: HYPRE_Complex
using HYPRE: HYPRE, HYPREMatrix, HYPRESolver, HYPREVector
using LinearSolve: HYPREAlgorithm, LinearCache, LinearProblem, LinearSolve,
                   OperatorAssumptions, default_tol, init_cacheval, __issquare,
                   __conditioning, LinearSolveAdjoint
using SciMLBase: LinearProblem, LinearAliasSpecifier, SciMLBase
using UnPack: @unpack
using Setfield: @set!

mutable struct HYPRECache
    solver::Union{HYPRE.HYPRESolver, Nothing}
    A::Union{HYPREMatrix, Nothing}
    b::Union{HYPREVector, Nothing}
    u::Union{HYPREVector, Nothing}
    isfresh_A::Bool
    isfresh_b::Bool
    isfresh_u::Bool
end

function LinearSolve.init_cacheval(alg::HYPREAlgorithm, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    return HYPRECache(nothing, nothing, nothing, nothing, true, true, true)
end

# Overload set_(A|b|u) in order to keep track of "isfresh" for all of them
const LinearCacheHYPRE = LinearCache{<:Any, <:Any, <:Any, <:Any, <:Any, HYPRECache}

function Base.setproperty!(cache::LinearCacheHYPRE, name::Symbol, x)
    if name === :A
        cache.cacheval.isfresh_A = true
        setfield!(cache, :isfresh, true)
        return setfield!(cache, name, x isa HYPREMatrix ? x : HYPREMatrix(x))
    elseif name == :b
        cache.cacheval.isfresh_b = true
        setfield!(cache, :isfresh, true)
        return setfield!(cache, name, x isa HYPREVector ? x : HYPREVector(x))
    elseif name == :u
        cache.cacheval.isfresh_u = true
        setfield!(cache, :isfresh, true)
        return setfield!(cache, name, x isa HYPREVector ? x : HYPREVector(x))
    end
    setfield!(cache, name, x)
end

# Note:
# SciMLBase.init is overloaded here instead of just LinearSolve.init_cacheval for two
# reasons:
# - HYPREArrays can't really be `deepcopy`d, so that is turned off by default
# - The solution vector/initial guess u0 can't be created with
#   fill!(similar(b, size(A, 2)), false) since HYPREArrays are not AbstractArrays.

function SciMLBase.init(prob::LinearProblem, alg::HYPREAlgorithm,
        args...;
        alias = LinearAliasSpecifier(),
        # TODO: Implement eltype for HYPREMatrix in HYPRE.jl? Looks useful
        #       even if it is not AbstractArray.
        abstol = default_tol(prob.A isa HYPREMatrix ? HYPRE_Complex :
                             eltype(prob.A)),
        reltol = default_tol(prob.A isa HYPREMatrix ? HYPRE_Complex :
                             eltype(prob.A)),
        # TODO: Implement length() for HYPREVector in HYPRE.jl?
        maxiters::Int = prob.b isa HYPREVector ? 1000 : length(prob.b),
        verbose::Bool = false,
        Pl = LinearAlgebra.I,
        Pr = LinearAlgebra.I,
        assumptions = OperatorAssumptions(),
        sensealg = LinearSolveAdjoint(),
        kwargs...)
    @unpack A, b, u0, p = prob

    if haskey(kwargs, :alias_A) || haskey(kwargs, :alias_b)
        aliases = LinearAliasSpecifier()

        if haskey(kwargs, :alias_A)
            message = "`alias_A` keyword argument is deprecated, to set `alias_A`,
            please use an ODEAliasSpecifier, e.g. `solve(prob, alias = LinearAliasSpecifier(alias_A = true))"
            Base.depwarn(message, :init)
            Base.depwarn(message, :solve)
            aliases = LinearAliasSpecifier(alias_A = values(kwargs).alias_A)
        end

        if haskey(kwargs, :alias_b)
            message = "`alias_b` keyword argument is deprecated, to set `alias_b`,
            please use an ODEAliasSpecifier, e.g. `solve(prob, alias = LinearAliasSpecifier(alias_b = true))"
            Base.depwarn(message, :init)
            Base.depwarn(message, :solve)
            aliases = LinearAliasSpecifier(
                alias_A = aliases.alias_A, alias_b = values(kwargs).alias_b)
        end
    else
        if alias isa Bool
            aliases = LinearAliasSpecifier(alias = alias)
        else
            aliases = alias
        end
    end

    if isnothing(aliases.alias_A)
        alias_A = false
    else
        alias_A = aliases.alias_A
    end

    if isnothing(aliases.alias_b)
        alias_b = false
    else
        alias_b = aliases.alias_b
    end

    A = A isa HYPREMatrix ? A : HYPREMatrix(A)
    b = b isa HYPREVector ? b : HYPREVector(b)
    u0 = u0 isa HYPREVector ? u0 : (u0 === nothing ? nothing : HYPREVector(u0))

    # Create solution vector/initial guess
    if u0 === nothing
        u0 = zero(b)
    end

    # Initialize internal alg cache
    cacheval = init_cacheval(alg, A, b, u0, Pl, Pr, maxiters, abstol, reltol, verbose,
        assumptions)
    Tc = typeof(cacheval)
    isfresh = true
    precsisfresh = false

    cache = LinearCache{
        typeof(A), typeof(b), typeof(u0), typeof(p), typeof(alg), Tc,
        typeof(Pl), typeof(Pr), typeof(reltol),
        typeof(__issquare(assumptions)), typeof(sensealg)
    }(A, b, u0, p, alg, cacheval, isfresh, precsisfresh, Pl, Pr, abstol, reltol,
        maxiters, verbose, assumptions, sensealg)
    return cache
end

# Solvers whose constructor requires passing the MPI communicator
const COMM_SOLVERS = Union{HYPRE.BiCGSTAB, HYPRE.FlexGMRES, HYPRE.GMRES, HYPRE.ParaSails,
    HYPRE.PCG}
create_solver(::Type{S}, comm) where {S <: COMM_SOLVERS} = S(comm)

# Solvers whose constructor should not be passed the MPI communicator
const NO_COMM_SOLVERS = Union{HYPRE.BoomerAMG, HYPRE.Hybrid, HYPRE.ILU}
create_solver(::Type{S}, comm) where {S <: NO_COMM_SOLVERS} = S()

function create_solver(alg::HYPREAlgorithm, cache::LinearCache)
    # If the solver is already instantiated, return it directly
    if alg.solver isa HYPRE.HYPRESolver
        return alg.solver
    end

    # Otherwise instantiate
    if !(alg.solver <: Union{COMM_SOLVERS, NO_COMM_SOLVERS})
        throw(ArgumentError("unknown or unsupported HYPRE solver: $(alg.solver)"))
    end
    comm = cache.cacheval.A.comm # communicator from the matrix
    solver = create_solver(alg.solver, comm)

    # Construct solver options
    solver_options = (;
        AbsoluteTol = cache.abstol,
        MaxIter = cache.maxiters,
        PrintLevel = Int(cache.verbose),
        Tol = cache.reltol)

    # Preconditioner (uses Pl even though it might not be a *left* preconditioner just *a*
    # preconditioner)
    if !(cache.Pl isa LinearAlgebra.UniformScaling)
        precond = if cache.Pl isa HYPRESolver
            cache.Pl
        elseif cache.Pl isa DataType && cache.Pl <: HYPRESolver
            create_solver(cache.Pl, comm)
        else
            throw(ArgumentError("unknown HYPRE preconditioner $(cache.Pl)"))
        end
        solver_options = merge(solver_options, (; Precond = precond))
    end

    # Filter out some options that are not supported for some solvers
    if solver isa HYPRE.Hybrid
        # Rename MaxIter to PCGMaxIter
        MaxIter = solver_options.MaxIter
        ks = filter(x -> x !== :MaxIter, keys(solver_options))
        solver_options = NamedTuple{ks}(solver_options)
        solver_options = merge(solver_options, (; PCGMaxIter = MaxIter))
    elseif solver isa HYPRE.BoomerAMG || solver isa HYPRE.ILU
        # Remove AbsoluteTol, Precond
        ks = filter(x -> !in(x, (:AbsoluteTol, :Precond)), keys(solver_options))
        solver_options = NamedTuple{ks}(solver_options)
    end

    # Set the options
    HYPRE.Internals.set_options(solver, pairs(solver_options))

    return solver
end

# TODO: How are args... and kwargs... supposed to be used here?
function SciMLBase.solve!(cache::LinearCache, alg::HYPREAlgorithm, args...; kwargs...)
    # It is possible to reach here without HYPRE.Init() being called if HYPRE structures are
    # only to be created here internally (i.e. when cache.A::SparseMatrixCSC and not a
    # ::HYPREMatrix created externally by the user). Be nice to the user and call it :)
    if !(cache.A isa HYPREMatrix || cache.b isa HYPREVector || cache.u isa HYPREVector ||
         alg.solver isa HYPRESolver)
        HYPRE.Init()
    end

    # Move matrix and vectors to HYPRE, if not already provided as HYPREArrays
    hcache = cache.cacheval
    if hcache.isfresh_A || hcache.A === nothing
        hcache.A = cache.A isa HYPREMatrix ? cache.A : HYPREMatrix(cache.A)
        hcache.isfresh_A = false
    end
    if hcache.isfresh_b || hcache.b === nothing
        hcache.b = cache.b isa HYPREVector ? cache.b : HYPREVector(cache.b)
        hcache.isfresh_b = false
    end
    if hcache.isfresh_u || hcache.u === nothing
        hcache.u = cache.u isa HYPREVector ? cache.u : HYPREVector(cache.u)
        hcache.isfresh_u = false
    end

    # Create the solver.
    if hcache.solver === nothing
        hcache.solver = create_solver(alg, cache)
    end

    # Done with cache updates; set it
    cache.cacheval = hcache
    cache.isfresh = false

    # Solve!
    HYPRE.solve!(hcache.solver, hcache.u, hcache.A, hcache.b)

    # Copy back if the output is not HYPREVector
    if cache.u !== hcache.u
        @assert !(cache.u isa HYPREVector)
        copy!(cache.u, hcache.u)
    end

    # Note: Inlining SciMLBase.build_linear_solution(alg, u, resid, cache; retcode, iters)
    # since some of the functions used in there does not play well with HYPREVector.

    T = cache.u isa HYPREVector ? HYPRE_Complex : eltype(cache.u) # eltype(u)
    N = 1 # length((size(u)...,))
    resid = HYPRE.GetFinalRelativeResidualNorm(hcache.solver)
    iters = Int(HYPRE.GetNumIterations(hcache.solver))
    retc = SciMLBase.ReturnCode.Default # TODO: Fetch from solver
    stats = nothing

    ret = SciMLBase.LinearSolution{T, N, typeof(cache.u), typeof(resid), typeof(alg),
        typeof(cache), typeof(stats)}(cache.u, resid, alg, retc,
        iters, cache, stats)

    return ret
end

# HYPREArrays are not AbstractArrays so perform some type-piracy
function SciMLBase.LinearProblem(A::HYPREMatrix, b::HYPREVector,
        p = SciMLBase.NullParameters();
        u0::Union{HYPREVector, Nothing} = nothing, kwargs...)
    return LinearProblem{true}(A, b, p; u0 = u0, kwargs)
end

end # module LinearSolveHYPRE
