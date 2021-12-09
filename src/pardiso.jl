
## Paradiso

import Pardiso

export PardisoJL

Base.@kwdef struct PardisoJL <: SciMLLinearSolveAlgorithm
    nprocs::Union{Int, Nothing} = nothing
    solver_type::Union{Int, Pardiso.Solver, Nothing} = nothing
    matrix_type::Union{Int, Pardiso.MatrixType, Nothing} = nothing
    solve_phase::Union{Int, Pardiso.Phase, Nothing} = nothing
    release_phase::Union{Int, Nothing} = nothing
    iparm::Union{Vector{Tuple{Int,Int}}, Nothing} = nothing
    dparm::Union{Vector{Tuple{Int,Int}}, Nothing} = nothing
end

function init_cacheval(alg::PardisoJL, cache::LinearCache)
    @unpack nprocs, solver_type, matrix_type, iparm, dparm = alg

    solver = Pardiso.PARDISO_LOADED[] ? Pardiso.PardisoSolver() : Pardiso.MKLPardisoSolver()

    Pardiso.pardisoinit(solver) # default initialization

    nprocs      !== nothing && Pardiso.set_nprocs!(ps, nprocs)
    solver_type !== nothing && Pardiso.set_solver!(solver, key)
    matrix_type !== nothing && Pardiso.set_matrixtype!(solver, matrix_type)
    cache.verbose && Pardiso.set_msglvl!(solver, Pardiso.MESSAGE_LEVEL_ON)

    if iparm !== nothing # pass in vector of tuples like [(iparm, key)]
        for i in length(iparm)
            Pardiso.set_iparm!(solver, iparm[i]...)
        end
    end

    if dparm !== nothing
        for i in length(dparm)
            Pardiso.set_dparm!(solver, dparm[i]...)
        end
    end

    return solver
end

function SciMLBase.solve(cache::LinearCache, alg::PardisoJL; kwargs...)
    @unpack A, b, u = cache
    if A isa DiffEqArrayOperator
        A = A.A
    end

    if cache.isfresh
        solver = init_cacheval(alg, cache)
        cache = set_cacheval(cache, solver)
    end

    abstol = cache.abstol
    reltol = cache.reltol
    kwargs = (abstol=abstol, reltol=reltol)

    """
    figure out whatever phase is. should set_phase call be in init_cacheval?
    can we use phase to store factorization in cache?
    """
    alg.solve_phase !== nothing && Pardiso.set_phase!(cacheval, alg.solve_phase)
    Pardiso.solve!(cache.cacheval, u, A, b)
    alg.release_phase !== nothing && Pardiso.set_phase!(cacheval, alg.release_phase) # is this necessary?

    return cache.u
end
