
## Paradiso

import Pardiso

export PardisoJL

struct PardisoJL{A} <: SciMLLinearSolveAlgorithm
    nthreads::Union{Int, Nothing}
    solver_type::Union{Int, Pardiso.Solver, Nothing}
    matrix_type::Union{Int, Pardiso.MatrixType, Nothing}
    solve_phase::Union{Int, Pardiso.Phase, Nothing}
    release_phase::Union{Int, Nothing}
    iparm::Union{A, Nothing}
    dparm::Union{A, Nothing}
end

function PardisoJL(solver_type=Pardiso.nothing,)

    return PardisoJL(nthreads, solver_type, matrix_type, solve_phase,
                     release_phase, iparm, dparm)
end

function init_cacheval(alg::PardisoJL, cache::LinearCache)
    @unpack nthreads, solver_type, matrix_type, iparm, dparm = alg

    solver = Pardiso.PARDISO_LOADED[] ? PardisoSolver() : MKLPardisoSolver()

    Pardiso.pardisoinit(solver) # default initialization

    nthreads    !== nothing && Pardiso.set_nprocs!(ps, nthreads)
    solver_type !== nothing && Pardiso.set_solver!(solver, key)
    matrix_type !== nothing && Pardiso.set_matrixtype!(solver, matrix_type)
    cache.verbose && Pardiso.set_msglvl!(solver, Pardiso.MESSAGE_LEVEL_ON)

    iparm !== nothing && begin # pass in vector of tuples like [(iparm, key)]
        for i in length(iparm)
            Pardiso.set_iparm!(solver, iparm[i]...)
        end
    end

    dparm !== nothing && begin
        for i in length(dparm)
            Pardiso.set_dparm!(solver, dparm[i]...)
        end
    end

    return solver
end

function SciMLBase.solve(cache::LinearCache, alg::PardisoJL; kwargs...)
    @unpack A, b, u, cacheval = cache

    if cache.isfresh
        solver = init_cacheval(alg, cache)
        cache = set_cacheval(cache, solver)
    end

    abstol = cache.abstol
    reltol = cache.reltol
    kwargs = (abstol=abstol, reltol=reltol, alg.kwargs...)

    """
    figure out whatever phase is. should set_phase call be in init_cacheval?
    can we use phase to store factorization in cache?
    """
    Pardiso.set_phase!(cacheval, alg.solve_phase)
    Pardiso.solve!(cacheval, u, A, b)
    Pardiso.set_phase!(cacheval, alg.release_phase) # is this necessary?

    return cache.u
end
