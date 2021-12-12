
## Pardiso

import Pardiso

export PardisoJL, MKLPardisoFactorize, MKLPardisoIterate

Base.@kwdef struct PardisoJL <: SciMLLinearSolveAlgorithm
    nprocs::Union{Int, Nothing} = nothing
    solver_type::Union{Int, Pardiso.Solver, Nothing} = nothing
    matrix_type::Union{Int, Pardiso.MatrixType, Nothing} = nothing
    fact_phase::Union{Int, Pardiso.Phase, Nothing} = nothing
    solve_phase::Union{Int, Pardiso.Phase, Nothing} = nothing
    release_phase::Union{Int, Nothing} = nothing
    iparm::Union{Vector{Tuple{Int,Int}}, Nothing} = nothing
    dparm::Union{Vector{Tuple{Int,Int}}, Nothing} = nothing
end

MKLPardisoFactorize(;kwargs...) = PardisoJL(;fact_phase=Pardiso.NUM_FACT,
                                             solve_phase=Pardiso.SOLVE_ITERATIVE_REFINE,
                                             kwargs...)
MKLPardisoIterate(;kwargs...) = PardisoJL(;solve_phase=Pardiso.NUM_FACT_SOLVE_REFINE,
                                           kwargs...)

# TODO schur complement functionality

function init_cacheval(alg::PardisoJL, cache::LinearCache)
    @unpack nprocs, solver_type, matrix_type, fact_phase, solve_phase, iparm, dparm = alg
    @unpack A, b, u = cache

    if A isa DiffEqArrayOperator
        A = A.A
    end

    solver =
    if Pardiso.PARDISO_LOADED[]
        solver = Pardiso.PardisoSolver()
        solver_type !== nothing && Pardiso.set_solver!(solver, solver_type)

        solver
    else
        solver = Pardiso.MKLPardisoSolver()
        nprocs !== nothing && Pardiso.set_nprocs!(solver, nprocs)

        solver
    end

    Pardiso.pardisoinit(solver) # default initialization

    matrix_type !== nothing && Pardiso.set_matrixtype!(solver, matrix_type)
    cache.verbose && Pardiso.set_msglvl!(solver, Pardiso.MESSAGE_LEVEL_ON)

    # pass in vector of tuples like [(iparm::Int, key::Int) ...]
    if iparm !== nothing
        for i in length(iparm)
            Pardiso.set_iparm!(solver, iparm[i]...)
        end
    end

    if dparm !== nothing
        for i in length(dparm)
            Pardiso.set_dparm!(solver, dparm[i]...)
        end
    end

    if (fact_phase !== nothing) | (solve_phase !== nothing)
        Pardiso.set_phase!(solver, Pardiso.ANALYSIS)
        Pardiso.pardiso(solver, u, A, b)
    end

    if fact_phase !== nothing
        Pardiso.set_phase!(solver, fact_phase)
        Pardiso.pardiso(solver, u, A, b)
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

    alg.solve_phase !== nothing && Pardiso.set_phase!(cache.cacheval, alg.solve_phase)
    Pardiso.pardiso(cache.cacheval, u, A, b)
    alg.release_phase !== nothing && Pardiso.set_phase!(cache.cacheval, alg.release_phase)

    return SciMLBase.build_linear_solution(alg,cache.u,nothing)
end
