Base.@kwdef struct PardisoJL <: SciMLLinearSolveAlgorithm
    nprocs::Union{Int, Nothing} = nothing
    solver_type::Union{Int, Pardiso.Solver, Nothing} = nothing
    matrix_type::Union{Int, Pardiso.MatrixType, Nothing} = nothing
    iparm::Union{Vector{Tuple{Int,Int}}, Nothing} = nothing
    dparm::Union{Vector{Tuple{Int,Int}}, Nothing} = nothing
end

MKLPardisoFactorize(;kwargs...) = PardisoJL(;kwargs...)
MKLPardisoIterate(;kwargs...) = PardisoJL(;kwargs...)
needs_concrete_A(alg::PardisoJL) = true

# TODO schur complement functionality

function init_cacheval(alg::PardisoJL, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    @unpack nprocs, solver_type, matrix_type, iparm, dparm = alg
    A = convert(AbstractMatrix,A)

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

    if matrix_type !== nothing
        Pardiso.set_matrixtype!(solver, matrix_type)
    else
        if eltype(A) <: Real
            Pardiso.set_matrixtype!(solver, Pardiso.REAL_NONSYM)
        elseif eltype(A) <: Complex
            Pardiso.set_matrixtype!(solver, Pardiso.COMPLEX_NONSYM)
        else
            error("Number type not supported by Pardiso")
        end
    end
    verbose && Pardiso.set_msglvl!(solver, Pardiso.MESSAGE_LEVEL_ON)

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

    Pardiso.set_phase!(solver, Pardiso.ANALYSIS)
    Pardiso.pardiso(solver, u, A, b)

    return solver
end

function SciMLBase.solve(cache::LinearCache, alg::PardisoJL; kwargs...)
    @unpack A, b, u = cache
    A = copy(convert(AbstractMatrix,A))

    #if cache.isfresh
    #    Pardiso.set_phase!(cache.cacheval, Pardiso.NUM_FACT)
    #    Pardiso.pardiso(cache.cacheval, u, A, b)
    #end
    #Pardiso.set_phase!(cache.cacheval, Pardiso.SOLVE_ITERATIVE_REFINE)

    Pardiso.set_phase!(cache.cacheval, Pardiso.ANALYSIS_NUM_FACT_SOLVE_REFINE)
    Pardiso.pardiso(cache.cacheval, u, A, b)

    return SciMLBase.build_linear_solution(alg,cache.u,nothing,cache)
end

# Add finalizer to release memory
# Pardiso.set_phase!(cache.cacheval, Pardiso.RELEASE_ALL)

export PardisoJL, MKLPardisoFactorize, MKLPardisoIterate
