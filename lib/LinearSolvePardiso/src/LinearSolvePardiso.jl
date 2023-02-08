module LinearSolvePardiso

using Pardiso, LinearSolve, SciMLBase
using SparseArrays
using SparseArrays: nonzeros, rowvals, getcolptr

using UnPack

Base.@kwdef struct PardisoJL <: LinearSolve.SciMLLinearSolveAlgorithm
    nprocs::Union{Int,Nothing} = nothing
    solver_type::Union{Int,Pardiso.Solver,Nothing} = nothing
    matrix_type::Union{Int,Pardiso.MatrixType,Nothing} = nothing
    iparm::Union{Vector{Tuple{Int,Int}},Nothing} = nothing
    dparm::Union{Vector{Tuple{Int,Int}},Nothing} = nothing
end

MKLPardisoFactorize(; kwargs...) = PardisoJL(; solver_type = 0, kwargs...)
MKLPardisoIterate(; kwargs...) = PardisoJL(; solver_type = 1, kwargs...)
LinearSolve.needs_concrete_A(alg::PardisoJL) = true

# TODO schur complement functionality

function LinearSolve.init_cacheval(
    alg::PardisoJL,
    A,
    b,
    u,
    Pl,
    Pr,
    maxiters::Int,
    abstol,
    reltol,
    verbose::Bool,
    assumptions::LinearSolve.OperatorAssumptions,
)
    @unpack nprocs, solver_type, matrix_type, iparm, dparm = alg
    A = convert(AbstractMatrix, A)

    solver = if Pardiso.PARDISO_LOADED[]
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

    # Make sure to say it's transposed because its CSC not CSR
    Pardiso.set_iparm!(solver, 12, 1)

    #=
    Note: It is recommended to use IPARM(11)=1 (scaling) and IPARM(13)=1 (matchings) for
    highly indefinite symmetric matrices e.g. from interior point optimizations or saddle point problems.
    It is also very important to note that the user must provide in the analysis phase (PHASE=11)
    the numerical values of the matrix A if IPARM(11)=1 (scaling) or PARM(13)=1 or 2 (matchings).

    The numerical values will be incorrect since the analysis is ran once and
    cached. If these two are not set, then Pardiso.NUM_FACT in the solve must
    be changed to Pardiso.ANALYSIS_NUM_FACT in the solver loop otherwise instabilities
    occur in the example https://github.com/SciML/OrdinaryDiffEq.jl/issues/1569
    =#
    Pardiso.set_iparm!(solver, 11, 0)
    Pardiso.set_iparm!(solver, 13, 0)

    Pardiso.set_phase!(solver, Pardiso.ANALYSIS)

    if alg.solver_type == 1
        # PARDISO uses a numerical factorization A = LU for the first system and
        # applies these exact factors L and U for the next steps in a
        # preconditioned Krylov-Subspace iteration. If the iteration does not
        # converge, the solver will automatically switch back to the numerical factorization.
        Pardiso.set_iparm!(solver, 3, round(Int, abs(log10(reltol)), RoundDown) * 10 + 1)
    end

    Pardiso.pardiso(
        solver,
        u,
        SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
        b,
    )

    return solver
end

function SciMLBase.solve(cache::LinearSolve.LinearCache, alg::PardisoJL; kwargs...)
    @unpack A, b, u = cache
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        Pardiso.set_phase!(cache.cacheval, Pardiso.NUM_FACT)
        Pardiso.pardiso(cache.cacheval, A, eltype(A)[])
    end

    Pardiso.set_phase!(cache.cacheval, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(cache.cacheval, u, A, b)
    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

# Add finalizer to release memory
# Pardiso.set_phase!(cache.cacheval, Pardiso.RELEASE_ALL)

export PardisoJL, MKLPardisoFactorize, MKLPardisoIterate

end
