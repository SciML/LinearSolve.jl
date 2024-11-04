module LinearSolvePardisoExt

using Pardiso, LinearSolve, SciMLBase
using SparseArrays
using SparseArrays: nonzeros, rowvals, getcolptr
using LinearSolve: PardisoJL

using UnPack

LinearSolve.needs_concrete_A(alg::PardisoJL) = true

# TODO schur complement functionality

function LinearSolve.init_cacheval(alg::PardisoJL,
        A,
        b,
        u,
        Pl,
        Pr,
        maxiters::Int,
        abstol,
        reltol,
        verbose::Bool,
        assumptions::LinearSolve.OperatorAssumptions)
    @unpack nprocs, solver_type, matrix_type, cache_analysis, iparm, dparm, vendor = alg
    A = convert(AbstractMatrix, A)

    if isnothing(vendor)
        if Pardiso.panua_is_available()
            vendor = :Panua
        else
            vendor = :MKL
        end
    end

    transposed_iparm = 1
    solver = if vendor == :MKL
        solver = if Pardiso.mkl_is_available()
            solver = Pardiso.MKLPardisoSolver()
            Pardiso.pardisoinit(solver)
            nprocs !== nothing && Pardiso.set_nprocs!(solver, nprocs)

            # for mkl 1 means conjugated and 2 means transposed.
            # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/pardiso-iparm-parameter.html#IPARM37
            transposed_iparm = 2

            solver
        else
            error("MKL Pardiso is not available. On MacOSX, possibly, try Panua Pardiso.")
        end
    elseif vendor == :Panua
        solver = if Pardiso.panua_is_available()
            solver = Pardiso.PardisoSolver()
            Pardiso.pardisoinit(solver)
            solver_type !== nothing && Pardiso.set_solver!(solver, solver_type)

            solver
        else
            error("Panua Pardiso is not available.")
        end
    else
        error("Pardiso vendor must be either `:MKL` or `:Panua`")
    end

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

    #=
    Note: It is recommended to use IPARM(11)=1 (scaling) and IPARM(13)=1 (matchings) for
    highly indefinite symmetric matrices e.g. from interior point optimizations or saddle point problems.
    It is also very important to note that the user must provide in the analysis phase (PHASE=11)
    the numerical values of the matrix A if IPARM(11)=1 (scaling) or PARM(13)=1 or 2 (matchings).

    The numerical values will be incorrect since the analysis is ran once and
    cached. If these two are not set, then Pardiso.NUM_FACT in the solve! must
    be changed to Pardiso.ANALYSIS_NUM_FACT in the solver loop otherwise instabilities
    occur in the example https://github.com/SciML/OrdinaryDiffEq.jl/issues/1569
    =#
    if cache_analysis
        Pardiso.set_iparm!(solver, 11, 0)
        Pardiso.set_iparm!(solver, 13, 0)
    end

    if alg.solver_type == 1
        # PARDISO uses a numerical factorization A = LU for the first system and
        # applies these exact factors L and U for the next steps in a
        # preconditioned Krylov-Subspace iteration. If the iteration does not
        # converge, the solver will automatically switch back to the numerical factorization.
        # Be aware that in the intel docs, iparm indexes are one lower.
        Pardiso.set_iparm!(solver, 4, round(Int, abs(log10(reltol)), RoundDown) * 10 + 1)
    end

    # pass in vector of tuples like [(iparm::Int, key::Int) ...]
    if iparm !== nothing
        for i in iparm
            Pardiso.set_iparm!(solver, i...)
        end
    end

    if dparm !== nothing
        for d in dparm
            Pardiso.set_dparm!(solver, d...)
        end
    end

    # Make sure to say it's transposed because its CSC not CSR
    # This is also the only value which should not be overwritten by users
    Pardiso.set_iparm!(solver, 12, transposed_iparm)

    if cache_analysis
        Pardiso.set_phase!(solver, Pardiso.ANALYSIS)
        Pardiso.pardiso(solver,
            u,
            SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
            b)
    end

    return solver
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::PardisoJL; kwargs...)
    @unpack A, b, u = cache
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        phase = alg.cache_analysis ? Pardiso.NUM_FACT : Pardiso.ANALYSIS_NUM_FACT
        Pardiso.set_phase!(cache.cacheval, phase)
        Pardiso.pardiso(cache.cacheval,
            SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)),
            eltype(A)[])
        cache.isfresh = false
    end
    Pardiso.set_phase!(cache.cacheval, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(cache.cacheval, u,
        SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A)), b)
    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

# Add finalizer to release memory
# Pardiso.set_phase!(cache.cacheval, Pardiso.RELEASE_ALL)

end
