module LinearSolvePardisoExt

using Pardiso, LinearSolve
using SparseArrays
using SparseArrays: nonzeros, rowvals, getcolptr
using LinearAlgebra: issymmetric, ishermitian
using LinearSolve: PardisoJL, LinearVerbosity
using SciMLLogging: SciMLLogging, @SciMLMessage, verbosity_to_bool
using LinearSolve.SciMLBase

# TODO schur complement functionality

function pardiso_csc(A)
    return SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A))
end

# True when the sparsity pattern of `A` is symmetric (stored values ignored).
# Implemented with public SparseArrays / LinearAlgebra APIs only.
function is_structurally_symmetric(A::SparseMatrixCSC)
    pattern = SparseMatrixCSC(
        size(A)..., getcolptr(A), rowvals(A), ones(Bool, length(nonzeros(A)))
    )
    return issymmetric(pattern)
end

"""
    default_pardiso_matrix_type(A)

Choose a Pardiso matrix type from the numerical structure of `A`. Symmetric /
Hermitian matrices use the indefinite types; structurally symmetric (but not
numerically symmetric / Hermitian) matrices use `REAL_SYM` /
`COMPLEX_STRUCT_SYM`; otherwise nonsymmetric types. Symmetric types require the
triangular storage from `Pardiso.get_matrix`.
"""
function default_pardiso_matrix_type(A)
    A = pardiso_csc(A)
    Tv = eltype(A)
    return if Tv <: Real
        if issymmetric(A)
            Pardiso.REAL_SYM_INDEF
        elseif is_structurally_symmetric(A)
            Pardiso.REAL_SYM
        else
            Pardiso.REAL_NONSYM
        end
    elseif Tv <: Complex
        if ishermitian(A)
            Pardiso.COMPLEX_HERM_INDEF
        elseif issymmetric(A)
            Pardiso.COMPLEX_SYM
        elseif is_structurally_symmetric(A)
            Pardiso.COMPLEX_STRUCT_SYM
        else
            Pardiso.COMPLEX_NONSYM
        end
    else
        error("Number type not supported by Pardiso")
    end
end

# Pardiso expects CSR; we pass CSC and set the transpose iparm. Symmetric / Hermitian
# matrix types additionally need the triangular compression from `get_matrix`.
# Duck-typed: `AbstractPardisoSolver` is not public (`Base.ispublic` false).
function pardiso_matrix(ps, A)
    return Pardiso.get_matrix(ps, pardiso_csc(A), :N)
end

function release_pardiso!(ps, A, b, u)
    Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(ps, u, pardiso_csc(A), b)
    return nothing
end

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
        verbose::Union{LinearVerbosity, Bool},
        assumptions::LinearSolve.OperatorAssumptions
    )
    (; nprocs, solver_type, matrix_type, cache_analysis, iparm, dparm, vendor) = alg
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
        if Pardiso.mkl_is_available()
            solver = Pardiso.MKLPardisoSolver()
            # for mkl 1 means conjugated and 2 means transposed.
            # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/pardiso-iparm-parameter.html#IPARM37
            transposed_iparm = 2
            solver
        else
            error("MKL Pardiso is not available. On MacOSX, possibly, try Panua Pardiso.")
        end
    elseif vendor == :Panua
        if Pardiso.panua_is_available()
            solver = Pardiso.PardisoSolver()
            solver_type !== nothing && Pardiso.set_solver!(solver, solver_type)
            solver
        else
            error("Panua Pardiso is not available.")
        end
    else
        error("Pardiso vendor must be either `:MKL` or `:Panua`")
    end

    # Matrix type must be set before pardisoinit so default iparms match the type
    # (Pardiso.jl examples / solve! all do set_matrixtype! then pardisoinit).
    if matrix_type !== nothing
        Pardiso.set_matrixtype!(solver, matrix_type)
    else
        Pardiso.set_matrixtype!(solver, default_pardiso_matrix_type(A))
    end
    Pardiso.pardisoinit(solver)
    if vendor == :MKL
        nprocs !== nothing && Pardiso.set_nprocs!(solver, nprocs)
    end

    if verbose isa Bool
        verbose_spec = LinearVerbosity(pardiso_verbosity = SciMLLogging.Silent())
    else
        verbose_spec = verbose
    end

    if verbosity_to_bool(verbose_spec.pardiso_verbosity)
        Pardiso.set_msglvl!(solver, Pardiso.MESSAGE_LEVEL_ON)
    end
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
        Pardiso.pardiso(
            solver,
            u,
            pardiso_matrix(solver, A),
            b
        )
    end

    return solver
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::PardisoJL; kwargs...)
    (; A, b, u) = cache
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        # Automatic matrix type can change when `A` is updated (e.g. symmetric →
        # nonsymmetric with the same sparsity). Explicit user overrides are kept.
        if alg.matrix_type === nothing
            new_type = default_pardiso_matrix_type(A)
            if Pardiso.get_matrixtype(cache.cacheval) != new_type
                release_pardiso!(cache.cacheval, A, b, u)
                cache.cacheval = LinearSolve.init_cacheval(
                    alg, A, b, u, cache.Pl, cache.Pr, cache.maxiters,
                    cache.abstol, cache.reltol, cache.verbose, cache.assumptions
                )
            end
        end
        A_pardiso = pardiso_matrix(cache.cacheval, A)
        phase = alg.cache_analysis ? Pardiso.NUM_FACT : Pardiso.ANALYSIS_NUM_FACT
        Pardiso.set_phase!(cache.cacheval, phase)
        Pardiso.pardiso(
            cache.cacheval,
            A_pardiso,
            eltype(A)[]
        )
        cache.isfresh = false
    else
        A_pardiso = pardiso_matrix(cache.cacheval, A)
    end
    Pardiso.set_phase!(cache.cacheval, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(
        cache.cacheval, u,
        A_pardiso, b
    )
    return SciMLBase.build_linear_solution(alg, cache.u, nothing, nothing)
end

LinearSolve._custom_can_reuse_adjoint_factorization(
    ::PardisoJL, ::Pardiso.AbstractPardisoSolver
) = true

function LinearSolve._custom_adjoint_factorization_solve(
        ::PardisoJL, solver::Pardiso.AbstractPardisoSolver, A, b
    )
    transposed_iparm = Pardiso.get_iparm(solver, 12)
    solution = similar(b)
    # Pardiso sees the CSC storage as CSR for transpose(A). With transpose mode
    # disabled, conjugating both sides solves the adjoint system for complex A.
    rhs = eltype(A) <: Real ? b : conj.(b)
    Pardiso.set_iparm!(solver, 12, 0)
    Pardiso.set_phase!(solver, Pardiso.SOLVE_ITERATIVE_REFINE)
    try
        Pardiso.pardiso(
            solver, solution,
            pardiso_matrix(solver, A), rhs
        )
    finally
        Pardiso.set_iparm!(solver, 12, transposed_iparm)
    end
    return eltype(A) <: Real ? solution : conj.(solution)
end

# Add finalizer to release memory
# Pardiso.set_phase!(cache.cacheval, Pardiso.RELEASE_ALL)

end
