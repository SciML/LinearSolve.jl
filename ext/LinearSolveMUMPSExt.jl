module LinearSolveMUMPSExt

using LinearAlgebra
using LinearSolve
using LinearSolve: LinearVerbosity, OperatorAssumptions
using SparseArrays
import MUMPS
import SciMLBase
import SciMLBase: ReturnCode, LinearSolution
using SciMLLogging: @SciMLMessage

mutable struct MUMPSCache
    solver::Any

    function MUMPSCache()
        cache = new(nothing)
        finalizer(_finalize_mumps_cache!, cache)
        return cache
    end
end

function _finalize_mumps_cache!(cache::MUMPSCache)
    solver = cache.solver
    solver === nothing && return
    try
        Base.finalize(solver)
    catch
    end
    cache.solver = nothing
    return
end

cleanup_mumps_cache!(cache::MUMPSCache) = _finalize_mumps_cache!(cache)
cleanup_mumps_cache!(cache::LinearSolve.LinearCache) = cleanup_mumps_cache!(cache.cacheval)
"""
    cleanup_mumps_cache!(cache::MUMPSCache)
    cleanup_mumps_cache!(cache::LinearCache)

Destroy the live `MUMPS.Mumps` object owned by a LinearSolve cache and reset the
cache to an empty state. Safe to call multiple times.

Prefer calling this explicitly when using `MUMPSFactorization` together with MPI,
before `MPI.Finalize()`, because MUMPS holds MPI-backed resources outside Julia's GC.
The cache also registers a finalizer as a safety net, but explicit cleanup is
strongly preferred for deterministic teardown.
"""
cleanup_mumps_cache!

LinearSolve.needs_concrete_A(::LinearSolve.MUMPSFactorization) = true

function LinearSolve.init_cacheval(
        alg::LinearSolve.MUMPSFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return MUMPSCache()
end

_mumps_eltype(::Type{Float16}) = Float32
_mumps_eltype(::Type{Float32}) = Float32
_mumps_eltype(::Type{Float64}) = Float64
_mumps_eltype(::Type{ComplexF16}) = ComplexF32
_mumps_eltype(::Type{ComplexF32}) = ComplexF32
_mumps_eltype(::Type{ComplexF64}) = ComplexF64
function _mumps_eltype(::Type{T}) where {T}
    throw(
        ArgumentError(
            "MUMPSFactorization only supports Float32, Float64, ComplexF32, and ComplexF64 inputs; got element type $T"
        )
    )
end

function _mumps_cntl(::Type{T}, cntl) where {T}
    if cntl === nothing
        return T <: Union{Float32, ComplexF32} ? copy(MUMPS.default_cntl32) :
            copy(MUMPS.default_cntl64)
    end
    return real(T).(copy(cntl))
end

function _mumps_solver(
        alg::LinearSolve.MUMPSFactorization,
        A::SparseMatrixCSC,
        b,
    )
    T = _mumps_eltype(promote_type(eltype(A), eltype(b)))
    icntl = alg.icntl === nothing ? MUMPS.get_icntl(;
            verbose = alg.verbose,
            ooc = alg.ooc,
            itref = alg.itref,
            user_perm = alg.user_perm
        ) : Int32.(copy(alg.icntl))
    cntl = _mumps_cntl(T, alg.cntl)
    return MUMPS.Mumps{T}(alg.sym, icntl, cntl; par = alg.par)
end

function _mumps_retcode(solver)
    return solver.err < 0 ? ReturnCode.Failure : ReturnCode.Success
end

function _solve_failed_solution(
        alg::LinearSolve.MUMPSFactorization,
        cache::LinearSolve.LinearCache,
        msg::AbstractString
    )
    @SciMLMessage(msg, cache.verbose, :solver_failure)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
    )
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::LinearSolve.MUMPSFactorization;
        kwargs...
    )
    A = convert(AbstractMatrix, cache.A)
    A_sparse = A isa SparseMatrixCSC ? A : sparse(A)
    mumps_cache = LinearSolve.@get_cacheval(cache, :MUMPSFactorization)

    if cache.isfresh || mumps_cache.solver === nothing
        _finalize_mumps_cache!(mumps_cache)
        solver = try
            _mumps_solver(alg, A_sparse, cache.b)
        catch err
            return _solve_failed_solution(alg, cache, sprint(showerror, err))
        end

        try
            MUMPS.associate_matrix!(solver, A_sparse)
            MUMPS.factorize!(solver)
        catch err
            Base.finalize(solver)
            return _solve_failed_solution(alg, cache, sprint(showerror, err))
        end

        if _mumps_retcode(solver) != ReturnCode.Success
            Base.finalize(solver)
            return _solve_failed_solution(
                alg,
                cache,
                "MUMPS factorization failed with error code $(solver.err)"
            )
        end

        mumps_cache.solver = solver
        cache.isfresh = false
    end

    solver = mumps_cache.solver
    try
        MUMPS.associate_rhs!(solver, cache.b)
        alg.transposed && transpose!(solver)
        MUMPS.mumps_solve!(solver; rhs_changed = true)
        alg.transposed && transpose!(solver)
        MUMPS.get_sol!(cache.u, solver)
    catch err
        alg.transposed && try
            transpose!(solver)
        catch
        end
        return _solve_failed_solution(alg, cache, sprint(showerror, err))
    end

    if _mumps_retcode(solver) != ReturnCode.Success
        return _solve_failed_solution(
            alg,
            cache,
            "MUMPS solve failed with error code $(solver.err)"
        )
    end

    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Success
    )
end

LinearSolve._custom_can_reuse_adjoint_factorization(
    ::LinearSolve.MUMPSFactorization, cache::MUMPSCache
) = true

function LinearSolve._custom_adjoint_factorization_solve(
        alg::LinearSolve.MUMPSFactorization, cache::MUMPSCache, A, b
    )
    solver = cache.solver
    solver === nothing && return nothing
    solution = similar(b)
    reverse_transposed = !alg.transposed
    # MUMPS exposes transpose but not adjoint solves. Conjugating both sides
    # turns a transpose solve with the cached factors into an adjoint solve.
    rhs = eltype(A) <: Real ? b : conj.(b)
    MUMPS.associate_rhs!(solver, rhs)
    reverse_transposed && transpose!(solver)
    try
        MUMPS.mumps_solve!(solver; rhs_changed = true)
        MUMPS.get_sol!(solution, solver)
    finally
        reverse_transposed && transpose!(solver)
    end
    return eltype(A) <: Real ? solution : conj.(solution)
end

end
