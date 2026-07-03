module LinearSolveKrylovKitExt

using LinearSolve, KrylovKit, LinearAlgebra
using LinearSolve: LinearCache, DEFAULT_PRECS
using SciMLLogging: SciMLLogging, @SciMLMessage, verbosity_to_int

function LinearSolve.KrylovKitJL(
        args...;
        KrylovAlg = KrylovKit.GMRES, gmres_restart = 0,
        precs = DEFAULT_PRECS,
        kwargs...
    )
    return KrylovKitJL(KrylovAlg, gmres_restart, precs, args, kwargs)
end

function LinearSolve.KrylovKitJL_CG(args...; kwargs...)
    return KrylovKitJL(args...; KrylovAlg = KrylovKit.CG, kwargs..., isposdef = true)
end

function LinearSolve.KrylovKitJL_GMRES(args...; kwargs...)
    return KrylovKitJL(args...; KrylovAlg = KrylovKit.GMRES, kwargs...)
end

LinearSolve.default_alias_A(::KrylovKitJL, ::Any, ::Any) = true
LinearSolve.default_alias_b(::KrylovKitJL, ::Any, ::Any) = true

function SciMLBase.solve!(cache::LinearCache, alg::KrylovKitJL; kwargs...)
    # KrylovKit doesn't use Pl/Pr, so warn if the user set one
    if !(cache.Pl isa LinearAlgebra.UniformScaling) ||
            !(cache.Pr isa LinearAlgebra.UniformScaling)
        @warn "KrylovKit does not support preconditioners. Pl/Pr will be ignored. Use KrylovJL_GMRES() if you need preconditioning." maxlog = 1
    end
    atol = float(cache.abstol)
    rtol = float(cache.reltol)
    maxiter = cache.maxiters
    verbosity = verbosity_to_int(cache.verbose.KrylovKit_verbosity)
    krylovdim = (alg.gmres_restart == 0) ? min(20, size(cache.A, 1)) : alg.gmres_restart

    kwargs = (
        atol = atol, rtol = rtol, maxiter = maxiter, verbosity = verbosity,
        krylovdim = krylovdim, alg.kwargs...,
    )

    x, info = KrylovKit.linsolve(cache.A, cache.b, cache.u; kwargs...)

    copy!(cache.u, x)
    resid = info.normres
    retcode = if info.converged == 1
        ReturnCode.Default
    else
        @SciMLMessage("Solver failed", cache.verbose, :convergence_failure)
        ReturnCode.ConvergenceFailure
    end

    iters = info.numiter
    return SciMLBase.build_linear_solution(
        alg, cache.u, resid, cache; retcode = retcode,
        iters = iters
    )
end

LinearSolve.update_tolerances_internal!(cache, alg::KrylovKitJL, atol, rtol) = nothing

function SciMLBase.solve(
        prob::LinearSolve.EigenvalueProblem,
        alg::LinearSolve.KrylovKitEigen,
        args...; kwargs...
    )
    nev = LinearSolve.default_nev(prob)
    which = prob.which
    kw = (; prob.kwargs..., alg.kwargs..., kwargs...)
    values, vectors, info = if prob.sigma !== nothing
        _shift_invert_eigsolve(prob, alg, nev, kw)
    elseif prob.B === nothing
        KrylovKit.eigsolve(prob.A, alg.args..., nev, which; kw...)
    else
        KrylovKit.geneigsolve((prob.A, prob.B), alg.args..., nev, which; kw...)
    end
    if prob.sigma !== nothing
        values = prob.sigma .+ inv.(values)
    end
    vecmat = reduce(hcat, vectors)
    values, vecmat = LinearSolve._select_eigenpairs(values, vecmat, nev, prob.which, prob.sigma)
    retcode = info.converged >= length(values) ? ReturnCode.Success : ReturnCode.ConvergenceFailure
    return LinearSolve.build_eigenvalue_solution(
        prob, alg, values, vecmat; retcode, resid = info.normres, stats = info
    )
end

function _shift_invert_eigsolve(prob, alg, nev, kw)
    A, B, sigma = prob.A, prob.B, prob.sigma
    T = isnothing(B) ? promote_type(eltype(A), typeof(sigma)) :
        promote_type(eltype(A), eltype(B), typeof(sigma))
    if isnothing(B)
        F = factorize(A - sigma * I)
        op = x -> F \ x
    elseif B isa LinearAlgebra.UniformScaling
        F = factorize(A - sigma * B)
        op = x -> F \ (B.λ * x)
    else
        F = factorize(A - sigma * B)
        op = x -> F \ (B * x)
    end
    if isempty(alg.args)
        return KrylovKit.eigsolve(op, size(A, 2), nev, :LM, T; kw...)
    else
        return KrylovKit.eigsolve(op, alg.args..., nev, :LM; kw...)
    end
end

end
