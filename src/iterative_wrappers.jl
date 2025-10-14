## Krylov.jl

"""
```julia
KrylovJL(args...; KrylovAlg = Krylov.gmres!,
    Pl = nothing, Pr = nothing,
    gmres_restart = 0, window = 0,
    kwargs...)
```

A generic wrapper over the Krylov.jl krylov-subspace iterative solvers.
"""
struct KrylovJL{F, I, P, A, K} <: AbstractKrylovSubspaceMethod
    KrylovAlg::F
    gmres_restart::I
    window::I
    precs::P
    args::A
    kwargs::K
end

function KrylovJL(args...; KrylovAlg = Krylov.gmres!,
        gmres_restart = 0, window = 0,
        precs = nothing,
        kwargs...)
    return KrylovJL(KrylovAlg, gmres_restart, window,
        precs, args, kwargs)
end

default_alias_A(::KrylovJL, ::Any, ::Any) = true
default_alias_b(::KrylovJL, ::Any, ::Any) = true

"""
```julia
KrylovJL_CG(args...; kwargs...)
```

A generic CG implementation for Hermitian and positive definite linear systems
"""
function KrylovJL_CG(args...; kwargs...)
    KrylovJL(args...; KrylovAlg = Krylov.cg!, kwargs...)
end

"""
```julia
KrylovJL_MINRES(args...; kwargs...)
```

A generic MINRES implementation for Hermitian linear systems
"""
function KrylovJL_MINRES(args...; kwargs...)
    KrylovJL(args...; KrylovAlg = Krylov.minres!, kwargs...)
end

"""
```julia
KrylovJL_GMRES(args...; gmres_restart = 0, window = 0, kwargs...)
```

A generic GMRES implementation for square non-Hermitian linear systems
"""
function KrylovJL_GMRES(args...; kwargs...)
    KrylovJL(args...; KrylovAlg = Krylov.gmres!, kwargs...)
end

"""
```julia
KrylovJL_BICGSTAB(args...; kwargs...)
```

A generic BICGSTAB implementation for square non-Hermitian linear systems
"""
function KrylovJL_BICGSTAB(args...; kwargs...)
    KrylovJL(args...; KrylovAlg = Krylov.bicgstab!, kwargs...)
end

"""
```julia
KrylovJL_LSMR(args...; kwargs...)
```

A generic LSMR implementation for least-squares problems
"""
function KrylovJL_LSMR(args...; kwargs...)
    KrylovJL(args...; KrylovAlg = Krylov.lsmr!, kwargs...)
end

"""
```julia
KrylovJL_CRAIGMR(args...; kwargs...)
```

A generic CRAIGMR implementation for least-norm problems
"""
function KrylovJL_CRAIGMR(args...; kwargs...)
    KrylovJL(args...; KrylovAlg = Krylov.craigmr!, kwargs...)
end

"""
```julia
KrylovJL_MINARES(args...; kwargs...)
```

A generic MINARES implementation for Hermitian linear systems
"""
function KrylovJL_MINARES(args...; kwargs...)
    KrylovJL(args...; KrylovAlg = Krylov.minares!, kwargs...)
end

function get_KrylovJL_solver(KrylovAlg)
    KS = if (KrylovAlg === Krylov.lsmr!)
        Krylov.LsmrWorkspace
    elseif (KrylovAlg === Krylov.cgs!)
        Krylov.CgsWorkspace
    elseif (KrylovAlg === Krylov.usymlq!)
        Krylov.UsymlqWorkspace
    elseif (KrylovAlg === Krylov.lnlq!)
        Krylov.LnlqWorkspace
    elseif (KrylovAlg === Krylov.bicgstab!)
        Krylov.BicgstabWorkspace
    elseif (KrylovAlg === Krylov.crls!)
        Krylov.CrlsWorkspace
    elseif (KrylovAlg === Krylov.lsqr!)
        Krylov.LsqrWorkspace
    elseif (KrylovAlg === Krylov.minres!)
        Krylov.MinresWorkspace
    elseif (KrylovAlg === Krylov.cgne!)
        Krylov.CgneWorkspace
    elseif (KrylovAlg === Krylov.dqgmres!)
        Krylov.DqgmresWorkspace
    elseif (KrylovAlg === Krylov.symmlq!)
        Krylov.SymmlqWorkspace
    elseif (KrylovAlg === Krylov.trimr!)
        Krylov.TrimrWorkspace
    elseif (KrylovAlg === Krylov.usymqr!)
        Krylov.UsymqrWorkspace
    elseif (KrylovAlg === Krylov.bilqr!)
        Krylov.BilqrWorkspace
    elseif (KrylovAlg === Krylov.cr!)
        Krylov.CrWorkspace
    elseif (KrylovAlg === Krylov.craigmr!)
        Krylov.CraigmrWorkspace
    elseif (KrylovAlg === Krylov.tricg!)
        Krylov.TricgWorkspace
    elseif (KrylovAlg === Krylov.craig!)
        Krylov.CraigWorkspace
    elseif (KrylovAlg === Krylov.diom!)
        Krylov.DiomWorkspace
    elseif (KrylovAlg === Krylov.lslq!)
        Krylov.LslqWorkspace
    elseif (KrylovAlg === Krylov.trilqr!)
        Krylov.TrilqrWorkspace
    elseif (KrylovAlg === Krylov.crmr!)
        Krylov.CrmrWorkspace
    elseif (KrylovAlg === Krylov.cg!)
        Krylov.CgWorkspace
    elseif (KrylovAlg === Krylov.cg_lanczos!)
        Krylov.CgLanczosShiftWorkspace
    elseif (KrylovAlg === Krylov.cgls!)
        Krylov.CglsWorkspace
    elseif (KrylovAlg === Krylov.cg_lanczos!)
        Krylov.CgLanczosWorkspace
    elseif (KrylovAlg === Krylov.bilq!)
        Krylov.BilqWorkspace
    elseif (KrylovAlg === Krylov.minres_qlp!)
        Krylov.MinresQlpWorkspace
    elseif (KrylovAlg === Krylov.qmr!)
        Krylov.QmrWorkspace
    elseif (KrylovAlg === Krylov.gmres!)
        Krylov.GmresWorkspace
    elseif (KrylovAlg === Krylov.fgmres!)
        Krylov.FgmresWorkspace
    elseif (KrylovAlg === Krylov.gpmr!)
        Krylov.GpmrWorkspace
    elseif (KrylovAlg === Krylov.fom!)
        Krylov.FomWorkspace
    elseif (KrylovAlg === Krylov.minares!)
        Krylov.MinaresWorkspace
    else
        error("Invalid Krylov method detected")
    end

    return KS
end

# zeroinit allows for init_cacheval to start by initing with A (0,0)
function init_cacheval(alg::KrylovJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::LinearVerbosity, assumptions::OperatorAssumptions; zeroinit = true)
    KS = get_KrylovJL_solver(alg.KrylovAlg)

    if zeroinit
        solver = if (alg.KrylovAlg === Krylov.dqgmres! ||
                     alg.KrylovAlg === Krylov.diom! ||
                     alg.KrylovAlg === Krylov.gmres! ||
                     alg.KrylovAlg === Krylov.fgmres! ||
                     alg.KrylovAlg === Krylov.gpmr! ||
                     alg.KrylovAlg === Krylov.fom!)
            if issparsematrixcsc(A)
                KS(makeempty_SparseMatrixCSC(A), eltype(b)[]; memory = 1)
            elseif A isa Matrix
                KS(Matrix{eltype(A)}(undef, 0, 0), eltype(b)[]; memory = 1)
            else
                KS(A, b; memory = 1)
            end
        else
            if issparsematrixcsc(A)
                KS(makeempty_SparseMatrixCSC(A), eltype(b)[])
            elseif A isa Matrix
                KS(Matrix{eltype(A)}(undef, 0, 0), eltype(b)[])
            else
                KS(A, b)
            end
        end
    else
        memory = (alg.gmres_restart == 0) ? min(20, size(A, 1)) : alg.gmres_restart

        solver = if (alg.KrylovAlg === Krylov.dqgmres! ||
                     alg.KrylovAlg === Krylov.diom! ||
                     alg.KrylovAlg === Krylov.gmres! ||
                     alg.KrylovAlg === Krylov.fgmres! ||
                     alg.KrylovAlg === Krylov.gpmr! ||
                     alg.KrylovAlg === Krylov.fom!)
            KS(A, b; memory)
        elseif (alg.KrylovAlg === Krylov.minres! ||
                alg.KrylovAlg === Krylov.symmlq! ||
                alg.KrylovAlg === Krylov.lslq! ||
                alg.KrylovAlg === Krylov.lsqr! ||
                alg.KrylovAlg === Krylov.lsmr!)
            (alg.window != 0) ? KS(A, b; window = alg.window) : KS(A, b)
        else
            KS(A, b)
        end
    end

    solver.x = u

    return solver
end

# Krylov.jl tries to init with `ArrayPartition(undef, ...)`. Avoid hitting that!
function init_cacheval(
        alg::LinearSolve.KrylovJL, A, b::RecursiveArrayTools.ArrayPartition, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity, ::LinearSolve.OperatorAssumptions)
    return nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::KrylovJL; kwargs...)
    if cache.precsisfresh && !isnothing(alg.precs)
        Pl, Pr = alg.precs(cache.A, cache.p)
        cache.Pl = Pl
        cache.Pr = Pr
        cache.precsisfresh = false
    end
    if cache.isfresh
        solver = init_cacheval(alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol, cache.verbose,
            cache.assumptions, zeroinit = false)
        cache.cacheval = solver
        cache.isfresh = false
    end

    M, N = cache.Pl, cache.Pr

    # use no-op preconditioner for Krylov.jl (LinearAlgebra.I) when M/N is identity
    M = _isidentity_struct(M) ? I : M
    N = _isidentity_struct(N) ? I : N

    atol = float(cache.abstol)
    rtol = float(cache.reltol)
    itmax = cache.maxiters
    verbose = cache.verbose ? 1 : 0

    cacheval = if cache.alg isa DefaultLinearSolver
        if alg.KrylovAlg === Krylov.gmres!
            @get_cacheval(cache, :KrylovJL_GMRES)
        elseif alg.KrylovAlg === Krylov.craigmr!
            @get_cacheval(cache, :KrylovJL_CRAIGMR)
        elseif alg.KrylovAlg === Krylov.lsmr!
            @get_cacheval(cache, :KrylovJL_LSMR)
        else
            error("Default linear solver can only be these three choices! Report this bug!")
        end
    else
        cache.cacheval
    end

    args = (cacheval, cache.A, cache.b)
    kwargs = (atol = atol, rtol, itmax, verbose,
        ldiv = true, history = true, alg.kwargs...)

    if cache.cacheval isa Krylov.CgWorkspace
        N !== I &&
            @warn "$(alg.KrylovAlg) doesn't support right preconditioning."
        Krylov.krylov_solve!(args...; M, kwargs...)
    elseif cache.cacheval isa Krylov.GmresWorkspace
        Krylov.krylov_solve!(args...; M, N, restart = alg.gmres_restart > 0, kwargs...)
    elseif cache.cacheval isa Krylov.BicgstabWorkspace
        Krylov.krylov_solve!(args...; M, N, kwargs...)
    elseif cache.cacheval isa Krylov.MinresWorkspace
        N !== I &&
            @warn "$(alg.KrylovAlg) doesn't support right preconditioning."
        Krylov.krylov_solve!(args...; M, kwargs...)
    else
        Krylov.krylov_solve!(args...; kwargs...)
    end

    stats = @get_cacheval(cache, :KrylovJL_GMRES).stats
    resid = !isempty(stats.residuals) ? last(stats.residuals) :
            zero(eltype(stats.residuals))

    retcode = if !stats.solved
        if stats.status == "maximum number of iterations exceeded"
            ReturnCode.MaxIters
        elseif stats.status == "solution good enough given atol and rtol"
            ReturnCode.ConvergenceFailure
        else
            ReturnCode.Failure
        end
    else
        ReturnCode.Success
    end

    # Copy the solution to the allocated output vector
    cacheval = @get_cacheval(cache, :KrylovJL_GMRES)
    if cache.u !== cacheval.x && ArrayInterface.can_setindex(cache.u)
        cache.u .= cacheval.x
    else
        cache.u = convert(typeof(cache.u), cacheval.x)
    end

    return SciMLBase.build_linear_solution(alg, cache.u, Ref(resid), cache;
        iters = stats.niter, retcode, stats)
end
