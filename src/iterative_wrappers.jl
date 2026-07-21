## Krylov.jl

"""
```julia
KrylovJL(args...; KrylovAlg = Krylov.gmres!,
    Pl = nothing, Pr = nothing,
    gmres_restart = 0, window = 0,
    warm_start = :none,
    kwargs...)
```

A generic wrapper over the Krylov.jl krylov-subspace iterative solvers.

`warm_start` controls the initial guess used when the same cache is solved
repeatedly (currently supported for GMRES and FGMRES; other methods ignore it):

  - `:none` (default): every solve starts from zero.
  - `:previous`: start from the previous solution `cache.u`.
  - `:hegedus`: start from the previous solution rescaled by the Hegedüs trick,
    `x₀ = ξ u` with `ξ = ⟨Au, b⟩ / ‖Au‖²`, which minimizes the initial residual
    along the direction of the previous solution and hence guarantees
    `‖b - A x₀‖ ≤ ‖b‖`.

Warm starting costs one extra operator application per solve (two for
`:hegedus`, plus one preconditioner application when a left preconditioner is
set). Benchmarks on stiff PDE Newton-Krylov solves (Brusselator, Allen-Cahn,
Burgers, advection-diffusion with KenCarp47/TRBDF2/FBDF + ILU) show `:hegedus`
reliably reduces GMRES iteration counts (median ≈ -17%), but wall time only
improves when each solve performs substantial Krylov work (≳5 iterations per
solve); with a preconditioner strong enough that solves take ≲3 iterations the
fixed per-solve overhead dominates the savings. `:previous` typically
*increases* Newton-Krylov iteration counts — the previous Newton increment
overshoots as the iteration converges — and is intended for sequences whose
solutions vary slowly, not for Newton loops.

!!! warning

    Do not enable `warm_start` inside Rosenbrock-type (W-method) integrators
    such as `Rodas5P`. They have no outer Newton iteration to absorb
    within-tolerance differences in stage solves, and warm starting there can
    degrade accuracy and trigger step-rejection feedback loops (observed:
    `:previous` producing 500x slowdowns and inaccurate results).

The stopping criterion is adjusted for warm-started solves (`reltol` is
measured against `‖M b‖` as in a cold start, not against the warm initial
residual), so warm starting never changes the meaning of the tolerances.
"""
struct KrylovJL{F, I, P, A, K} <: AbstractKrylovSubspaceMethod
    KrylovAlg::F
    gmres_restart::I
    window::I
    warm_start::Symbol
    precs::P
    args::A
    kwargs::K
end

function KrylovJL(
        args...; KrylovAlg = Krylov.gmres!,
        gmres_restart = 0, window = 0,
        warm_start::Symbol = :none,
        precs = nothing,
        kwargs...
    )
    warm_start in (:none, :previous, :hegedus) ||
        throw(
        ArgumentError(
            "warm_start must be :none, :previous, or :hegedus, got :$warm_start"
        )
    )
    return KrylovJL(
        KrylovAlg, gmres_restart, window, warm_start,
        precs, args, kwargs
    )
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
    return KrylovJL(args...; KrylovAlg = Krylov.cg!, kwargs...)
end

"""
```julia
KrylovJL_MINRES(args...; kwargs...)
```

A generic MINRES implementation for Hermitian linear systems
"""
function KrylovJL_MINRES(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.minres!, kwargs...)
end

"""
```julia
KrylovJL_GMRES(args...; gmres_restart = 0, window = 0, warm_start = :none, kwargs...)
```

A generic GMRES implementation for square non-Hermitian linear systems

`warm_start` (`:none`, `:previous`, or `:hegedus`) selects the initial guess
used when the same cache is solved repeatedly; see [`KrylovJL`](@ref).
"""
function KrylovJL_GMRES(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.gmres!, kwargs...)
end

"""
```julia
KrylovJL_FGMRES(args...; gmres_restart = 0, window = 0, warm_start = :none, kwargs...)
```

A generic FGMRES implementation for square non-Hermitian linear systems

`warm_start` (`:none`, `:previous`, or `:hegedus`) selects the initial guess
used when the same cache is solved repeatedly; see [`KrylovJL`](@ref).
"""
function KrylovJL_FGMRES(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.fgmres!, kwargs...)
end

"""
```julia
KrylovJL_BICGSTAB(args...; kwargs...)
```

A generic BICGSTAB implementation for square non-Hermitian linear systems
"""
function KrylovJL_BICGSTAB(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.bicgstab!, kwargs...)
end

"""
```julia
KrylovJL_LSMR(args...; kwargs...)
```

A generic LSMR implementation for least-squares problems
"""
function KrylovJL_LSMR(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.lsmr!, kwargs...)
end

"""
```julia
KrylovJL_CRAIGMR(args...; kwargs...)
```

A generic CRAIGMR implementation for least-norm problems
"""
function KrylovJL_CRAIGMR(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.craigmr!, kwargs...)
end

"""
```julia
KrylovJL_MINARES(args...; kwargs...)
```

A generic MINARES implementation for Hermitian linear systems
"""
function KrylovJL_MINARES(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.minares!, kwargs...)
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
function init_cacheval(
        alg::KrylovJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions; zeroinit = true
    )
    KS = get_KrylovJL_solver(alg.KrylovAlg)

    if zeroinit
        solver = if (
                alg.KrylovAlg === Krylov.dqgmres! ||
                    alg.KrylovAlg === Krylov.diom! ||
                    alg.KrylovAlg === Krylov.gmres! ||
                    alg.KrylovAlg === Krylov.fgmres! ||
                    alg.KrylovAlg === Krylov.gpmr! ||
                    alg.KrylovAlg === Krylov.fom!
            )
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
        # Check if memory is specified in kwargs, otherwise compute from gmres_restart
        kwargs_nt = NamedTuple(alg.kwargs)
        memory = if haskey(kwargs_nt, :memory)
            kwargs_nt[:memory]
        elseif alg.gmres_restart == 0
            min(20, size(A, 1))
        else
            alg.gmres_restart
        end

        solver = if (
                alg.KrylovAlg === Krylov.dqgmres! ||
                    alg.KrylovAlg === Krylov.diom! ||
                    alg.KrylovAlg === Krylov.gmres! ||
                    alg.KrylovAlg === Krylov.fgmres! ||
                    alg.KrylovAlg === Krylov.gpmr! ||
                    alg.KrylovAlg === Krylov.fom!
            )
            KS(A, b; memory)
        elseif (
                alg.KrylovAlg === Krylov.minres! ||
                    alg.KrylovAlg === Krylov.symmlq! ||
                    alg.KrylovAlg === Krylov.lslq! ||
                    alg.KrylovAlg === Krylov.lsqr! ||
                    alg.KrylovAlg === Krylov.lsmr!
            )
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
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, ::LinearSolve.OperatorAssumptions
    )
    return nothing
end

# Batched (matrix) right-hand sides: Krylov.jl provides block methods for GMRES
# and MINRES, so those get real block workspaces; the other methods have no
# block variant and error informatively at `init` time
# (`_check_batched_rhs_support`). The `nothing` fallback exists so the default
# polyalgorithm can still initialize its (unused) Krylov cacheval slots when a
# factorization algorithm is chosen for a batched problem.
function init_cacheval(
        alg::LinearSolve.KrylovJL, A, b::AbstractMatrix, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        ::LinearSolve.OperatorAssumptions; zeroinit = true
    )
    if alg.KrylovAlg === Krylov.gmres!
        return Krylov.BlockGmresWorkspace(A, b)
    elseif alg.KrylovAlg === Krylov.minres!
        return Krylov.BlockMinresWorkspace(A, b)
    end
    return nothing
end

# Krylov.jl provides block methods for GMRES and MINRES, so those KrylovJL
# variants support batched right-hand sides natively (via BlockGmresWorkspace /
# BlockMinresWorkspace); the remaining Krylov methods have no block variant.
function _check_batched_rhs_support(alg::KrylovJL, b::AbstractMatrix)
    (alg.KrylovAlg === Krylov.gmres! || alg.KrylovAlg === Krylov.minres!) &&
        return nothing
    throw(
        ArgumentError(
            "$(nameof(typeof(alg))) with $(alg.KrylovAlg) supports only vector `b`: " *
                "Krylov.jl provides block (batched) methods only for GMRES and MINRES. " *
                "Use KrylovJL_GMRES/KrylovJL_MINRES, a factorization algorithm, or " *
                "solve column-by-column."
        )
    )
end

# Krylov.jl workspaces the `warm_start` option applies to: square-system
# solvers with `Krylov.warm_start!` support where restarting from the previous
# solution is meaningful.
const _WARM_STARTABLE_WORKSPACES = Union{Krylov.GmresWorkspace, Krylov.FgmresWorkspace}

"""
    _krylov_warm_start!(workspace, cache, mode, M, atol, rtol) -> (atol, rtol)

Warm start `workspace` from the previous solution `cache.u` (raw for
`mode === :previous`, Hegedüs-rescaled for `mode === :hegedus`) and return the
adjusted stopping tolerances. Krylov.jl measures `rtol` against the warm-start
residual `‖M (b - A x₀)‖` rather than `‖M b‖`, so `rtol * ‖M b‖` is folded
into `atol` (and `rtol` zeroed) to keep the stopping threshold identical to a
cold start's. No-op (returning the tolerances unchanged) for unsupported
workspaces and for zero or nonfinite previous solutions.
"""
function _krylov_warm_start!(workspace, cache, mode::Symbol, M, atol, rtol)
    workspace isa _WARM_STARTABLE_WORKSPACES || return atol, rtol
    u = cache.u
    (u isa AbstractVector && eltype(u) <: Number) || return atol, rtol
    unorm = norm(u)
    (iszero(unorm) || !isfinite(unorm)) && return atol, rtol
    if mode === :hegedus
        Au = mul!(similar(cache.b), cache.A, u)
        d = real(dot(Au, Au))
        (iszero(d) || !isfinite(d)) && return atol, rtol
        Krylov.warm_start!(workspace, (dot(Au, cache.b) / d) .* u)
    else
        Krylov.warm_start!(workspace, u)
    end
    bnorm = M === I ? norm(cache.b) : norm(ldiv!(similar(cache.b), M, cache.b))
    return atol + rtol * bnorm, zero(rtol)
end

function SciMLBase.solve!(cache::LinearCache, alg::KrylovJL; kwargs...)
    if cache.precsisfresh && !isnothing(alg.precs)
        Pl, Pr = alg.precs(cache.A, cache.p)
        cache.Pl = Pl
        cache.Pr = Pr
        cache.precsisfresh = false
    end
    if cache.isfresh
        solver = init_cacheval(
            alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol, cache.verbose,
            cache.assumptions, zeroinit = false
        )
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
    verbose = cache.verbose

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

    krylovJL_verbose = verbosity_to_int(verbose.KrylovJL_verbosity)

    if alg.warm_start !== :none
        atol, rtol = _krylov_warm_start!(cacheval, cache, alg.warm_start, M, atol, rtol)
    end

    args = (cacheval, cache.A, cache.b)
    # Filter out workspace creation parameters (memory, window) from kwargs
    # These parameters are only used when creating the workspace, not when solving
    kwargs_nt = NamedTuple(alg.kwargs)
    filtered_kwargs = Base.structdiff(kwargs_nt, NamedTuple{(:memory, :window)})
    kwargs = (
        atol = atol, rtol, itmax, verbose = krylovJL_verbose,
        ldiv = true, history = true, filtered_kwargs...,
    )

    if cache.cacheval isa Krylov.CgWorkspace
        N !== I &&
            @SciMLMessage(
            "$(alg.KrylovAlg) doesn't support right preconditioning.",
            verbose, :no_right_preconditioning
        )
        Krylov.krylov_solve!(args...; M, kwargs...)
    elseif cache.cacheval isa Krylov.GmresWorkspace
        Krylov.krylov_solve!(args...; M, N, restart = alg.gmres_restart > 0, kwargs...)
    elseif cache.cacheval isa Krylov.FgmresWorkspace
        Krylov.krylov_solve!(args...; M, N, kwargs...)
    elseif cache.cacheval isa Krylov.BicgstabWorkspace
        Krylov.krylov_solve!(args...; M, N, kwargs...)
    elseif cache.cacheval isa Krylov.MinresWorkspace
        N !== I &&
            @SciMLMessage(
            "$(alg.KrylovAlg) doesn't support right preconditioning.",
            verbose, :no_right_preconditioning
        )
        Krylov.krylov_solve!(args...; M, kwargs...)
    elseif cache.cacheval isa Krylov.BlockGmresWorkspace
        Krylov.krylov_solve!(args...; M, N, restart = alg.gmres_restart > 0, kwargs...)
    elseif cache.cacheval isa Krylov.BlockMinresWorkspace
        N !== I &&
            @SciMLMessage(
            "$(alg.KrylovAlg) doesn't support right preconditioning.",
            verbose, :no_right_preconditioning
        )
        Krylov.krylov_solve!(args...; M, kwargs...)
    elseif cache.cacheval isa Krylov.LsmrWorkspace ||
            cache.cacheval isa Krylov.LsqrWorkspace ||
            cache.cacheval isa Krylov.LslqWorkspace
        Krylov.krylov_solve!(args...; M, N, kwargs...)
    elseif cache.cacheval isa Krylov.CglsWorkspace ||
            cache.cacheval isa Krylov.CrlsWorkspace
        N !== I &&
            @SciMLMessage(
            "$(alg.KrylovAlg) doesn't support right preconditioning.",
            verbose, :no_right_preconditioning
        )
        Krylov.krylov_solve!(args...; M, kwargs...)
    else
        Krylov.krylov_solve!(args...; kwargs...)
    end

    stats = @get_cacheval(cache, :KrylovJL_GMRES).stats
    resid = !isempty(stats.residuals) ? last(stats.residuals) :
        zero(eltype(stats.residuals))

    retcode = if !stats.solved
        if stats.status == "maximum number of iterations exceeded"
            @SciMLMessage("Solver reached maximum number of iterations", cache.verbose, :max_iters)
            ReturnCode.MaxIters
        elseif stats.status == "solution good enough given atol and rtol"
            @SciMLMessage("Solver failed to converge", cache.verbose, :convergence_failure)
            ReturnCode.ConvergenceFailure
        else
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            ReturnCode.Failure
        end
    else
        ReturnCode.Success
    end

    # Copy the solution to the allocated output array (block workspaces store
    # the batched solution in `X` rather than `x`)
    cacheval = @get_cacheval(cache, :KrylovJL_GMRES)
    xsol = cacheval isa Union{Krylov.BlockGmresWorkspace, Krylov.BlockMinresWorkspace} ?
        cacheval.X : cacheval.x
    if cache.u !== xsol && ArrayInterface.can_setindex(cache.u)
        cache.u .= xsol
    else
        cache.u = convert(typeof(cache.u), xsol)
    end

    return SciMLBase.build_linear_solution(
        alg, cache.u, Ref(resid), nothing;
        iters = stats.niter, retcode, stats
    )
end

update_tolerances_internal!(cache, alg::KrylovJL, atol, rtol) = nothing
