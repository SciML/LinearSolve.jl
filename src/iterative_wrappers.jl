## Krylov.jl

"""
    EnumX.@enumx WarmStart

Selects the initial guess used by a [`KrylovJL`](@ref) GMRES/FGMRES algorithm
when the same cache is solved repeatedly (`cache.b = newb; solve!(cache)`), e.g.
the sequence of linear solves inside an implicit ODE integrator's Newton
iteration. Other Krylov methods ignore the setting. The previous solution stored
in `cache.u` seeds the next solve via `Krylov.warm_start!`.

Warm-started solves keep the cold-start stopping criterion: `reltol` is measured
against `‖M b‖`, not against the warm initial residual, so the tolerances mean
the same thing in every mode.

!!! warning

    Do not use warm starting inside Rosenbrock-type (W-method) integrators such
    as `Rodas5P`. They have no outer Newton iteration to absorb within-tolerance
    differences in stage solves, and warm starting there can degrade accuracy
    and trigger step-rejection feedback loops (observed: `WarmStart.Previous`
    producing 500x slowdowns and inaccurate results).
"""
EnumX.@enumx WarmStart begin
    """
    `WarmStart.Auto`

    Default. Let the context decide. In standalone LinearSolve use this behaves
    as `WarmStart.None` (cold start), so it never changes behavior on its own. A
    higher-level caller that knows the surrounding algorithm may resolve it to a
    concrete mode.
    """
    Auto
    """
    `WarmStart.None`

    Every solve starts from a zero initial guess (cold start).
    """
    None
    """
    `WarmStart.Previous`

    Start from the previous solution `cache.u` unchanged. Only appropriate when
    successive *solutions* vary slowly. Inside a Newton iteration the linear
    unknown is the Newton increment, whose magnitude shrinks as the iteration
    converges, so the raw previous increment typically overshoots and *increases*
    the iteration count; prefer `WarmStart.Hegedus` there. Costs one extra
    operator application per solve.
    """
    Previous
    """
    `WarmStart.Hegedus`

    Start from the previous solution rescaled by the Hegedüs trick, `x₀ = ξ u`
    with `ξ = ⟨Au, b⟩ / ‖Au‖²`, which minimizes the initial residual along the
    direction of the previous solution and hence guarantees `‖b - A x₀‖ ≤ ‖b‖`
    (never worse than a cold start). The guess is used only when it reduces the
    residual by at least a factor of two; otherwise the solve starts cold. This
    rejects nearly orthogonal previous directions whose rescaling would amplify
    round-off without materially improving the initial residual. Costs two extra
    operator applications per solve, plus one preconditioner application when a
    left preconditioner is set.

    Prefer this to `WarmStart.Previous` when warm starting is explicitly wanted,
    but benchmark against a cold start. The projection can reduce Krylov work
    when successive systems are predictive, while on other residual spectra it
    can increase the iteration count. The fixed per-solve overhead can also
    dominate when the cold solve takes only a few iterations.
    """
    Hegedus
end

"""
    KrylovJL(args...; KrylovAlg = Krylov.gmres!, gmres_restart = 0, window = 0,
        warm_start = WarmStart.Auto, precs = nothing, kwargs...)

A generic wrapper over the Krylov.jl Krylov-subspace iterative solvers. The chosen
in-place solver is run through `Krylov.krylov_solve!` on a Krylov.jl workspace
that `solve!` builds on the first solve, rebuilds whenever `A` has been replaced
(`cache.A = newA` or `reinit!` with a new `A`), and otherwise reuses across
solves of the same cache (for example when only `b` changes). Preconditioners,
whether given as `Pl`/`Pr` to `init`/`solve` or built by `precs`, are handed to
Krylov.jl as its `M` (left) and `N` (right) operators, for the methods that take
them (see the last paragraph). The named constructors `KrylovJL_GMRES`,
`KrylovJL_CG`, `KrylovJL_MINRES`, `KrylovJL_FGMRES`, `KrylovJL_BICGSTAB`,
`KrylovJL_LSMR`, `KrylovJL_CRAIGMR`, and `KrylovJL_MINARES` fix `KrylovAlg` and
forward everything else here.

## Positional Arguments

  - `args...`: stored on the algorithm but not used by `solve!`.

## Keyword Arguments

  - `KrylovAlg`: the Krylov.jl in-place solver function to run. It must be one of
    the `Krylov.<method>!` functions LinearSolve knows how to build a workspace
    for: `cg!`, `cr!`, `cgs!`, `minres!`, `minres_qlp!`, `minares!`, `symmlq!`,
    `cg_lanczos!`, `gmres!`, `fgmres!`, `dqgmres!`, `diom!`, `fom!`, `gpmr!`,
    `bicgstab!`, `bilq!`, `bilqr!`, `qmr!`, `usymlq!`, `usymqr!`, `tricg!`,
    `trimr!`, `trilqr!`, `cgls!`, `crls!`, `lsqr!`, `lslq!`, `lsmr!`, `cgne!`,
    `crmr!`, `lnlq!`, `craig!`, or `craigmr!`. Any other value errors with
    "Invalid Krylov method detected". Defaults to `Krylov.gmres!`.
  - `gmres_restart`: Krylov subspace size for the GMRES-like methods (`gmres!`,
    `fgmres!`, `dqgmres!`, `diom!`, `fom!`, `gpmr!`). `0` sizes the workspace with
    `memory = min(20, size(A, 1))` and runs GMRES without restarting; a positive
    value sets `memory = gmres_restart` and, for `gmres!` only, also passes
    `restart = true` so that GMRES restarts every `gmres_restart` iterations. For
    the other GMRES-like methods it only sizes the workspace (pass
    `restart = true` in `kwargs` to make FGMRES restart). Ignored by every other
    method. Defaults to `0`.
  - `window`: the `window` argument of the Krylov.jl workspace constructor for
    `minres!`, `symmlq!`, `lslq!`, `lsqr!`, and `lsmr!` (the number of iterations
    used to estimate a lower bound on the error), forwarded when nonzero. Ignored
    by every other method. Defaults to `0`.
  - `warm_start`: a [`WarmStart`](@ref) value selecting the initial guess used when
    the same cache is solved repeatedly (GMRES and FGMRES only): `WarmStart.Auto`
    (cold start unless a caller resolves it), `WarmStart.None`,
    `WarmStart.Previous`, or the recommended `WarmStart.Hegedus`. Defaults to
    `WarmStart.Auto`.
  - `precs`: a preconditioner builder, a function `(A, p) -> (Pl, Pr)`. It is
    called at `init` (explicit `Pl`/`Pr` passed to `init`/`solve` take
    precedence) and again in `solve!` whenever `A` or `p` has changed since the
    preconditioners were last built (`reinit!` without `reuse_precs = true`), and
    its result becomes `cache.Pl`/`cache.Pr`. `nothing` means no preconditioning
    unless `Pl`/`Pr` are given to `init`/`solve`. Defaults to `nothing`.
  - `kwargs...`: any remaining keywords are forwarded to `Krylov.krylov_solve!` on
    every solve (for example `callback`, `reorthogonalization`, `timemax`, or
    `restart`), on top of the `atol`/`rtol` tolerances, `itmax = maxiters`,
    `verbose`, `ldiv = true`, and `history = true` that LinearSolve supplies. Two
    exceptions are consumed at workspace construction instead of being
    forwarded: `memory` overrides the subspace size derived from
    `gmres_restart`, and `window` is dropped (use the `window` keyword above).

Right preconditioning is only available for some methods. For `cg!`, `minres!`,
`cgls!`, and `crls!` a non-identity `Pr` triggers the `no_right_preconditioning`
verbosity message and is discarded, since Krylov.jl takes only a centered
preconditioner `M` for them. `Pl` and `Pr` are passed through for the GMRES,
FGMRES, BiCGSTAB, LSMR, LSQR, and LSLQ workspaces. For every other `KrylovAlg`
(including `craigmr!` and `minares!`), `solve!` calls the Krylov solver without
`M`/`N`, so any `Pl`/`Pr` are silently ignored.
"""
struct KrylovJL{F, I, P, A, K} <: AbstractKrylovSubspaceMethod
    KrylovAlg::F
    gmres_restart::I
    window::I
    warm_start::WarmStart.T
    precs::P
    args::A
    kwargs::K
end

function KrylovJL(
        args...; KrylovAlg = Krylov.gmres!,
        gmres_restart = 0, window = 0,
        warm_start::WarmStart.T = WarmStart.Auto,
        precs = nothing,
        kwargs...
    )
    return KrylovJL(
        KrylovAlg, gmres_restart, window, warm_start,
        precs, args, kwargs
    )
end

default_alias_A(::KrylovJL, ::Any, ::Any) = true
default_alias_b(::KrylovJL, ::Any, ::Any) = true

"""
    KrylovJL_CG(args...; kwargs...)

Conjugate gradient for Hermitian positive definite linear systems, wrapping
`Krylov.cg!` via `KrylovJL` (equivalent to
`KrylovJL(args...; KrylovAlg = Krylov.cg!, kwargs...)`). All keyword arguments
(`precs`, and any Krylov.jl solve keywords such as `callback`) are those of
`KrylovJL`; `gmres_restart` and `window` have no effect here. Only left
(centered) preconditioning is supported: a right preconditioner `Pr` triggers
the `no_right_preconditioning` verbosity message and is discarded.
"""
function KrylovJL_CG(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.cg!, kwargs...)
end

"""
    KrylovJL_MINRES(args...; window = 0, kwargs...)

MINRES for Hermitian (possibly indefinite) linear systems, wrapping
`Krylov.minres!` via `KrylovJL` (equivalent to
`KrylovJL(args...; KrylovAlg = Krylov.minres!, kwargs...)`). Keyword arguments
are those of `KrylovJL`. `window` sizes the error-estimation window of the
MINRES workspace when nonzero (defaults to `0`, meaning Krylov.jl's default);
`gmres_restart` has no effect here. Only left (centered) preconditioning is
supported: a right preconditioner `Pr` triggers the `no_right_preconditioning`
verbosity message and is discarded. Batched (matrix) right-hand sides are
supported through Krylov.jl's block MINRES.
"""
function KrylovJL_MINRES(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.minres!, kwargs...)
end

"""
    KrylovJL_GMRES(args...; gmres_restart = 0, warm_start = WarmStart.Auto,
        precs = nothing, kwargs...)

GMRES for square non-Hermitian linear systems, wrapping `Krylov.gmres!` via
`KrylovJL` (equivalent to `KrylovJL(args...; KrylovAlg = Krylov.gmres!, kwargs...)`).
This is the general-purpose iterative choice, and the one the default
polyalgorithm falls back to for operators without a matrix representation.

## Keyword Arguments

  - `gmres_restart`: `0` allocates a workspace with `memory = min(20, size(A, 1))`
    and runs GMRES without restarting; a positive value sets the workspace memory
    to `gmres_restart` and passes `restart = true`, so GMRES restarts every
    `gmres_restart` iterations. Defaults to `0`.
  - `warm_start`: a [`WarmStart`](@ref) value selecting the initial guess used
    when the same cache is solved repeatedly; see `KrylovJL` and `WarmStart`.
    Defaults to `WarmStart.Auto`.
  - `precs`: a preconditioner builder `(A, p) -> (Pl, Pr)`; see `KrylovJL`.
    Defaults to `nothing`.
  - `kwargs...`: forwarded to `Krylov.krylov_solve!` as described for `KrylovJL`
    (`memory` overrides the subspace size derived from `gmres_restart`).

Both left and right preconditioners are supported. `window` is accepted but has
no effect for GMRES. Batched (matrix) right-hand sides are supported through
Krylov.jl's block GMRES.
"""
function KrylovJL_GMRES(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.gmres!, kwargs...)
end

"""
    KrylovJL_FGMRES(args...; gmres_restart = 0, warm_start = WarmStart.Auto,
        precs = nothing, kwargs...)

Flexible GMRES for square non-Hermitian linear systems, wrapping `Krylov.fgmres!`
via `KrylovJL` (equivalent to
`KrylovJL(args...; KrylovAlg = Krylov.fgmres!, kwargs...)`). Use it in place of
`KrylovJL_GMRES` when the right preconditioner changes between iterations, for
example when it is itself an iterative solve.

## Keyword Arguments

  - `gmres_restart`: `0` allocates a workspace with `memory = min(20, size(A, 1))`;
    a positive value sets the workspace memory to `gmres_restart`. Unlike
    `KrylovJL_GMRES`, no `restart` flag is passed for FGMRES, so this only sizes
    the workspace; pass `restart = true` in `kwargs` to make FGMRES restart every
    `gmres_restart` iterations. Defaults to `0`.
  - `warm_start`: a [`WarmStart`](@ref) value selecting the initial guess used
    when the same cache is solved repeatedly; see `KrylovJL` and `WarmStart`.
    Defaults to `WarmStart.Auto`.
  - `precs`: a preconditioner builder `(A, p) -> (Pl, Pr)`; see `KrylovJL`.
    Defaults to `nothing`.
  - `kwargs...`: forwarded to `Krylov.krylov_solve!` as described for `KrylovJL`
    (`memory` overrides the subspace size derived from `gmres_restart`).

Both left and right preconditioners are supported. `window` is accepted but has
no effect for FGMRES.
"""
function KrylovJL_FGMRES(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.fgmres!, kwargs...)
end

"""
    KrylovJL_BICGSTAB(args...; kwargs...)

BiCGSTAB for square non-Hermitian linear systems, wrapping `Krylov.bicgstab!` via
`KrylovJL` (equivalent to
`KrylovJL(args...; KrylovAlg = Krylov.bicgstab!, kwargs...)`). Its memory use
does not grow with the iteration count, unlike unrestarted GMRES, at the cost of
a less regular convergence history. All keyword arguments (`precs`, and any Krylov.jl solve
keywords such as `callback`) are those of `KrylovJL`; `gmres_restart` and
`window` have no effect here. Both left and right preconditioners are supported.
"""
function KrylovJL_BICGSTAB(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.bicgstab!, kwargs...)
end

"""
    KrylovJL_LSMR(args...; window = 0, kwargs...)

LSMR for least-squares problems (rectangular or rank-deficient `A`, minimizing
`‖b - A x‖`), wrapping `Krylov.lsmr!` via `KrylovJL` (equivalent to
`KrylovJL(args...; KrylovAlg = Krylov.lsmr!, kwargs...)`). It is the default
polyalgorithm's choice for tall systems whose `A` is an operator without a matrix
representation. Keyword arguments are those of `KrylovJL`. `window` sizes the
error-estimation window of the LSMR workspace when nonzero (defaults to `0`,
meaning Krylov.jl's default); `gmres_restart` has no effect here. Both left and
right preconditioners are passed through to Krylov.jl.
"""
function KrylovJL_LSMR(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.lsmr!, kwargs...)
end

"""
    KrylovJL_CRAIGMR(args...; kwargs...)

CRAIGMR for least-norm problems (underdetermined `A`, returning the minimum-norm
solution of `A x = b`), wrapping `Krylov.craigmr!` via `KrylovJL` (equivalent to
`KrylovJL(args...; KrylovAlg = Krylov.craigmr!, kwargs...)`). It is the default
polyalgorithm's choice for wide systems whose `A` is an operator without a matrix
representation. Keyword arguments (`precs`, and any Krylov.jl solve keywords)
are those of `KrylovJL`; `gmres_restart` and `window` have no effect here.
Preconditioners are not passed through for this method:
`solve!` calls `Krylov.craigmr!` without `M`/`N`, so any `Pl`/`Pr` supplied to
`init`/`solve` or built by `precs` are silently ignored.
"""
function KrylovJL_CRAIGMR(args...; kwargs...)
    return KrylovJL(args...; KrylovAlg = Krylov.craigmr!, kwargs...)
end

"""
    KrylovJL_MINARES(args...; kwargs...)

MINARES for Hermitian (possibly indefinite or singular) linear systems,
wrapping `Krylov.minares!` via `KrylovJL` (equivalent to
`KrylovJL(args...; KrylovAlg = Krylov.minares!, kwargs...)`). It minimizes the
norm of `A r` rather than that of the residual `r`, and is an alternative to
`KrylovJL_MINRES` for singular or nearly singular Hermitian systems. Keyword
arguments (`precs`, and any Krylov.jl solve keywords) are those of `KrylovJL`;
`gmres_restart` and `window` have no effect here. Preconditioners are not passed
through for this method: `solve!` calls `Krylov.minares!` without `M`/`N`, so
any `Pl`/`Pr` supplied to `init`/`solve` or built by `precs` are silently
ignored.
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

# Krylov.jl allocates its workspace as `S(undef, n)` with `S = typeof(b)`, which an
# `ArrayPartition` cannot provide: it is stored as several separate arrays, so there
# is no way to know how to split `n` across them, and its constructor says so rather
# than guessing. `Krylov.KrylovConstructor` builds the workspace with `similar`
# instead, which an `ArrayPartition` does support, so the solve runs on the
# partitioned vectors directly with no flattening and no copying.
#
# This covers array types only. A right-hand side that is not an array at all, such
# as a parameter object implementing the SciMLStructures interface, supports neither
# `S(undef, n)` nor `similar`, and needs canonicalizing to a flat buffer instead.
# That case is not handled here.
#
# The previous method here returned `nothing` to dodge the workspace allocation, but
# it declared no `zeroinit`, so the `solve!` path (which always passes
# `zeroinit = false`) never reached it and hit the error anyway.
#
# `solve!` needs no specialization: the workspace holds the caller's array type
# throughout, so the generic path writes the answer straight into it.
# See SciML/LinearSolve.jl#384.
function init_cacheval(
        alg::LinearSolve.KrylovJL, A, b::RecursiveArrayTools.ArrayPartition, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::LinearSolve.OperatorAssumptions; zeroinit = true
    )
    KS = get_KrylovJL_solver(alg.KrylovAlg)
    kwargs_nt = NamedTuple(alg.kwargs)
    # `b` sizes the range vectors and `u` the domain ones, so a rectangular operator
    # gets the right shape on each side.
    constructor = Krylov.KrylovConstructor(b, u)

    solver = if (
            alg.KrylovAlg === Krylov.dqgmres! ||
                alg.KrylovAlg === Krylov.diom! ||
                alg.KrylovAlg === Krylov.gmres! ||
                alg.KrylovAlg === Krylov.fgmres! ||
                alg.KrylovAlg === Krylov.gpmr! ||
                alg.KrylovAlg === Krylov.fom!
        )
        memory = if haskey(kwargs_nt, :memory)
            kwargs_nt[:memory]
        elseif alg.gmres_restart == 0
            min(20, size(A, 1))
        else
            alg.gmres_restart
        end
        KS(constructor; memory)
    elseif (
            alg.KrylovAlg === Krylov.minres! ||
                alg.KrylovAlg === Krylov.symmlq! ||
                alg.KrylovAlg === Krylov.lslq! ||
                alg.KrylovAlg === Krylov.lsqr! ||
                alg.KrylovAlg === Krylov.lsmr!
        )
        (alg.window != 0) ? KS(constructor; window = alg.window) : KS(constructor)
    else
        KS(constructor)
    end

    solver.x = u
    return solver
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

const _HEGEDUS_MAX_RESIDUAL_RATIO = 0.5
const _HEGEDUS_MIN_COSINE = sqrt(1 - _HEGEDUS_MAX_RESIDUAL_RATIO^2)

"""
    _krylov_warm_start!(workspace, cache, mode, M, atol, rtol) -> (atol, rtol)

Warm start `workspace` from the previous solution `cache.u` (raw for
`WarmStart.Previous`, Hegedüs-rescaled for `WarmStart.Hegedus`) and return the
adjusted stopping tolerances. Krylov.jl measures `rtol` against the warm-start
residual `‖M (b - A x₀)‖` rather than `‖M b‖`, so `rtol * ‖M b‖` is folded
into `atol` (and `rtol` zeroed) to keep the stopping threshold identical to a
cold start's. No-op (returning the tolerances unchanged) for unsupported
workspaces and for zero or nonfinite previous solutions.
"""
function _krylov_warm_start!(workspace, cache, mode::WarmStart.T, M, atol, rtol)
    workspace isa _WARM_STARTABLE_WORKSPACES || return atol, rtol
    u = cache.u
    (u isa AbstractVector && eltype(u) <: Number) || return atol, rtol
    unorm = norm(u)
    (iszero(unorm) || !isfinite(unorm)) && return atol, rtol
    if mode == WarmStart.Hegedus
        Au = mul!(similar(cache.b), cache.A, u)
        d = real(dot(Au, Au))
        (iszero(d) || !isfinite(d)) && return atol, rtol
        Aub = dot(Au, cache.b)
        isfinite(Aub) || return atol, rtol
        bnorm = norm(cache.b)
        (iszero(bnorm) || !isfinite(bnorm)) && return atol, rtol
        abs(Aub) < _HEGEDUS_MIN_COSINE * sqrt(d) * bnorm && return atol, rtol
        Krylov.warm_start!(workspace, (Aub / d) .* u)
        bnorm = M === I ? bnorm : norm(ldiv!(similar(cache.b), M, cache.b))
    else
        Krylov.warm_start!(workspace, u)
        bnorm = M === I ? norm(cache.b) : norm(ldiv!(similar(cache.b), M, cache.b))
    end
    return atol + rtol * bnorm, zero(rtol)
end

# The methods for symmetric systems below take centered preconditioning only, and warn on
# and discard a right preconditioner. Keep this in step with the dispatch in `solve!`.
function _supports_right_preconditioning(alg::KrylovJL)
    return !(
        alg.KrylovAlg === Krylov.cg! || alg.KrylovAlg === Krylov.minres! ||
            alg.KrylovAlg === Krylov.block_minres! || alg.KrylovAlg === Krylov.cgls! ||
            alg.KrylovAlg === Krylov.crls!
    )
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

    # Auto resolves to a cold start in standalone LinearSolve; only Previous and
    # Hegedus actually warm start (a context-aware caller rewrites Auto upstream).
    if alg.warm_start == WarmStart.Previous || alg.warm_start == WarmStart.Hegedus
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

    if cacheval isa Krylov.CgWorkspace
        N !== I &&
            @SciMLMessage(
            "$(alg.KrylovAlg) doesn't support right preconditioning.",
            verbose, :no_right_preconditioning
        )
        Krylov.krylov_solve!(args...; M, kwargs...)
    elseif cacheval isa Krylov.GmresWorkspace
        Krylov.krylov_solve!(args...; M, N, restart = alg.gmres_restart > 0, kwargs...)
    elseif cacheval isa Krylov.FgmresWorkspace
        Krylov.krylov_solve!(args...; M, N, kwargs...)
    elseif cacheval isa Krylov.BicgstabWorkspace
        Krylov.krylov_solve!(args...; M, N, kwargs...)
    elseif cacheval isa Krylov.MinresWorkspace
        N !== I &&
            @SciMLMessage(
            "$(alg.KrylovAlg) doesn't support right preconditioning.",
            verbose, :no_right_preconditioning
        )
        Krylov.krylov_solve!(args...; M, kwargs...)
    elseif cacheval isa Krylov.BlockGmresWorkspace
        Krylov.krylov_solve!(args...; M, N, restart = alg.gmres_restart > 0, kwargs...)
    elseif cacheval isa Krylov.BlockMinresWorkspace
        N !== I &&
            @SciMLMessage(
            "$(alg.KrylovAlg) doesn't support right preconditioning.",
            verbose, :no_right_preconditioning
        )
        Krylov.krylov_solve!(args...; M, kwargs...)
    elseif cacheval isa Krylov.LsmrWorkspace ||
            cacheval isa Krylov.LsqrWorkspace ||
            cacheval isa Krylov.LslqWorkspace
        Krylov.krylov_solve!(args...; M, N, kwargs...)
    elseif cacheval isa Krylov.CglsWorkspace ||
            cacheval isa Krylov.CrlsWorkspace
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
