"""
    SimpleGMRES(; restart::Bool = true, blocksize::Int = 0, warm_start::Bool = false,
        memory::Int = 20)

A simple GMRES implementation for square non-Hermitian linear systems.

This implementation handles Block Diagonal Matrices with Uniformly Sized Square Blocks with
specialized dispatches.

## Arguments

  - `restart::Bool`: If `true`, then the solver will restart after `memory` iterations.

  - `memory::Int = 20`: The number of iterations before restarting. If restart is false, this
    value is used to allocate memory and later expanded if more memory is required.
  - `blocksize::Int = 0`: If blocksize is `> 0`, the solver assumes that the matrix has a
    uniformly sized block diagonal structure with square blocks of size `blocksize`. Misusing
    this option will lead to incorrect results.

      + If this is set `≤ 0` and during runtime we get a Block Diagonal Matrix, then we will
        check if the specialized dispatch can be used.
  - `warm_start::Bool = false`: If `true`, the initial residual is formed as `b - A*Δx` from
    the cache's internal `Δx` buffer instead of as `b`, and `Δx` is folded into the returned
    solution (following the Krylov.jl `warm_start` convention). Note that in the current
    implementation nothing copies the problem's `u0` into `Δx`: with `restart = true` the
    buffer is uninitialized, and with `restart = false` it has length zero (so the first
    `solve!` throws a `DimensionMismatch`). This option therefore does not seed the
    iteration from `u0`. Leave it at the default `false`.

!!! warning

    Most users should be using the `KrylovJL_GMRES` solver instead of this implementation.

!!! tip

    We can automatically detect if the matrix is a Block Diagonal Matrix with Uniformly
    Sized Square Blocks. If this is the case, then we can use a specialized dispatch.
    However, on most modern systems performing a single matrix-vector multiplication is
    faster than performing multiple smaller matrix-vector multiplications (as in the case
    of Block Diagonal Matrix). We recommend making the matrix dense (if size permits) and
    specifying the `blocksize` argument.
"""
struct SimpleGMRES{UBD} <: AbstractKrylovSubspaceMethod
    restart::Bool
    memory::Int
    blocksize::Int
    warm_start::Bool

    function SimpleGMRES{UBD}(;
            restart::Bool = true, blocksize::Int = 0,
            warm_start::Bool = false, memory::Int = 20
        ) where {UBD}
        UBD && @assert blocksize > 0
        return new{UBD}(restart, memory, blocksize, warm_start)
    end

    function SimpleGMRES(;
            restart::Bool = true, blocksize::Int = 0,
            warm_start::Bool = false, memory::Int = 20
        )
        return SimpleGMRES{blocksize > 0}(;
            restart, memory, blocksize,
            warm_start
        )
    end
end

@concrete mutable struct SimpleGMRESCache{UBD}
    memory::Int
    n::Int
    restart::Bool
    maxiters::Int
    blocksize::Int
    ε
    PlisI::Bool
    PrisI::Bool
    Pl
    Pr
    Δx
    q
    p
    x
    A
    b
    abstol
    reltol
    w
    V
    s
    c
    z
    R
    β
    warm_start::Bool
end

"""
    reinit_cacheval!(cacheval::SimpleGMRESCache, b)

Re-establish the starting iterate and the initial residual `r₀` that
`solve!(::SimpleGMRESCache, ::LinearCache)` consumes, so that the cache can be
solved again against `b`.

A `SimpleGMRESCache` is single-use as built: `w` doubles as the Arnoldi scratch
vector `ANvₖ`, and the solution is accumulated into `x` in place. Re-entering
`solve!` without this leaves `r₀` holding the last Krylov vector and `x` holding
the previous solution, and leaves the tolerance `ε` scaled by the *first* right
hand side's norm.
"""
function reinit_cacheval!(cacheval::SimpleGMRESCache, b)
    (; A, Δx, q, x, w, Pl, PlisI, restart, warm_start) = cacheval
    T = eltype(x)

    cacheval.b = b
    fill!(x, zero(T))
    if warm_start
        mul!(w, A, Δx)
        axpby!(one(T), b, -one(T), w)
        restart && axpy!(one(T), Δx, x)
    else
        vec(w) .= vec(b)
    end

    r₀ = PlisI ? w : q
    PlisI || ldiv!(r₀, Pl, w)  # r₀ = Pl(b - Ax₀)
    cacheval.β = _norm2(r₀)
    cacheval.ε = cacheval.abstol + cacheval.reltol * cacheval.β
    return cacheval
end

"""
    (c, s, ρ) = _sym_givens(a, b)

Numerically stable symmetric Givens reflection.
Given `a` and `b` reals, return `(c, s, ρ)` such that

    [ c  s ] [ a ] = [ ρ ]
    [ s -c ] [ b ] = [ 0 ].
"""
function _sym_givens(a::T, b::T) where {T <: AbstractFloat}
    # This has taken from Krylov.jl
    if b == 0
        c = ifelse(a == 0, one(T), sign(a)) # In Julia, sign(0) = 0.
        s = zero(T)
        ρ = abs(a)
    elseif a == 0
        c = zero(T)
        s = sign(b)
        ρ = abs(b)
    elseif abs(b) > abs(a)
        t = a / b
        s = sign(b) / sqrt(one(T) + t * t)
        c = s * t
        ρ = b / s  # Computationally better than ρ = a / c since |c| ≤ |s|.
    else
        t = b / a
        c = sign(a) / sqrt(one(T) + t * t)
        s = c * t
        ρ = a / c  # Computationally better than ρ = b / s since |s| ≤ |c|
    end
    return (c, s, ρ)
end

function _sym_givens!(c, s, R, nr::Int, inner_iter::Int, bsize::Int, Hbis)
    if __is_extension_loaded(Val(:KernelAbstractions))
        return _fast_sym_givens!(c, s, R, nr, inner_iter, bsize, Hbis)
    end
    __res = _sym_givens.(R[nr + inner_iter], Hbis)
    GPUArraysCore.@allowscalar foreach(1:bsize) do i
        c[inner_iter][i] = __res[i][1]
        s[inner_iter][i] = __res[i][2]
        R[nr + inner_iter][i] = __res[i][3]
    end
    return c, s, R
end

_no_preconditioner(::Nothing) = true
_no_preconditioner(::IdentityOperator) = true
_no_preconditioner(::UniformScaling) = true
_no_preconditioner(_) = false

_norm2(x) = norm(x, 2)
_norm2(x, dims) = .√(sum(abs2, x; dims))

default_alias_A(::SimpleGMRES, ::Any, ::Any) = false
default_alias_b(::SimpleGMRES, ::Any, ::Any) = false

# The Arnoldi bookkeeping (`s`, `c`, `z`, `R`) is a scalar per Krylov vector for a
# general matrix, and one entry per diagonal block for a uniform block diagonal
# matrix, where every block runs its own GMRES over a shared Krylov index. That is
# the only thing separating the two cases: the Arnoldi loop, the Givens
# bookkeeping, the stopping criteria and the restart logic are identical. The two
# layouts below carry the element level operations, so the iteration itself is
# written once and `SimpleGMRESCache{UBD}` selects the layout.
struct ScalarLayout end

struct BatchedLayout{F}
    # Reshape a length `n` vector into `(blocksize, bsize)`, one column per block.
    batch::F
    bsize::Int
end

_layout(::Val{false}, blocksize::Int, n::Int) = ScalarLayout()

function _layout(::Val{true}, blocksize::Int, n::Int)
    bsize = n ÷ blocksize
    return BatchedLayout(Base.Fix2(reshape, (blocksize, bsize)), bsize)
end

function _layout(cache::SimpleGMRESCache{UBD}) where {UBD}
    return _layout(Val(UBD), cache.blocksize, cache.n)
end

# Storage for `len` slots of Arnoldi bookkeeping.
_gmres_scratch(::ScalarLayout, u, len::Int) = Vector{eltype(u)}(undef, len)
_gmres_scratch(l::BatchedLayout, u, len::Int) = [similar(u, l.bsize) for _ in 1:len]

# A fresh slot matching what `_gmres_scratch` produced for `proto`.
_gmres_element(::ScalarLayout, proto, ::Type{T}) where {T} = zero(T)
_gmres_element(l::BatchedLayout, proto, ::Type) = similar(first(proto), l.bsize)

# `restart = false` lets the Arnoldi basis outrun `memory`, so the QR bookkeeping
# has to grow with it.
function _gmres_grow_pass!(layout, R, s, c, inner_iter::Int, ::Type{T}) where {T}
    append!(R, [_gmres_element(layout, R, T) for _ in 1:inner_iter])
    push!(s, _gmres_element(layout, s, T))
    push!(c, _gmres_element(layout, c, T))
    return nothing
end

function _gmres_grow_krylov!(layout, V, z, ::Type{T}) where {T}
    push!(V, similar(first(V)))
    push!(z, _gmres_element(layout, z, T))
    return nothing
end

# Initial ζ₁ and V₁ for the current pass.
function _gmres_start!(::ScalarLayout, z, V, r₀)
    β = _norm2(r₀)
    z[1] = β
    V[1] .= r₀ ./ β
    return nothing
end

function _gmres_start!(l::BatchedLayout, z, V, r₀)
    β = _norm2(l.batch(r₀), 1)
    z[1] .= vec(β)
    V[1] .= vec(l.batch(r₀) ./ β)
    return nothing
end

# One Gram-Schmidt step: hᵢₖ = (vᵢ)ᴴq stored at `R[k]`, then q ← q - hᵢₖvᵢ.
function _gmres_orth_step!(::ScalarLayout, R, k::Int, V, i::Int, q)
    R[k] = dot(V[i], q)
    axpy!(-R[k], V[i], q)
    return nothing
end

function _gmres_orth_step!(l::BatchedLayout, R, k::Int, V, i::Int, q)
    sum!(R[k]', l.batch(V[i]) .* l.batch(q))
    q .-= vec(R[k]' .* l.batch(V[i]))
    return nothing
end

# hₖ₊₁.ₖ = ‖vₖ₊₁‖₂
_gmres_hbis(::ScalarLayout, q) = _norm2(q)
_gmres_hbis(l::BatchedLayout, q) = vec(_norm2(l.batch(q), 1))

# Apply the previous Givens reflection Ωᵢ to rows i, i+1 of the current column.
# [cᵢ  sᵢ] [ r̄ᵢ.ₖ ] = [ rᵢ.ₖ ]
# [s̄ᵢ -cᵢ] [rᵢ₊₁.ₖ]   [r̄ᵢ₊₁.ₖ]
function _gmres_apply_givens!(::ScalarLayout, c, s, R, nr::Int, i::Int)
    Rtmp = c[i] * R[nr + i] + s[i] * R[nr + i + 1]
    R[nr + i + 1] = conj(s[i]) * R[nr + i] - c[i] * R[nr + i + 1]
    R[nr + i] = Rtmp
    return nothing
end

function _gmres_apply_givens!(::BatchedLayout, c, s, R, nr::Int, i::Int)
    Rtmp = c[i] .* R[nr + i] .+ s[i] .* R[nr + i + 1]
    R[nr + i + 1] .= conj.(s[i]) .* R[nr + i] .- c[i] .* R[nr + i + 1]
    R[nr + i] .= Rtmp
    return nothing
end

# Compute and apply the current Givens reflection Ωₖ.
# [cₖ  sₖ] [ r̄ₖ.ₖ ] = [rₖ.ₖ]
# [s̄ₖ -cₖ] [hₖ₊₁.ₖ]   [ 0  ]
function _gmres_new_givens!(::ScalarLayout, c, s, R, nr::Int, k::Int, Hbis)
    c[k], s[k], R[nr + k] = _sym_givens(R[nr + k], Hbis)
    return nothing
end

function _gmres_new_givens!(l::BatchedLayout, c, s, R, nr::Int, k::Int, Hbis)
    _sym_givens!(c, s, R, nr, k, l.bsize, Hbis)
    return nothing
end

# Update zₖ = (Qₖ)ᴴβe₁, returning the carried over ζₖ₊₁.
function _gmres_update_z!(::ScalarLayout, z, c, s, k::Int)
    ζₖ₊₁ = conj(s[k]) * z[k]
    z[k] = c[k] * z[k]
    return ζₖ₊₁
end

function _gmres_update_z!(::BatchedLayout, z, c, s, k::Int)
    ζₖ₊₁ = conj.(s[k]) .* z[k]
    z[k] .= c[k] .* z[k]
    return ζₖ₊₁
end

# ‖ Pl(b - Axₖ) ‖₂ = |ζₖ₊₁|. The batched solve stops on its slowest block.
_gmres_rnorm(::ScalarLayout, ζₖ₊₁) = abs(ζₖ₊₁)
_gmres_rnorm(::BatchedLayout, ζₖ₊₁) = maximum(abs, ζₖ₊₁)

_gmres_maximum(::ScalarLayout, Hbis) = Hbis
_gmres_maximum(::BatchedLayout, Hbis) = maximum(Hbis)

# hₖ₊₁.ₖvₖ₊₁ = q
function _gmres_next_v!(::ScalarLayout, V, z, q, Hbis, ζₖ₊₁, k::Int)
    @. V[k + 1] = q / Hbis
    z[k + 1] = ζₖ₊₁
    return nothing
end

function _gmres_next_v!(l::BatchedLayout, V, z, q, Hbis, ζₖ₊₁, k::Int)
    V[k + 1] .= vec(l.batch(q) ./ Hbis')
    z[k + 1] .= ζₖ₊₁
    return nothing
end

# yᵢ ← yᵢ - rᵢⱼyⱼ
function _gmres_backsub_update!(::ScalarLayout, y, i::Int, R, pos::Int, j::Int)
    y[i] = y[i] - R[pos] * y[j]
    return nothing
end

function _gmres_backsub_update!(::BatchedLayout, y, i::Int, R, pos::Int, j::Int)
    y[i] .= y[i] .- R[pos] .* y[j]
    return nothing
end

# yᵢ ← yᵢ / rᵢᵢ. Rₖ can be singular if the system is inconsistent, which is what
# the return value reports.
function _gmres_backsub_solve!(
        ::ScalarLayout, y, i::Int, R, pos::Int, btol, ::Type{T}
    ) where {T}
    if abs(R[pos]) ≤ btol
        y[i] = zero(T)
        return true
    end
    y[i] = y[i] / R[pos]
    return false
end

function _gmres_backsub_solve!(
        ::BatchedLayout, y, i::Int, R, pos::Int, btol, ::Type{T}
    ) where {T}
    singular = abs.(R[pos]) .≤ btol
    y[i] .= ifelse.(singular, zero(T), y[i] ./ R[pos])
    return any(singular)
end

# xₖ ← xₖ + yᵢvᵢ
function _gmres_add_correction!(::ScalarLayout, xr, V, y, i::Int)
    axpy!(y[i], V[i], xr)
    return nothing
end

function _gmres_add_correction!(l::BatchedLayout, xr, V, y, i::Int)
    xr .+= vec(l.batch(V[i]) .* y[i]')
    return nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::SimpleGMRES; kwargs...)
    if cache.isfresh
        solver = init_cacheval(
            alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol, cache.verbose,
            cache.assumptions; zeroinit = false
        )
        cache.cacheval = solver
        cache.isfresh = false
    else
        # Unconditional: an in-place `cache.b .= ...` reaches no hook, and a
        # resolve against an unchanged `b` needs the reset just as much.
        reinit_cacheval!(cache.cacheval, cache.b)
    end
    return SciMLBase.solve!(cache.cacheval, cache)
end

function init_cacheval(alg::SimpleGMRES{UDB}, args...; kwargs...) where {UDB}
    return _init_cacheval(Val(UDB), alg, args...; kwargs...)
end

function _init_cacheval(
        ::Val{UBD}, alg::SimpleGMRES, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, ::Union{LinearVerbosity, Bool}, ::OperatorAssumptions;
        zeroinit = true, blocksize = alg.blocksize, kwargs...
    ) where {UBD}
    (; memory, restart, warm_start) = alg
    # `blocksize` is a property of the batched layout only. A `SimpleGMRES{false}`
    # reaching here through the `BlockDiagonal` dispatch is handed the block size of
    # a matrix it will not treat blockwise, so the scalar cache keeps recording the
    # algorithm's own setting.
    blocksize = UBD ? blocksize : alg.blocksize

    if zeroinit
        layout = _layout(Val(UBD), blocksize, 0)
        return SimpleGMRESCache{UBD}(
            memory, 0, restart, maxiters, blocksize,
            zero(eltype(u)) * reltol + abstol, false, false, Pl, Pr, similar(u, 0),
            similar(u, 0), similar(u, 0), u, A, b, abstol, reltol, similar(u, 0),
            Vector{typeof(u)}(undef, 0), _gmres_scratch(layout, u, 0),
            _gmres_scratch(layout, u, 0), _gmres_scratch(layout, u, 0),
            _gmres_scratch(layout, u, 0), zero(eltype(u)), warm_start
        )
    end

    T = eltype(u)
    n = LinearAlgebra.checksquare(A)
    UBD && @assert mod(n, blocksize) == 0 "The blocksize must divide the size of the matrix."
    @assert n == length(b) "The size of `A` and `b` must match."
    memory = min(memory, maxiters)
    layout = _layout(Val(UBD), blocksize, n)

    PlisI = _no_preconditioner(Pl)
    PrisI = _no_preconditioner(Pr)

    Δx = restart ? similar(u, n) : similar(u, 0)
    q = PlisI ? similar(u, 0) : similar(u, n)
    p = PrisI ? similar(u, 0) : similar(u, n)
    x = u
    x .= zero(T)

    w = similar(u, n)
    V = [similar(u) for _ in 1:memory]
    s = _gmres_scratch(layout, u, memory)
    c = _gmres_scratch(layout, u, memory)
    z = _gmres_scratch(layout, u, memory)
    R = _gmres_scratch(layout, u, (memory * (memory + 1)) ÷ 2)

    q = PlisI ? w : q
    r₀ = PlisI ? w : q

    # Initial residual r₀.
    if warm_start
        mul!(w, A, Δx)
        axpby!(one(T), b, -one(T), w)
        restart && axpy!(one(T), Δx, x)
    else
        w .= b
    end
    PlisI || ldiv!(r₀, Pl, w)  # r₀ = Pl(b - Ax₀)
    β = _norm2(r₀)         # β = ‖r₀‖₂

    ε = abstol + reltol * β

    return SimpleGMRESCache{UBD}(
        memory, n, restart, maxiters, blocksize, ε, PlisI, PrisI,
        Pl, Pr, Δx, q, p, x, A, b, abstol, reltol, w, V, s, c, z, R, β, warm_start
    )
end

function SciMLBase.solve!(cache::SimpleGMRESCache, lincache::LinearCache)
    (; memory, restart, maxiters, ε, PlisI, PrisI, Pl, Pr) = cache
    (; Δx, q, p, x, A, b, w, V, s, c, z, R, β, warm_start) = cache
    layout = _layout(cache)

    T = eltype(x)
    q = PlisI ? w : q
    r₀ = PlisI ? w : q
    xr = restart ? Δx : x

    if β == 0
        return SciMLBase.build_linear_solution(
            lincache.alg, x, r₀, nothing;
            retcode = ReturnCode.Success
        )
    end

    rNorm = β
    npass = 0        # Number of pass

    iter = 0        # Cumulative number of iterations
    inner_iter = 0  # Number of iterations in a pass

    # Tolerance for breakdown detection.
    btol = eps(T)^(3 / 4)

    # Stopping criterion
    breakdown = false
    inconsistent = false
    solved = rNorm ≤ ε
    inner_maxiters = maxiters
    tired = iter ≥ maxiters
    inner_tired = inner_iter ≥ inner_maxiters
    status = ReturnCode.Default

    while !(solved || tired || breakdown)
        # Initialize workspace.
        # TODO: Check that not zeroing out (V, s, c, R, z) doesn't lead to incorrect results.
        nr = 0  # Number of coefficients stored in Rₖ.

        if restart
            xr .= zero(T)  # xr === Δx when restart is set to true
            if npass ≥ 1
                mul!(w, A, x)
                axpby!(one(T), b, -one(T), w)
                PlisI || ldiv!(r₀, Pl, w)
            end
        end

        # Initial ζ₁ and V₁
        _gmres_start!(layout, z, V, r₀)

        npass = npass + 1
        inner_iter = 0
        inner_tired = false

        while !(solved || inner_tired || breakdown)
            # Update iteration index
            inner_iter += 1
            # Update workspace if more storage is required and restart is set to false
            if !restart && (inner_iter > memory)
                _gmres_grow_pass!(layout, R, s, c, inner_iter, T)
            end

            # Continue the Arnoldi process.
            p = PrisI ? V[inner_iter] : p
            PrisI || ldiv!(p, Pr, V[inner_iter])  # p ← Nvₖ
            mul!(w, A, p)                         # w ← ANvₖ
            PlisI || ldiv!(q, Pl, w)                 # q ← MANvₖ
            for i in 1:inner_iter
                _gmres_orth_step!(layout, R, nr + i, V, i, q)
            end

            # Compute hₖ₊₁.ₖ
            Hbis = _gmres_hbis(layout, q)

            # Update the QR factorization of Hₖ₊₁.ₖ, applying the previous Givens
            # reflections Ωᵢ before computing the current one.
            for i in 1:(inner_iter - 1)
                _gmres_apply_givens!(layout, c, s, R, nr, i)
            end
            _gmres_new_givens!(layout, c, s, R, nr, inner_iter, Hbis)

            ζₖ₊₁ = _gmres_update_z!(layout, z, c, s, inner_iter)

            # Update residual norm estimate.
            rNorm = _gmres_rnorm(layout, ζₖ₊₁)

            # Update the number of coefficients in Rₖ
            nr = nr + inner_iter

            # Stopping conditions that do not depend on user input.
            # This is to guard against tolerances that are unreasonably small.
            resid_decrease_mach = (rNorm + one(T) ≤ one(T))

            # Update stopping criterion.
            resid_decrease_lim = rNorm ≤ ε
            breakdown = _gmres_maximum(layout, Hbis) ≤ btol
            solved = resid_decrease_lim || resid_decrease_mach
            inner_tired = restart ? inner_iter ≥ min(memory, inner_maxiters) :
                inner_iter ≥ inner_maxiters

            # Compute vₖ₊₁.
            if !(solved || inner_tired || breakdown)
                if !restart && (inner_iter ≥ memory)
                    _gmres_grow_krylov!(layout, V, z, T)
                end
                _gmres_next_v!(layout, V, z, q, Hbis, ζₖ₊₁, inner_iter)
            end
        end

        # Compute yₖ by solving Rₖyₖ = zₖ with backward substitution.
        y = z  # yᵢ = zᵢ
        for i in inner_iter:-1:1
            pos = nr + i - inner_iter      # position of rᵢ.ₖ
            for j in inner_iter:-1:(i + 1)
                _gmres_backsub_update!(layout, y, i, R, pos, j)
                pos = pos - j + 1            # position of rᵢ.ⱼ₋₁
            end
            inconsistent |= _gmres_backsub_solve!(layout, y, i, R, pos, btol, T)
        end

        # Form xₖ = NVₖyₖ
        for i in 1:inner_iter
            _gmres_add_correction!(layout, xr, V, y, i)
        end
        if !PrisI
            p .= xr
            ldiv!(xr, Pr, p)
        end
        restart && axpy!(one(T), xr, x)

        # Update inner_itmax, iter, tired and overtimed variables.
        inner_maxiters = inner_maxiters - inner_iter
        iter = iter + inner_iter
        tired = iter ≥ maxiters
    end

    # Termination status
    tired && (status = ReturnCode.MaxIters)
    solved && (status = ReturnCode.Success)
    inconsistent && (status = ReturnCode.Infeasible)

    # Update x
    warm_start && !restart && axpy!(one(T), Δx, x)
    cache.warm_start = false

    return SciMLBase.build_linear_solution(
        lincache.alg, x, rNorm, nothing;
        retcode = status, iters = iter
    )
end
