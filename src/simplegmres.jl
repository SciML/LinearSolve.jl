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

    function SimpleGMRES{UBD}(; restart::Bool = true, blocksize::Int = 0,
            warm_start::Bool = false, memory::Int = 20) where {UBD}
        UBD && @assert blocksize > 0
        return new{UBD}(restart, memory, blocksize, warm_start)
    end

    function SimpleGMRES(; restart::Bool = true, blocksize::Int = 0,
            warm_start::Bool = false, memory::Int = 20)
        return SimpleGMRES{blocksize > 0}(; restart, memory, blocksize,
            warm_start)
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

function update_cacheval!(cache::LinearCache, cacheval::SimpleGMRESCache, name::Symbol, x)
    (name != :b || cache.isfresh) && return cacheval
    vec(cacheval.w) .= vec(x)
    fill!(cacheval.x, 0)
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

function SciMLBase.solve!(cache::LinearCache, alg::SimpleGMRES; kwargs...)
    if cache.isfresh
        solver = init_cacheval(alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol, cache.verbose,
            cache.assumptions; zeroinit = false)
        cache.cacheval = solver
        cache.isfresh = false
    end
    return SciMLBase.solve!(cache.cacheval, cache)
end

function init_cacheval(alg::SimpleGMRES{UDB}, args...; kwargs...) where {UDB}
    return _init_cacheval(Val(UDB), alg, args...; kwargs...)
end

function _init_cacheval(::Val{false}, alg::SimpleGMRES, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, ::Bool, ::OperatorAssumptions; zeroinit = true, kwargs...)
    @unpack memory, restart, blocksize, warm_start = alg

    if zeroinit
        return SimpleGMRESCache{false}(memory, 0, restart, maxiters, blocksize,
            zero(eltype(u)) * reltol + abstol, false, false, Pl, Pr, similar(u, 0),
            similar(u, 0), similar(u, 0), u, A, b, abstol, reltol, similar(u, 0),
            Vector{typeof(u)}(undef, 0), Vector{eltype(u)}(undef, 0),
            Vector{eltype(u)}(undef, 0), Vector{eltype(u)}(undef, 0),
            Vector{eltype(u)}(undef, 0), zero(eltype(u)), warm_start)
    end

    T = eltype(u)
    n = LinearAlgebra.checksquare(A)
    @assert n==length(b) "The size of `A` and `b` must match."
    memory = min(memory, maxiters)

    PlisI = _no_preconditioner(Pl)
    PrisI = _no_preconditioner(Pr)

    Δx = restart ? similar(u, n) : similar(u, 0)
    q = PlisI ? similar(u, 0) : similar(u, n)
    p = PrisI ? similar(u, 0) : similar(u, n)
    x = u
    x .= zero(T)

    w = similar(u, n)
    V = [similar(u) for _ in 1:memory]
    s = Vector{eltype(x)}(undef, memory)
    c = Vector{eltype(x)}(undef, memory)

    z = Vector{eltype(x)}(undef, memory)
    R = Vector{eltype(x)}(undef, (memory * (memory + 1)) ÷ 2)

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
    PlisI || mul!(r₀, Pl, w)  # r₀ = Pl(b - Ax₀)
    β = _norm2(r₀)         # β = ‖r₀‖₂

    rNorm = β
    ε = abstol + reltol * rNorm

    return SimpleGMRESCache{false}(
        memory, n, restart, maxiters, blocksize, ε, PlisI, PrisI,
        Pl, Pr, Δx, q, p, x, A, b, abstol, reltol, w, V, s, c, z, R, β, warm_start)
end

function SciMLBase.solve!(cache::SimpleGMRESCache{false}, lincache::LinearCache)
    @unpack memory, n, restart, maxiters, blocksize, ε, PlisI, PrisI, Pl, Pr = cache
    @unpack Δx, q, p, x, A, b, abstol, reltol, w, V, s, c, z, R, β, warm_start = cache

    T = eltype(x)
    q = PlisI ? w : q
    r₀ = PlisI ? w : q
    xr = restart ? Δx : x

    if β == 0
        return SciMLBase.build_linear_solution(lincache.alg, x, r₀, lincache;
            retcode = ReturnCode.Success)
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
        β = _norm2(r₀)
        z[1] = β
        V[1] .= r₀ / β

        npass = npass + 1
        inner_iter = 0
        inner_tired = false

        while !(solved || inner_tired || breakdown)
            # Update iteration index
            inner_iter += 1
            # Update workspace if more storage is required and restart is set to false
            if !restart && (inner_iter > memory)
                append!(R, zeros(T, inner_iter))
                push!(s, zero(T))
                push!(c, zero(T))
            end

            # Continue the Arnoldi process.
            p = PrisI ? V[inner_iter] : p
            PrisI || ldiv!(p, Pr, V[inner_iter])  # p ← Nvₖ
            mul!(w, A, p)                         # w ← ANvₖ
            PlisI || ldiv!(q, Pl, w)                 # q ← MANvₖ
            for i in 1:inner_iter
                R[nr + i] = dot(V[i], q)       # hᵢₖ = (vᵢ)ᴴq
                axpy!(-R[nr + i], V[i], q)     # q ← q - hᵢₖvᵢ
            end

            # Compute hₖ₊₁.ₖ
            Hbis = _norm2(q)  # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂

            # Update the QR factorization of Hₖ₊₁.ₖ.
            # Apply previous Givens reflections Ωᵢ.
            # [cᵢ  sᵢ] [ r̄ᵢ.ₖ ] = [ rᵢ.ₖ ]
            # [s̄ᵢ -cᵢ] [rᵢ₊₁.ₖ]   [r̄ᵢ₊₁.ₖ]
            for i in 1:(inner_iter - 1)
                Rtmp = c[i] * R[nr + i] + s[i] * R[nr + i + 1]
                R[nr + i + 1] = conj(s[i]) * R[nr + i] - c[i] * R[nr + i + 1]
                R[nr + i] = Rtmp
            end

            # Compute and apply current Givens reflection Ωₖ.
            # [cₖ  sₖ] [ r̄ₖ.ₖ ] = [rₖ.ₖ]
            # [s̄ₖ -cₖ] [hₖ₊₁.ₖ]   [ 0  ]
            (c[inner_iter], s[inner_iter], R[nr + inner_iter]) = _sym_givens(
                R[nr + inner_iter],
                Hbis)

            # Update zₖ = (Qₖ)ᴴβe₁
            ζₖ₊₁ = conj(s[inner_iter]) * z[inner_iter]
            z[inner_iter] = c[inner_iter] * z[inner_iter]

            # Update residual norm estimate.
            # ‖ Pl(b - Axₖ) ‖₂ = |ζₖ₊₁|
            rNorm = abs(ζₖ₊₁)

            # Update the number of coefficients in Rₖ
            nr = nr + inner_iter

            # Stopping conditions that do not depend on user input.
            # This is to guard against tolerances that are unreasonably small.
            resid_decrease_mach = (rNorm + one(T) ≤ one(T))

            # Update stopping criterion.
            resid_decrease_lim = rNorm ≤ ε
            breakdown = Hbis ≤ btol
            solved = resid_decrease_lim || resid_decrease_mach
            inner_tired = restart ? inner_iter ≥ min(memory, inner_maxiters) :
                          inner_iter ≥ inner_maxiters

            # Compute vₖ₊₁.
            if !(solved || inner_tired || breakdown)
                if !restart && (inner_iter ≥ memory)
                    push!(V, similar(first(V)))
                    push!(z, zero(T))
                end
                @. V[inner_iter + 1] = q / Hbis  # hₖ₊₁.ₖvₖ₊₁ = q
                z[inner_iter + 1] = ζₖ₊₁
            end
        end

        # Compute yₖ by solving Rₖyₖ = zₖ with backward substitution.
        y = z  # yᵢ = zᵢ
        for i in inner_iter:-1:1
            pos = nr + i - inner_iter      # position of rᵢ.ₖ
            for j in inner_iter:-1:(i + 1)
                y[i] = y[i] - R[pos] * y[j]  # yᵢ ← yᵢ - rᵢⱼyⱼ
                pos = pos - j + 1            # position of rᵢ.ⱼ₋₁
            end
            # Rₖ can be singular if the system is inconsistent
            if abs(R[pos]) ≤ btol
                y[i] = zero(T)
                inconsistent = true
            else
                y[i] = y[i] / R[pos]  # yᵢ ← yᵢ / rᵢᵢ
            end
        end

        # Form xₖ = NVₖyₖ
        for i in 1:inner_iter
            axpy!(y[i], V[i], xr)
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

    return SciMLBase.build_linear_solution(lincache.alg, x, rNorm, lincache;
        retcode = status, iters = iter)
end

function _init_cacheval(::Val{true}, alg::SimpleGMRES, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, ::Bool, ::OperatorAssumptions; zeroinit = true,
        blocksize = alg.blocksize)
    @unpack memory, restart, warm_start = alg

    if zeroinit
        return SimpleGMRESCache{true}(memory, 0, restart, maxiters, blocksize,
            zero(eltype(u)) * reltol + abstol, false, false, Pl, Pr, similar(u, 0),
            similar(u, 0), similar(u, 0), u, A, b, abstol, reltol, similar(u, 0),
            [u], [u], [u], [u], [u], zero(eltype(u)), warm_start)
    end

    T = eltype(u)
    n = LinearAlgebra.checksquare(A)
    @assert mod(n, blocksize)==0 "The blocksize must divide the size of the matrix."
    @assert n==length(b) "The size of `A` and `b` must match."
    memory = min(memory, maxiters)
    bsize = n ÷ blocksize

    PlisI = _no_preconditioner(Pl)
    PrisI = _no_preconditioner(Pr)

    Δx = restart ? similar(u, n) : similar(u, 0)
    q = PlisI ? similar(u, 0) : similar(u, n)
    p = PrisI ? similar(u, 0) : similar(u, n)
    x = u
    x .= zero(T)

    w = similar(u, n)
    V = [similar(u) for _ in 1:memory]
    s = [similar(u, bsize) for _ in 1:memory]
    c = [similar(u, bsize) for _ in 1:memory]

    z = [similar(u, bsize) for _ in 1:memory]
    R = [similar(u, bsize) for _ in 1:((memory * (memory + 1)) ÷ 2)]

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

    rNorm = β
    ε = abstol + reltol * rNorm

    return SimpleGMRESCache{true}(memory, n, restart, maxiters, blocksize, ε, PlisI, PrisI,
        Pl, Pr, Δx, q, p, x, A, b, abstol, reltol, w, V, s, c, z, R, β, warm_start)
end

function SciMLBase.solve!(cache::SimpleGMRESCache{true}, lincache::LinearCache)
    @unpack memory, n, restart, maxiters, blocksize, ε, PlisI, PrisI, Pl, Pr = cache
    @unpack Δx, q, p, x, A, b, abstol, reltol, w, V, s, c, z, R, β, warm_start = cache
    bsize = n ÷ blocksize

    __batch = Base.Fix2(reshape, (blocksize, bsize))

    T = eltype(x)
    q = PlisI ? w : q
    r₀ = PlisI ? w : q
    xr = restart ? Δx : x

    if β == 0
        return SciMLBase.build_linear_solution(lincache.alg, x, r₀, lincache;
            retcode = ReturnCode.Success)
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
        β = _norm2(__batch(r₀), 1)
        z[1] .= vec(β)
        V[1] .= vec(__batch(r₀) ./ β)

        npass = npass + 1
        inner_iter = 0
        inner_tired = false

        while !(solved || inner_tired || breakdown)
            # Update iteration index
            inner_iter += 1
            # Update workspace if more storage is required and restart is set to false
            if !restart && (inner_iter > memory)
                append!(R, [similar(first(R), bsize) for _ in 1:inner_iter])
                push!(s, similar(first(s), bsize))
                push!(c, similar(first(c), bsize))
            end

            # Continue the Arnoldi process.
            p = PrisI ? V[inner_iter] : p
            PrisI || ldiv!(p, Pr, V[inner_iter])  # p ← Nvₖ
            mul!(w, A, p)                         # w ← ANvₖ
            PlisI || ldiv!(q, Pl, w)                 # q ← MANvₖ
            for i in 1:inner_iter
                sum!(R[nr + i]', __batch(V[i]) .* __batch(q))
                q .-= vec(R[nr + i]' .* __batch(V[i])) # q ← q - hᵢₖvᵢ
            end

            # Compute hₖ₊₁.ₖ
            Hbis = vec(_norm2(__batch(q), 1))  # hₖ₊₁.ₖ = ‖vₖ₊₁‖₂

            # Update the QR factorization of Hₖ₊₁.ₖ.
            # Apply previous Givens reflections Ωᵢ.
            # [cᵢ  sᵢ] [ r̄ᵢ.ₖ ] = [ rᵢ.ₖ ]
            # [s̄ᵢ -cᵢ] [rᵢ₊₁.ₖ]   [r̄ᵢ₊₁.ₖ]
            for i in 1:(inner_iter - 1)
                Rtmp = c[i] .* R[nr + i] .+ s[i] .* R[nr + i + 1]
                R[nr + i + 1] .= conj.(s[i]) .* R[nr + i] .- c[i] .* R[nr + i + 1]
                R[nr + i] .= Rtmp
            end

            # Compute and apply current Givens reflection Ωₖ.
            # [cₖ  sₖ] [ r̄ₖ.ₖ ] = [rₖ.ₖ]
            # [s̄ₖ -cₖ] [hₖ₊₁.ₖ]   [ 0  ]
            _sym_givens!(c, s, R, nr, inner_iter, bsize, Hbis)

            # Update zₖ = (Qₖ)ᴴβe₁
            ζₖ₊₁ = conj.(s[inner_iter]) .* z[inner_iter]
            z[inner_iter] .= c[inner_iter] .* z[inner_iter]

            # Update residual norm estimate.
            # ‖ Pl(b - Axₖ) ‖₂ = |ζₖ₊₁|
            rNorm = maximum(abs, ζₖ₊₁)

            # Update the number of coefficients in Rₖ
            nr = nr + inner_iter

            # Stopping conditions that do not depend on user input.
            # This is to guard against tolerances that are unreasonably small.
            resid_decrease_mach = (rNorm + one(T) ≤ one(T))

            # Update stopping criterion.
            resid_decrease_lim = rNorm ≤ ε
            breakdown = maximum(Hbis) ≤ btol
            solved = resid_decrease_lim || resid_decrease_mach
            inner_tired = restart ? inner_iter ≥ min(memory, inner_maxiters) :
                          inner_iter ≥ inner_maxiters

            # Compute vₖ₊₁.
            if !(solved || inner_tired || breakdown)
                if !restart && (inner_iter ≥ memory)
                    push!(V, similar(first(V)))
                    push!(z, similar(first(z), bsize))
                end
                V[inner_iter + 1] .= vec(__batch(q) ./ Hbis')  # hₖ₊₁.ₖvₖ₊₁ = q
                z[inner_iter + 1] .= ζₖ₊₁
            end
        end

        # Compute yₖ by solving Rₖyₖ = zₖ with backward substitution.
        y = z  # yᵢ = zᵢ
        for i in inner_iter:-1:1
            pos = nr + i - inner_iter      # position of rᵢ.ₖ
            for j in inner_iter:-1:(i + 1)
                y[i] .= y[i] .- R[pos] .* y[j]  # yᵢ ← yᵢ - rᵢⱼyⱼ
                pos = pos - j + 1            # position of rᵢ.ⱼ₋₁
            end
            # Rₖ can be singular if the system is inconsistent
            y[i] .= ifelse.(abs.(R[pos]) .≤ btol, zero(T), y[i] ./ R[pos])  # yᵢ ← yᵢ / rᵢᵢ
            inconsistent = any(abs.(R[pos]) .≤ btol)
        end

        # Form xₖ = NVₖyₖ
        for i in 1:inner_iter
            xr .+= vec(__batch(V[i]) .* y[i]')
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

    return SciMLBase.build_linear_solution(lincache.alg, x, rNorm, lincache;
        retcode = status, iters = iter)
end
