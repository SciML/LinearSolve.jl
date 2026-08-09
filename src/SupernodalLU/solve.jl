# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# Solve phase: supernodal forward/back substitution over the dense panels,
# plus iterative refinement (the Schenk–Gärtner recipe for recovering
# accuracy lost to static pivot perturbation).  Single-RHS solves run through gemv-shaped
# kernels on a factor-owned workspace; multi-RHS solves run the same sweeps
# with gemm-shaped kernels over a scratch matrix grown on demand — both are
# allocation-free after warmup.
#
# Index spaces: with V = M[qf,qf] and P·V = L·U, the forward solve runs in
# factor-row space (update rows via `rowsfac`), the back solve in column space
# (U12 columns are the un-permuted column ids in `rows`).

# Supernode panels are small — for a 2D Poisson factorization the pivot block
# width `np` has mean ~12 and median 6 — so the BLAS `trsv`/`gemv` calls that
# would serve them are dominated by call overhead rather than arithmetic: a
# 12x12 unit-triangular solve (~72 flops) measured ~260 ns through BLAS.  Below
# `PANEL_BLAS_CUTOFF` the sweeps therefore use the column-oriented kernels
# below, which are the same algorithms written out with `@inbounds`/`@simd`;
# above it BLAS wins and is used unchanged.  Measured on `_solve_panels!`:
# -43 % (n=400), -40 % (n=1600), -37 % (n=3600), -35 % (n=10000), taking a
# poisson-2D k=40 solve from 124 us to ~77 us (UMFPACK 71 us, PureKLU 66 us).
# BLAS thread count is irrelevant at these sizes (123.1 us at 1 thread vs
# 122.8 us at 64).  Results are bit-comparable to the BLAS path to ~2e-16.
const PANEL_BLAS_CUTOFF = 64

# x := L11 \ x for the unit-lower-triangular pivot block stored in W[1:np,1:np]
# (column-oriented forward substitution: subtract each solved entry's column).
@inline function _unit_lower_solve!(W::AbstractMatrix{Tv}, x::AbstractVector{Tv}, np::Int) where {Tv}
    @inbounds for j in 1:np
        xj = x[j]
        iszero(xj) && continue
        @simd for i in (j + 1):np
            x[i] = muladd(-W[i, j], xj, x[i])
        end
    end
    return nothing
end

# x := U11 \ x for the upper-triangular pivot block (column-oriented backward
# substitution; the diagonal carries U's pivots).
@inline function _upper_solve!(W::AbstractMatrix{Tv}, x::AbstractVector{Tv}, np::Int) where {Tv}
    @inbounds for j in np:-1:1
        xj = x[j] / W[j, j]
        x[j] = xj
        iszero(xj) && continue
        @simd for i in 1:(j - 1)
            x[i] = muladd(-W[i, j], xj, x[i])
        end
    end
    return nothing
end

# Multi-RHS pivot-block solves.  Three backends, picked by panel width.
#
# `LinearAlgebra.ldiv!` on a triangular wrapper is not one of them: it
# heap-allocates a fixed 64 B per call on Julia 1.11 for a matrix right-hand
# side (not for a vector, and not on 1.10 or 1.12), which breaks the
# allocation-free guarantee for every panel of every sweep.
#
#   nrhs == 1                   the in-tree column kernels at every panel width.
#   no TriangularSolve          kernels through `PANEL_KERNEL_MAX_NP`, then BLAS.
#   TriangularSolve available   TriangularSolve through `PANEL_BLAS_MIN_NP`, then
#                               BLAS (apart from the one-column case above).
#
# With RecursiveFactorization loaded, SciML/LinearSolve.jl#1172 measured
# TriangularSolve through panels of width 1792.  The no-extension fallback
# stays conservative at 256.
const PANEL_KERNEL_MAX_NP = 256
const PANEL_BLAS_MIN_NP = 1792

# `trsm!` needs a BLAS element type in a strided array; everything else stays on
# the kernels, which are generic and beat the stdlib fallback for those types.
@inline _panel_blas_eligible(::Type{Tv}) where {Tv} = Tv <: LinearAlgebra.BlasFloat

@inline function _unit_lower_solve!(W::AbstractMatrix{Tv}, X::AbstractMatrix{Tv}, np::Int) where {Tv}
    @inbounds for r in axes(X, 2)
        _unit_lower_solve!(W, view(X, :, r), np)
    end
    return nothing
end

@inline function _upper_solve!(W::AbstractMatrix{Tv}, X::AbstractMatrix{Tv}, np::Int) where {Tv}
    @inbounds for r in axes(X, 2)
        _upper_solve!(W, view(X, :, r), np)
    end
    return nothing
end

# `trsm!` for the BLAS element types; anything else keeps the kernels.
_panel_unit_lower_trsm!(Ws::AbstractMatrix{Tv}, Yb::AbstractMatrix{Tv}, np::Int) where {Tv} =
    _unit_lower_solve!(Ws, Yb, np)
function _panel_unit_lower_trsm!(
        Ws::StridedMatrix{Tv}, Yb::StridedMatrix{Tv}, np::Int
    ) where {Tv <: LinearAlgebra.BlasFloat}
    BLAS.trsm!('L', 'L', 'N', 'U', one(Tv), view(Ws, 1:np, 1:np), Yb)
    return nothing
end

_panel_upper_trsm!(Ws::AbstractMatrix{Tv}, Yb::AbstractMatrix{Tv}, np::Int) where {Tv} =
    _upper_solve!(Ws, Yb, np)
function _panel_upper_trsm!(
        Ws::StridedMatrix{Tv}, Yb::StridedMatrix{Tv}, np::Int
    ) where {Tv <: LinearAlgebra.BlasFloat}
    BLAS.trsm!('L', 'U', 'N', 'N', one(Tv), view(Ws, 1:np, 1:np), Yb)
    return nothing
end

# Overridable hooks for the solve-phase pivot-block trsms, the same arrangement
# the factorization phase uses for `_panel_rdiv!`/`_panel_ldiv!` in numeric.jl.
# They exist because nothing else can reach these calls: `TriangularSolve.ldiv!`
# is a distinct function from `LinearAlgebra.ldiv!` rather than an extension of
# it, so a bare `ldiv!` here can never dispatch to it however the environment is
# set up; and `defaultalg` selects algorithms for a `LinearProblem`, which a
# dense triangular sub-solve inside a sweep is not.  Routing has to be explicit.
#
# These defaults cover the no-TriangularSolve case: kernels, then BLAS.
# LinearSolveRecursiveFactorizationExt overrides both to slot TriangularSolve
# into the middle band.
function _panel_solve_unit_lower!(
        Ws::AbstractMatrix{Tv}, Yb::AbstractMatrix{Tv}, np::Int
    ) where {Tv}
    if np <= PANEL_KERNEL_MAX_NP || size(Yb, 2) == 1 || !_panel_blas_eligible(Tv)
        _unit_lower_solve!(Ws, Yb, np)
    else
        _panel_unit_lower_trsm!(Ws, Yb, np)
    end
    return nothing
end

function _panel_solve_upper!(
        Ws::AbstractMatrix{Tv}, Yb::AbstractMatrix{Tv}, np::Int
    ) where {Tv}
    if np <= PANEL_KERNEL_MAX_NP || size(Yb, 2) == 1 || !_panel_blas_eligible(Tv)
        _upper_solve!(Ws, Yb, np)
    else
        _panel_upper_trsm!(Ws, Yb, np)
    end
    return nothing
end

# t := A * x for the L21 block W[np+1:np+nu, 1:np] (column-oriented gemv).
@inline function _panel_gemv!(
        t::AbstractVector{Tv}, W::AbstractMatrix{Tv}, x::AbstractVector{Tv},
        np::Int, nu::Int
    ) where {Tv}
    @inbounds for k in 1:nu
        t[k] = zero(Tv)
    end
    @inbounds for j in 1:np
        xj = x[j]
        iszero(xj) && continue
        @simd for k in 1:nu
            t[k] = muladd(W[np + k, j], xj, t[k])
        end
    end
    return nothing
end

# x := x - Z * t for the U12 block (column-oriented gemv, accumulating into x).
@inline function _panel_gemv_sub!(
        x::AbstractVector{Tv}, Z::AbstractMatrix{Tv}, t::AbstractVector{Tv},
        np::Int, nu::Int
    ) where {Tv}
    @inbounds for k in 1:nu
        tk = t[k]
        iszero(tk) && continue
        @simd for i in 1:np
            x[i] = muladd(-Z[i, k], tk, x[i])
        end
    end
    return nothing
end

# Ensure the panel scratch can hold the widest update-row set times `nrhs`.
# Called once per solve so the sweeps below carry no growth branch and are
# provably allocation-free (see the AllocCheck test in test/qa/allocations.jl).
@inline function _ensure_panel_scratch!(F::SupernodalLUFactor, nrhs::Int)
    need = F.sym.maxnu * nrhs
    length(F.gbuf) < need && resize!(F.gbuf, need)
    return nothing
end

# y := U \ (L \ (P * y)) in permuted space; y enters as V-row-ordered rhs.
# Precondition: `_ensure_panel_scratch!(F, 1)` has been called.
function _solve_panels!(y::AbstractVector{Tv}, F::SupernodalLUFactor{Tv}) where {Tv}
    sym = F.sym
    sstart = sym.sstart
    nsuper = length(sstart) - 1
    buf = F.gbuf
    @inbounds for s in 1:nsuper                       # L c = y  (forward)
        c1 = sstart[s]
        c2 = sstart[s + 1] - 1
        np = c2 - c1 + 1
        Rf = F.rowsfac[s]
        nu = length(Rf)
        Ws = F.W[s]
        xb = view(y, c1:c2)
        if np < PANEL_BLAS_CUTOFF
            _unit_lower_solve!(Ws, xb, np)
        else
            ldiv!(UnitLowerTriangular(view(Ws, 1:np, 1:np)), xb)
        end
        if nu > 0
            t = view(buf, 1:nu)
            if np < PANEL_BLAS_CUTOFF
                _panel_gemv!(t, Ws, xb, np, nu)
            else
                mul!(t, view(Ws, (np + 1):(np + nu), 1:np), xb)
            end
            @simd for k in 1:nu
                y[Rf[k]] -= t[k]
            end
        end
    end
    @inbounds for s in nsuper:-1:1                    # U z = c  (backward)
        c1 = sstart[s]
        c2 = sstart[s + 1] - 1
        np = c2 - c1 + 1
        R = sym.rows[s]
        nu = length(R)
        Ws = F.W[s]
        xb = view(y, c1:c2)
        if nu > 0
            t = view(buf, 1:nu)
            @simd for k in 1:nu
                t[k] = y[R[k]]
            end
            if np < PANEL_BLAS_CUTOFF
                _panel_gemv_sub!(xb, F.Z[s], t, np, nu)
            else
                mul!(xb, F.Z[s], t, -one(Tv), one(Tv))
            end
        end
        if np < PANEL_BLAS_CUTOFF
            _upper_solve!(Ws, xb, np)
        else
            ldiv!(UpperTriangular(view(Ws, 1:np, 1:np)), xb)
        end
    end
    return y
end

# Multi-RHS variant: same sweeps with gemm-shaped updates on an n×nrhs block.
# Precondition: `_ensure_panel_scratch!(F, nrhs)` has been called.
function _solve_panels!(Y::AbstractMatrix{Tv}, F::SupernodalLUFactor{Tv}) where {Tv}
    sym = F.sym
    sstart = sym.sstart
    nsuper = length(sstart) - 1
    nrhs = size(Y, 2)
    buf = F.gbuf
    @inbounds for s in 1:nsuper                       # forward
        c1 = sstart[s]
        c2 = sstart[s + 1] - 1
        np = c2 - c1 + 1
        Rf = F.rowsfac[s]
        nu = length(Rf)
        Ws = F.W[s]
        Yb = view(Y, c1:c2, :)
        _panel_solve_unit_lower!(Ws, Yb, np)
        if nu > 0
            T = reshape(view(buf, 1:(nu * nrhs)), nu, nrhs)
            mul!(T, view(Ws, (np + 1):(np + nu), 1:np), Yb)
            for r in 1:nrhs, k in 1:nu
                Y[Rf[k], r] -= T[k, r]
            end
        end
    end
    @inbounds for s in nsuper:-1:1                    # backward
        c1 = sstart[s]
        c2 = sstart[s + 1] - 1
        np = c2 - c1 + 1
        R = sym.rows[s]
        nu = length(R)
        Ws = F.W[s]
        Yb = view(Y, c1:c2, :)
        if nu > 0
            T = reshape(view(buf, 1:(nu * nrhs)), nu, nrhs)
            for r in 1:nrhs, k in 1:nu
                T[k, r] = Y[R[k], r]
            end
            mul!(Yb, F.Z[s], T, -one(Tv), one(Tv))
        end
        _panel_solve_upper!(Ws, Yb, np)
    end
    return Y
end

# x .= A \ b through the factorization (no refinement).  With matching the
# factorized matrix is M = (Dr·A·Dc)[σ,:], so gather picks up σ and Dr, and
# the scatter applies Dc.  Safe when x aliases b (b is fully read first).
function _solve_once!(x::AbstractVector{Tv}, F::SupernodalLUFactor{Tv}, b::AbstractVector) where {Tv}
    n = F.sym.n
    y = F.work
    p = F.p
    qf = F.sym.qf
    rp = F.rowperm
    Rs = F.Rs
    Cs = F.Cs
    _ensure_panel_scratch!(F, 1)
    @inbounds for k in 1:n
        i = rp[p[k]]
        y[k] = Rs[i] * b[i]
    end
    _solve_panels!(y, F)
    @inbounds for j in 1:n
        jq = qf[j]
        x[jq] = Cs[jq] * y[j]
    end
    return x
end

# Grow-on-demand multi-RHS workspace (reallocates only when nrhs changes).
function _scratch_mat!(F::SupernodalLUFactor{Tv}, nrhs::Int) where {Tv}
    S = F.solve_scratch
    if size(S, 2) != nrhs
        S = Matrix{Tv}(undef, F.sym.n, nrhs)
        F.solve_scratch = S
    end
    return S
end

function _solve_once!(X::AbstractMatrix{Tv}, F::SupernodalLUFactor{Tv}, B::AbstractMatrix) where {Tv}
    n = F.sym.n
    nrhs = size(B, 2)
    Y = _scratch_mat!(F, nrhs)
    _ensure_panel_scratch!(F, nrhs)
    p = F.p
    qf = F.sym.qf
    rp = F.rowperm
    Rs = F.Rs
    Cs = F.Cs
    @inbounds for r in 1:nrhs, k in 1:n
        i = rp[p[k]]
        Y[k, r] = Rs[i] * B[i, r]
    end
    _solve_panels!(Y, F)
    @inbounds for r in 1:nrhs, j in 1:n
        jq = qf[j]
        X[jq, r] = Cs[jq] * Y[j, r]
    end
    return X
end

# Refinement exists to recover the accuracy lost to *perturbed* pivots.  MC64
# matching improves conditioning rather than degrading it, so a matched factor
# with no perturbed pivot needs none — refining there measured a 2.0-2.2x cost
# per solve for a residual change of ~1.5e-15 -> 7e-16.
_auto_refine(F::SupernodalLUFactor) = F.nperturbed > 0 ? 3 : 0

# Dispatch on the `refine` type rather than branching on its value: a value
# branch leaves `Int(refine::Symbol)` reachable, which is both a runtime
# dispatch and a MethodError instead of a usable message for a bad symbol.
_refine_steps(::SupernodalLUFactor, refine::Integer) = Int(refine)
function _refine_steps(F::SupernodalLUFactor, refine::Symbol)
    refine === :auto ||
        throw(ArgumentError("`refine` must be `:auto` or an integer number of steps"))
    return _auto_refine(F)
end

"""
    solve!(x, F::SupernodalLUFactor, b; refine=:auto) -> x
    solve(F::SupernodalLUFactor, b; refine=:auto) -> x

Solve `A x = b` (also accepts matrix right-hand sides).  `refine` is the
number of iterative-refinement steps; `:auto` refines (up to 3 steps,
stopping early on stagnation) whenever the factorization was numerically
delicate — static pivot perturbation occurred or MC64 matching preprocessing
was applied — and does 0 steps otherwise.  This is the accuracy-recovery
mechanism the Schenk–Gärtner method prescribes for restricted pivoting.  Allocation-free after
warmup.
"""
function solve!(
        x::AbstractVector{Tv}, F::SupernodalLUFactor{Tv}, b::AbstractVector;
        refine::Union{Symbol, Integer} = :auto
    ) where {Tv}
    nref = _refine_steps(F, refine)
    _solve_once!(x, F, b)
    if nref > 0
        r = F.ir_r
        dx = F.ir_dx
        prevn = Inf
        for _ in 1:nref
            copyto!(r, b)
            mul!(r, F.A, x, -one(Tv), one(Tv))       # r = b - A x
            rn = norm(r)
            (iszero(rn) || rn >= 0.5 * prevn) && break  # converged / stagnated
            prevn = rn
            _solve_once!(dx, F, r)
            x .+= dx
        end
    end
    return x
end

function solve!(
        X::AbstractMatrix{Tv}, F::SupernodalLUFactor{Tv}, B::AbstractMatrix;
        refine::Union{Symbol, Integer} = :auto
    ) where {Tv}
    size(X) == size(B) || throw(DimensionMismatch("X and B sizes differ"))
    nref = _refine_steps(F, refine)
    if nref == 0
        return _solve_once!(X, F, B)
    end
    for r in 1:size(B, 2)                            # refined: column-by-column
        solve!(view(X, :, r), F, view(B, :, r); refine = nref)
    end
    return X
end

function solve(F::SupernodalLUFactor{Tv}, b::AbstractVecOrMat; kwargs...) where {Tv}
    x = similar(b, promote_type(Tv, eltype(b)))
    return solve!(x, F, b; kwargs...)
end

function LinearAlgebra.ldiv!(
        x::AbstractVecOrMat, F::SupernodalLUFactor, b::AbstractVecOrMat
    )
    return solve!(x, F, b)
end

function LinearAlgebra.ldiv!(F::SupernodalLUFactor{Tv}, b::AbstractVector) where {Tv}
    copyto!(F.btmp, b)
    return solve!(b, F, F.btmp)
end

function LinearAlgebra.ldiv!(F::SupernodalLUFactor{Tv}, B::AbstractMatrix) where {Tv}
    for r in 1:size(B, 2)
        br = view(B, :, r)
        copyto!(F.btmp, br)
        solve!(br, F, F.btmp)
    end
    return B
end

Base.:\(F::SupernodalLUFactor, b::AbstractVecOrMat) = solve(F, b)
