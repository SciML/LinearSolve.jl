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

# y := U \ (L \ (P * y)) in permuted space; y enters as V-row-ordered rhs.
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
        ldiv!(UnitLowerTriangular(view(Ws, 1:np, 1:np)), xb)
        if nu > 0
            length(buf) < nu && resize!(buf, max(nu, 2 * length(buf)))
            t = view(buf, 1:nu)
            mul!(t, view(Ws, (np + 1):(np + nu), 1:np), xb)
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
            length(buf) < nu && resize!(buf, max(nu, 2 * length(buf)))
            t = view(buf, 1:nu)
            @simd for k in 1:nu
                t[k] = y[R[k]]
            end
            mul!(xb, F.Z[s], t, -one(Tv), one(Tv))
        end
        ldiv!(UpperTriangular(view(Ws, 1:np, 1:np)), xb)
    end
    return y
end

# Multi-RHS variant: same sweeps with gemm-shaped updates on an n×nrhs block.
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
        ldiv!(UnitLowerTriangular(view(Ws, 1:np, 1:np)), Yb)
        if nu > 0
            length(buf) < nu * nrhs && resize!(buf, max(nu * nrhs, 2 * length(buf)))
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
            length(buf) < nu * nrhs && resize!(buf, max(nu * nrhs, 2 * length(buf)))
            T = reshape(view(buf, 1:(nu * nrhs)), nu, nrhs)
            for r in 1:nrhs, k in 1:nu
                T[k, r] = Y[R[k], r]
            end
            mul!(Yb, F.Z[s], T, -one(Tv), one(Tv))
        end
        ldiv!(UpperTriangular(view(Ws, 1:np, 1:np)), Yb)
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

_auto_refine(F::SupernodalLUFactor) = (F.nperturbed > 0 || F.matched) ? 3 : 0

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
    nref = refine === :auto ? _auto_refine(F) : Int(refine)
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
    nref = refine === :auto ? _auto_refine(F) : Int(refine)
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
