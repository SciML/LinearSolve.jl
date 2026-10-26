# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# Solves.  Right-hand sides are processed as blocks (one column per RHS), so
# every per-front operation is a dense GEMM / triangular solve.
#
# B = A (the default, any shape): with the column singletons first,
#     A ~ [R11 R12; 0 A22],
# the singleton rows can be satisfied exactly for any x2, so the least-squares
# problem is A22 x2 ≈ b2 — x2[q] = R⁻¹ (Qᴴb2)[rows of R], Qᴴ applied front by
# front in postorder on the slot vector, R back-substituted in reverse — then
# x1 = R11⁻¹ (b1 - R12 x2) by substitution.  Dead columns get zero (Heath's
# basic solution, as SPQR).
#
# B = Aᴴ (`wide = :minnorm`; B[:, q] = QR, so A[q, :] = Rᴴ Qᴴ): with L the live part of Rᴴ
# (square lower triangular) and E the rows of Rᴴ at dead columns (equations of
# dependent rows of A), A[q, :] x = b[q] becomes, for x = Q y with y on the
# live rows of R,
#     L y ≈ c₁,   E y ≈ c₂.
# Full row rank (no E): y = L⁻¹ c₁, the minimum-norm solution.  Otherwise the
# least-squares y is, with G = E L⁻¹ and K = I + G Gᴴ (d × d, Cholesky at
# factorization time),
#     y = y₀ + L⁻¹ L⁻ᴴ Eᴴ K⁻¹ (c₂ - E y₀),   y₀ = L⁻¹ c₁,
# where c₂ - E y₀ is exactly what a forward substitution leaves behind at the
# dead columns.  Since x = Q y stays in the range of the live columns of Aᴴ,
# this is the minimum-norm least-squares solution.

function _ensure_rhs!(F::SupernodalQRFactor{Tv}, k::Int) where {Tv}
    size(F.C, 2) >= k && return F
    sym = F.sym
    F.W = Matrix{Tv}(undef, sym.m, k)
    F.C = Matrix{Tv}(undef, sym.nmf, k)
    F.T = Matrix{Tv}(undef, sym.nmf, k)
    F.Y = Matrix{Tv}(undef, F.maxmf, k)
    F.Z = Matrix{Tv}(undef, sym.maxnp, k)
    F.U = Matrix{Tv}(undef, sym.maxnu, k)
    return F
end

# Workspace views shaped like the right-hand-side block: vectors for a single
# right-hand side, so the per-front products are BLAS-2 (gemv/trsv) rather
# than one-column BLAS-3 calls.
_zview(F::SupernodalQRFactor, n::Int, ::AbstractVector) = view(F.Z, 1:n, 1)
_zview(F::SupernodalQRFactor, n::Int, C::AbstractMatrix) = view(F.Z, 1:n, 1:size(C, 2))
_uview(F::SupernodalQRFactor, n::Int, ::AbstractVector) = view(F.U, 1:n, 1)
_uview(F::SupernodalQRFactor, n::Int, C::AbstractMatrix) = view(F.U, 1:n, 1:size(C, 2))
_bview(M::Matrix, ::AbstractVector) = view(M, :, 1)
_bview(M::Matrix, B::AbstractMatrix) = view(M, :, 1:size(B, 2))

# Apply Q_sᴴ (adj) or Q_s to the slot rows of W for every front: postorder
# for Qᴴ = Q_Sᴴ ⋯ Q_1ᴴ, reverse postorder for Q = Q_1 ⋯ Q_S.
function _apply_q!(F::SupernodalQRFactor, W::AbstractVecOrMat, adj::Bool)
    ns = length(F.sym.sstart) - 1
    for ss in 1:ns
        _apply_front_q!(W, F, adj ? ss : ns - ss + 1, adj)
    end
    return W
end

function _apply_front_q!(W::AbstractVecOrMat, F::SupernodalQRFactor, s::Int, adj::Bool)
    nr = F.nrefl[s]
    nr == 0 && return W
    g = F.frows[s]
    mf = length(g)
    Fs = F.fronts[s]
    tau = F.tau[s]
    hc = F.hcol[s]
    ve = F.vend[s]
    k = size(W, 2)
    Y = view(F.Y, 1:mf, 1:k)
    @inbounds for c in 1:k, i in 1:mf
        Y[i, c] = W[g[i], c]
    end
    @inbounds for c in 1:k, kk in 1:nr
        q = adj ? kk : nr - kk + 1
        j = hc[q]
        τ = adj ? conj(tau[q]) : tau[q]
        iszero(τ) && continue
        last = ve[q]                 # the reflector's staircase end
        acc = Y[q, c]
        @simd for i in (q + 1):last
            acc += conj(Fs[i, j]) * Y[i, c]
        end
        acc *= τ
        Y[q, c] -= acc
        @simd for i in (q + 1):last
            Y[i, c] -= Fs[i, j] * acc
        end
    end
    @inbounds for c in 1:k, i in 1:mf
        W[g[i], c] = Y[i, c]
    end
    return W
end

# Live R rows read from / written to their slots, indexed by pivot column.
function _slots_to_cols!(C::AbstractVecOrMat, F::SupernodalQRFactor, W::AbstractVecOrMat)
    sym = F.sym
    @inbounds for s in 1:(length(sym.sstart) - 1)
        c1 = sym.sstart[s]
        g = F.frows[s]
        hc = F.hcol[s]
        for q in 1:F.npiv[s], c in axes(C, 2)
            C[c1 + hc[q] - 1, c] = W[g[q], c]
        end
    end
    return C
end

function _cols_to_slots!(W::AbstractVecOrMat, F::SupernodalQRFactor{Tv}, C::AbstractVecOrMat) where {Tv}
    sym = F.sym
    fill!(W, zero(Tv))
    @inbounds for s in 1:(length(sym.sstart) - 1)
        c1 = sym.sstart[s]
        g = F.frows[s]
        hc = F.hcol[s]
        for q in 1:F.npiv[s], c in axes(C, 2)
            W[g[q], c] = C[c1 + hc[q] - 1, c]
        end
    end
    return W
end

# R x = c in place: on entry C[pivot column of each live R row] holds its
# right-hand side; on exit C holds x (dead columns zero).
function _r_backward!(C::AbstractVecOrMat, F::SupernodalQRFactor{Tv}) where {Tv}
    sym = F.sym
    k = size(C, 2)
    @inbounds for s in (length(sym.sstart) - 1):-1:1
        c1 = sym.sstart[s]
        np = sym.sstart[s + 1] - c1
        npv = F.npiv[s]
        if npv == 0
            for c in 1:k, a in 1:np
                C[c1 + a - 1, c] = zero(Tv)
            end
            continue
        end
        hc = F.hcol[s]
        Z = _zview(F, npv, C)
        for c in 1:k, q in 1:npv
            Z[q, c] = C[c1 + hc[q] - 1, c]
        end
        Rp = F.Rp[s]
        if !isempty(Rp)
            # packed live rows: [R11 | R12] over live columns only
            rc = F.Rcols[s]
            nl = length(rc)
            if nl > 0
                Ub = _uview(F, nl, C)
                for c in 1:k, t in 1:nl
                    Ub[t, c] = C[rc[t], c]
                end
                mul!(Z, view(Rp, :, (npv + 1):(npv + nl)), Ub, -one(Tv), one(Tv))
            end
            ldiv!(UpperTriangular(view(Rp, :, 1:npv)), Z)
            for c in 1:k, a in 1:np
                C[c1 + a - 1, c] = zero(Tv)
            end
            for c in 1:k, q in 1:npv
                C[c1 + hc[q] - 1, c] = Z[q, c]
            end
            continue
        end
        U = sym.cols[s]
        nu = length(U)
        Fs = F.fronts[s]
        if nu > 0
            Ub = _uview(F, nu, C)
            for c in 1:k, t in 1:nu
                Ub[t, c] = C[U[t], c]
            end
            mul!(Z, view(Fs, 1:npv, (np + 1):(np + nu)), Ub, -one(Tv), one(Tv))
        end
        if npv == np
            ldiv!(UpperTriangular(view(Fs, 1:np, 1:np)), Z)
        else
            for q in npv:-1:1
                j = hc[q]
                piv = Fs[q, j]
                for c in 1:k
                    zq = Z[q, c] / piv
                    Z[q, c] = zq
                    for q2 in 1:(q - 1)
                        Z[q2, c] -= Fs[q2, j] * zq
                    end
                end
            end
        end
        for c in 1:k, a in 1:np
            C[c1 + a - 1, c] = zero(Tv)
        end
        for c in 1:k, q in 1:npv
            C[c1 + hc[q] - 1, c] = Z[q, c]
        end
    end
    return C
end

# Rᴴ y = c in place, column by column in postorder: on exit C[pivot column of
# each live R row] holds y, and C at each dead column holds what is left of
# its equation, c_dead - (E y).
function _rh_forward!(C::AbstractVecOrMat, F::SupernodalQRFactor{Tv}) where {Tv}
    sym = F.sym
    k = size(C, 2)
    @inbounds for s in 1:(length(sym.sstart) - 1)
        npv = F.npiv[s]
        npv == 0 && continue
        c1 = sym.sstart[s]
        np = sym.sstart[s + 1] - c1
        U = sym.cols[s]
        nu = length(U)
        Fs = F.fronts[s]
        hc = F.hcol[s]
        Z = _zview(F, npv, C)
        if npv == np
            for c in 1:k, a in 1:np
                Z[a, c] = C[c1 + a - 1, c]
            end
            ldiv!(UpperTriangular(view(Fs, 1:np, 1:np))', Z)
        else
            for q in 1:npv
                j = hc[q]
                piv = conj(Fs[q, j])
                for c in 1:k
                    zq = C[c1 + j - 1, c] / piv
                    Z[q, c] = zq
                    for t in (j + 1):np
                        C[c1 + t - 1, c] -= conj(Fs[q, t]) * zq
                    end
                end
            end
        end
        if nu > 0
            Ub = _uview(F, nu, C)
            mul!(Ub, view(Fs, 1:npv, (np + 1):(np + nu))', Z)
            for c in 1:k, t in 1:nu
                C[U[t], c] -= Ub[t, c]
            end
        end
        for c in 1:k, q in 1:npv
            C[c1 + hc[q] - 1, c] = Z[q, c]
        end
    end
    return C
end

# T[pivot column of live row r, :] += Σ_j R[r, dead[j]] V[j - j0 + 1, :] over
# dead columns with j0 ≤ j < j0 + size(V, 1)  (i.e. T += Eᴴ V, restricted to a
# block of dead columns).
function _add_eh!(T::AbstractVecOrMat, F::SupernodalQRFactor, V::AbstractMatrix, j0::Int)
    sym = F.sym
    deadidx = F.deadidx
    nv = size(V, 1)
    @inbounds for s in 1:(length(sym.sstart) - 1)
        c1 = sym.sstart[s]
        np = sym.sstart[s + 1] - c1
        U = sym.cols[s]
        nf = np + length(U)
        Fs = F.fronts[s]
        hc = F.hcol[s]
        for q in 1:F.npiv[s]
            row = c1 + hc[q] - 1
            for t in (hc[q] + 1):nf
                gc = t <= np ? c1 + t - 1 : U[t - np]
                jj = deadidx[gc] - j0 + 1
                (1 <= jj <= nv && deadidx[gc] > 0) || continue
                r = Fs[q, t]
                for c in axes(T, 2)
                    T[row, c] += r * V[jj, c]
                end
            end
        end
    end
    return T
end

# K = I + G Gᴴ, G = E L⁻¹, by blocks of dead columns: column j of G Gᴴ is
# G (L⁻ᴴ Eᴴ eⱼ) = -(what a forward solve of L⁻ᴴ Eᴴ eⱼ leaves at the dead
# columns).
function _deficiency_system(F::SupernodalQRFactor{Tv}) where {Tv}
    d = length(F.dead)
    d == 0 && return _empty_chol(Tv)
    K = Matrix{Tv}(I(d))
    nb = min(d, 64)
    _ensure_rhs!(F, nb)
    for j0 in 1:nb:d
        w = min(nb, d - j0 + 1)
        T = view(F.T, :, 1:w)
        fill!(T, zero(Tv))
        _add_eh!(T, F, Matrix{Tv}(I(w)), j0)
        _r_backward!(T, F)
        _rh_forward!(T, F)
        for c in 1:w, i in 1:d
            K[i, j0 + c - 1] -= T[F.dead[i], c]
        end
    end
    return LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(K))
end

function _check_dims(F::SupernodalQRFactor, x::AbstractVecOrMat, b::AbstractVecOrMat)
    m, n = size(F)
    size(b, 1) == m || throw(DimensionMismatch("b has $(size(b, 1)) rows, expected $m"))
    size(x, 1) == n || throw(DimensionMismatch("x has $(size(x, 1)) rows, expected $n"))
    size(x, 2) == size(b, 2) || throw(DimensionMismatch("x and b have different numbers of columns"))
    return nothing
end

function _solve_block!(X::AbstractVecOrMat, F::SupernodalQRFactor{Tv}, B::AbstractVecOrMat) where {Tv}
    sym = F.sym
    qf = sym.qf
    k = size(B, 2)
    _ensure_rhs!(F, k)
    W = _bview(F.W, B)
    C = _bview(F.C, B)
    if sym.transposed
        @inbounds for c in 1:k, i in eachindex(qf)
            C[i, c] = B[qf[i], c]
        end
        _rh_forward!(C, F)
        d = length(F.dead)
        if d > 0
            V = Matrix{Tv}(undef, d, k)
            @inbounds for c in 1:k, i in 1:d
                V[i, c] = C[F.dead[i], c]
            end
            ldiv!(F.Kfac, V)
            T = _bview(F.T, B)
            fill!(T, zero(Tv))
            _add_eh!(T, F, V, 1)
            _r_backward!(T, F)
            _rh_forward!(T, F)
            C .+= T
        end
        _cols_to_slots!(W, F, C)
        _apply_q!(F, W, false)
        copyto!(X, W)
    else
        copyto!(W, B)
        _apply_q!(F, W, true)
        _slots_to_cols!(C, F, W)
        _r_backward!(C, F)
        @inbounds for c in 1:k
            for i in eachindex(qf)
                X[qf[i], c] = C[i, c]
            end
            for j in sym.ecol
                X[j, c] = zero(Tv)
            end
        end
        _singleton_solve!(X, F, B)
    end
    return X
end

# x1 = R11⁻¹ (b1 - R12 x2) by substitution over the compacted singleton rows,
# in reverse discovery order (R11 is upper triangular in that order).
function _singleton_solve!(X::AbstractVecOrMat, F::SupernodalQRFactor, B::AbstractVecOrMat)
    sym = F.sym
    nsing = length(sym.scol)
    nsing == 0 && return X
    sptr = F.sptr
    scolx = F.scolx
    sval = F.sval
    @inbounds for c in axes(B, 2), l in nsing:-1:1
        acc = B[sym.srow[l], c]
        for p in sptr[l]:(sptr[l + 1] - 1)
            acc -= sval[p] * X[scolx[p], c]
        end
        X[sym.scol[l], c] = acc / F.spiv[l]
    end
    return X
end

"""
    solve!(x, F::SupernodalQRFactor, b) -> x

Least-squares solution of `A x ≈ b`: a basic solution (as SPQR) when `A` is
wide or rank deficient, or the minimum-norm least-squares solution for a wide
`A` factored with `wide = :minnorm`.  `b` and `x` may be vectors or matrices.
"""
function solve!(x::AbstractVecOrMat, F::SupernodalQRFactor, b::AbstractVecOrMat)
    _check_dims(F, x, b)
    return _solve_block!(x, F, b)
end

function Base.:\(F::SupernodalQRFactor{Tv}, b::AbstractVector) where {Tv}
    T = promote_type(Tv, eltype(b))
    return solve!(Vector{T}(undef, size(F, 2)), F, b)
end

function Base.:\(F::SupernodalQRFactor{Tv}, B::AbstractMatrix) where {Tv}
    T = promote_type(Tv, eltype(B))
    return solve!(Matrix{T}(undef, size(F, 2), size(B, 2)), F, B)
end

LinearAlgebra.ldiv!(x::AbstractVecOrMat, F::SupernodalQRFactor, b::AbstractVecOrMat) =
    solve!(x, F, b)
