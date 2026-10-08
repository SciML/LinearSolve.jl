# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# Numeric multifrontal QR.  Fronts are processed in assembly-tree postorder
# (children before parents).  Front s is a dense matrix over its pivot
# columns followed by `cols[s]`; its rows are the rows of B assigned to it
# plus the rows of each child's contribution block, read in place from the
# child's factored front.  The front is triangularized by Householder QR:
# the first rows of the result are rows of R, the next (at most |cols[s]|)
# rows form the upper-trapezoidal contribution block passed to the parent,
# and the remaining rows are zero.
#
# Every row of every front is tagged with a "slot" — the row of B it started
# out as (contribution rows inherit the slot of the child row they came
# from).  The orthogonal transformation of front s acts on exactly its slots,
# which is how Qᴴb and Qz are applied front by front at solve time without
# ever forming Q.
#
# Rank deficiency follows Heath (1982), as in Davis (2011): a pivot column
# whose remaining 2-norm is at most `tol` is declared dead — dropped, its
# solution component set to zero — and does not consume a row.  Fronts are
# factored in panels of `_NB` columns: each panel is factored by LAPACK
# `geqrf!` (BLAS element types) and accepted iff none of its pivot columns
# fell below `tol` — Householder's |Rⱼⱼ| is exactly the remaining column norm
# Heath tests, so the decision is identical — otherwise the panel is restored
# and refactored column by column with the dead-column rule.  The panel's
# reflectors then update the trailing columns as one compact-WY block
# reflector (three GEMMs), so rank-deficient fronts stay BLAS-3 too.

const _NB = 32

"""
    SupernodalQRFactor

Multifrontal sparse QR factorization of `B = A[:, q]` (or of `Aᴴ` when
`sym.transposed`), produced by [`snqr`](@ref).  Front `s` stores its
factored dense front in `fronts[s]`: Householder vectors below the
diagonal, rows of R on and above it.  Reflector `k` of front `s` (scalar
`tau[s][k]`, LAPACK convention `H = I - τvvᴴ`) eliminates front column
`hcol[s][k]` below row `k`; the first `npiv[s]` reflectors belong to live
pivot columns, the next `nrefl[s] - npiv[s]` to the contribution block.

For a rank-deficient factorization of `Aᴴ`, the dead columns of `B` are rows
of `A` whose equations the triangular solve cannot enforce; `dead` lists
them and `Kfac` is the Cholesky factorization of the small `d × d` system
that turns the solve into the minimum-norm least-squares solution (see
solve.jl).  Column singletons (`sym.srow`/`sym.scol`) are not stored:
their rows of R are rows of `A`, read back at solve time.
"""
mutable struct SupernodalQRFactor{Tv, Tr <: Real, Ti <: Integer}
    sym::QRSymbolic
    A::SparseMatrixCSC{Tv, Ti}     # the user's matrix (m × n)
    fronts::Vector{Matrix{Tv}}
    tau::Vector{Vector{Tv}}
    hcol::Vector{Vector{Int}}
    vend::Vector{Vector{Int}}      # last row reflector k's vector can reach (staircase)
    frows::Vector{Vector{Int}}     # slot of each front row
    npiv::Vector{Int}
    nrefl::Vector{Int}
    rank::Int
    tol::Tr                        # dead-column threshold of the last factorization
    usertol::Tr                    # requested threshold; negative = SPQR default rule
    relmap::Vector{Int}            # multifrontal column -> front column
    dead::Vector{Int}              # dead multifrontal columns
    deadidx::Vector{Int}           # multifrontal column -> index into `dead`, 0 if live
    Kfac::LinearAlgebra.Cholesky{Tv, Matrix{Tv}}
    # R rows of fronts that touch dead columns, packed to their live columns
    # for the back substitution (dead columns have x = 0): Rp[s] is
    # npiv × (npiv + |Rcols[s]|), live pivot block first (upper triangular),
    # then the live update columns Rcols[s]; empty when front s has no dead
    # columns and is solved from the front directly
    Rp::Vector{Matrix{Tv}}
    Rcols::Vector{Vector{Int}}
    # singleton rows of R, compacted at factorization time: row l has pivot
    # spiv[l] and off-pivot entries sval[p] in columns scolx[p] of x,
    # p ∈ sptr[l]:sptr[l+1]-1
    sptr::Vector{Int}
    scolx::Vector{Int}
    sval::Vector{Tv}
    spiv::Vector{Tv}
    # assembly workspaces (staircase row sort)
    rkey::Vector{Int}
    stair::Vector{Int}
    scur::Vector{Int}
    # solve workspaces, (re)sized for the widest right-hand-side block seen
    W::Matrix{Tv}                  # slot space (rows of B)
    C::Matrix{Tv}                  # multifrontal column space
    T::Matrix{Tv}                  # multifrontal column space, rank-deficiency correction
    Y::Matrix{Tv}                  # front gather
    Z::Matrix{Tv}                  # pivot block
    U::Matrix{Tv}                  # update columns
    maxmf::Int
end

LinearAlgebra.rank(F::SupernodalQRFactor) = F.rank

# Value of the p-th entry of B's row-wise access.
@inline function _bval(F::SupernodalQRFactor, nzA, p::Int)
    v = @inbounds nzA[F.sym.bpos[p]]
    return F.sym.transposed ? conj(v) : v
end

# Dead-column tolerance: an explicit `usertol` ≥ 0, else the SPQR rule.
function _set_tol!(F::SupernodalQRFactor{Tv, Tr}) where {Tv, Tr}
    F.tol = F.usertol >= 0 ? F.usertol : Tr(_default_tol(F.A, F.sym.transposed))
    return F.tol
end

# Fill front s: zero it, then scatter the children's contribution rows and
# the rows of B assigned to s, sorted by leftmost front column (counting sort)
# so the front is in staircase form: column j is nonzero only in rows
# 1:stair[j].  Householder steps preserve that pattern exactly, so panels,
# their trailing updates and the stored reflectors stop at the staircase.
# (Re)allocates the front when its numeric height differs from the stored one
# (dead columns in a descendant change contribution-block heights).
function _assemble!(F::SupernodalQRFactor{Tv}, s::Int) where {Tv}
    sym = F.sym
    c1 = sym.sstart[s]
    np = sym.sstart[s + 1] - c1
    U = sym.cols[s]
    nu = length(U)
    nf = np + nu
    relmap = F.relmap
    @inbounds for a in 1:np
        relmap[c1 + a - 1] = a
    end
    @inbounds for t in 1:nu
        relmap[U[t]] = np + t
    end
    mf = sym.frp[s + 1] - sym.frp[s]
    @inbounds for kk in sym.chp[s]:(sym.chp[s + 1] - 1)
        d = sym.chld[kk]
        mf += F.nrefl[d] - F.npiv[d]
    end
    # leftmost front column of every incoming row, in arrival order
    key = F.rkey
    length(key) < mf && resize!(key, mf)
    idx = 0
    @inbounds for kk in sym.chp[s]:(sym.chp[s + 1] - 1)
        d = sym.chld[kk]
        hd = F.hcol[d]
        npd = sym.sstart[d + 1] - sym.sstart[d]
        Ud = sym.cols[d]
        for k in (F.npiv[d] + 1):F.nrefl[d]
            idx += 1
            key[idx] = relmap[Ud[hd[k] - npd]]
        end
    end
    @inbounds for r in sym.frp[s]:(sym.frp[s + 1] - 1)
        i = sym.fri[r]
        lm = nf
        for p in sym.brp[i]:(sym.brp[i + 1] - 1)
            lm = min(lm, relmap[sym.bcol[p]])
        end
        idx += 1
        key[idx] = lm
    end
    # counting sort: stair[j] = #rows with leftmost column ≤ j, scur = slots
    stair = F.stair
    length(stair) < nf && resize!(stair, nf)
    cur = F.scur
    length(cur) < nf && resize!(cur, nf)
    @inbounds for j in 1:nf
        stair[j] = 0
    end
    @inbounds for t in 1:mf
        stair[key[t]] += 1
    end
    acc = 0
    @inbounds for j in 1:nf
        cur[j] = acc + 1
        acc += stair[j]
        stair[j] = acc
    end
    Fs = F.fronts[s]
    if size(Fs) != (mf, nf)
        Fs = Matrix{Tv}(undef, mf, nf)
        F.fronts[s] = Fs
    end
    fill!(Fs, zero(Tv))
    g = F.frows[s]
    resize!(g, mf)
    idx = 0
    @inbounds for kk in sym.chp[s]:(sym.chp[s + 1] - 1)
        d = sym.chld[kk]
        Fd = F.fronts[d]
        hd = F.hcol[d]
        gd = F.frows[d]
        npd = sym.sstart[d + 1] - sym.sstart[d]
        Ud = sym.cols[d]
        nfd = npd + length(Ud)
        for k in (F.npiv[d] + 1):F.nrefl[d]
            idx += 1
            row = cur[key[idx]]
            cur[key[idx]] = row + 1
            g[row] = gd[k]
            for t in hd[k]:nfd
                Fs[row, relmap[Ud[t - npd]]] = Fd[k, t]
            end
        end
    end
    nzA = nonzeros(F.A)
    @inbounds for r in sym.frp[s]:(sym.frp[s + 1] - 1)
        i = sym.fri[r]
        idx += 1
        row = cur[key[idx]]
        cur[key[idx]] = row + 1
        g[row] = i
        for p in sym.brp[i]:(sym.brp[i + 1] - 1)
            Fs[row, relmap[sym.bcol[p]]] = _bval(F, nzA, p)
        end
    end
    return Fs
end

# Householder reflector annihilating Fs[r+1:mf, j], given nrm = ‖Fs[r:mf, j]‖
# (LinearAlgebra.reflector! convention: returns τ, stores β in Fs[r, j] and
# v[2:end] below it, v[1] = 1 implicit).  A column whose norm is below
# floatmin — e.g. subnormal cancellation residue — is treated as already
# annihilated (τ = 0): scaling by 1/(ξ₁ + ν) would overflow to Inf and turn
# the zeros below into NaN.  The perturbation is under floatmin.
@inline function _house!(Fs::AbstractMatrix{Tv}, r::Int, j::Int, mf::Int, nrm) where {Tv}
    if nrm < floatmin(real(Tv))
        @inbounds for i in (r + 1):mf
            Fs[i, j] = zero(Tv)
        end
        return zero(Tv)
    end
    @inbounds ξ1 = Fs[r, j]
    ν = copysign(nrm, real(ξ1))
    ξ1 += ν
    @inbounds Fs[r, j] = -ν
    ξinv = inv(ξ1)
    @inbounds for i in (r + 1):mf
        Fs[i, j] *= ξinv
    end
    return ξ1 / ν
end

# Apply Hᴴ = I - conj(τ) v vᴴ (v from Fs[r:mf, j]) to columns `cs` of Fs.
@inline function _apply_house_adj!(Fs::AbstractMatrix, r::Int, j::Int, mf::Int, τ, cs)
    iszero(τ) && return Fs
    @inbounds for c in cs
        acc = Fs[r, c]
        @simd for i in (r + 1):mf
            acc += conj(Fs[i, j]) * Fs[i, c]
        end
        acc *= conj(τ)
        Fs[r, c] -= acc
        @simd for i in (r + 1):mf
            Fs[i, c] -= Fs[i, j] * acc
        end
    end
    return Fs
end

# Apply (H_r0 ⋯ H_{r0+nk-1})ᴴ = I - V Tᴴ Vᴴ to columns `cs` of Fs (compact WY
# representation, T by the LAPACK larft recursion
# T[1:i-1, i] = -τᵢ T[1:i-1, 1:i-1] Vᴴ vᵢ).
function _apply_block_adj!(
        Fs::Matrix{Tv}, r0::Int, nk::Int, mf::Int, tau::Vector{Tv},
        hc::Vector{Int}, cs::UnitRange{Int}
    ) where {Tv}
    mrow = mf - r0 + 1
    V = zeros(Tv, mrow, nk)
    @inbounds for q in 1:nk
        col = hc[r0 + q - 1]
        V[q, q] = one(Tv)
        for i in (q + 1):mrow
            V[i, q] = Fs[r0 + i - 1, col]
        end
    end
    S = V' * V
    T = zeros(Tv, nk, nk)
    @inbounds for i in 1:nk
        τ = tau[r0 + i - 1]
        T[i, i] = τ
        for a in 1:(i - 1)
            acc = zero(Tv)
            for c in a:(i - 1)
                acc += T[a, c] * S[c, i]
            end
            T[a, i] = -τ * acc
        end
    end
    C = view(Fs, r0:mf, cs)
    Wk = V' * C
    lmul!(UpperTriangular(T)', Wk)
    mul!(C, V, Wk, -one(Tv), one(Tv))
    return nothing
end

_panel_geqrf!(P::StridedMatrix{<:LinearAlgebra.BlasFloat}, tau) = (LAPACK.geqrf!(P, tau); nothing)
_panel_geqrf!(P, tau) = nothing

# Panel Householder QR of front s with Heath's dead-column rule on its np
# pivot columns.  Contribution-block columns are always reduced (while rows
# remain) so the block passed up is upper trapezoidal.
function _factor_front!(F::SupernodalQRFactor{Tv}, s::Int) where {Tv}
    sym = F.sym
    np = sym.sstart[s + 1] - sym.sstart[s]
    Fs = _assemble!(F, s)
    stair = F.stair
    mf, nf = size(Fs)
    tau = F.tau[s]
    hc = F.hcol[s]
    ve = F.vend[s]
    resize!(tau, min(mf, nf))
    resize!(hc, min(mf, nf))
    resize!(ve, min(mf, nf))
    tol = F.tol
    blas = Tv <: LinearAlgebra.BlasFloat
    backup = blas && np > 0 ? Matrix{Tv}(undef, mf, min(_NB, nf)) : Matrix{Tv}(undef, 0, 0)
    r = 1
    npiv = 0
    j = 1
    @inbounds while j <= nf && r <= mf
        jend = min(j + _NB - 1, nf)
        w = jend - j + 1
        r0 = r
        rl = mf                      # last row the panel's reflectors reach
        done = false
        if blas && mf - r + 1 >= w
            # below row stair[jend] the panel is zero (staircase)
            rl = min(mf, max(stair[jend], r + w - 1))
            P = view(Fs, r:rl, j:jend)
            npc = clamp(np - j + 1, 0, w)
            npc > 0 && copyto!(view(backup, 1:(rl - r + 1), 1:w), P)
            _panel_geqrf!(P, view(tau, r:(r + w - 1)))
            ok = true
            for q in 1:npc
                # `!(… > tol)` so that a NaN pivot also sends the panel down
                # the checked path
                if !(abs(Fs[r + q - 1, j + q - 1]) > tol)
                    ok = false
                    break
                end
            end
            if ok
                for q in 1:w
                    hc[r + q - 1] = j + q - 1
                    ve[r + q - 1] = rl
                end
                npiv += npc
                r += w
                done = true
            else
                copyto!(P, view(backup, 1:(rl - r + 1), 1:w))
                rl = mf
            end
        end
        if !done
            for jj in j:jend
                r > mf && break
                nrm = norm(view(Fs, r:mf, jj))
                if jj <= np && nrm <= tol
                    for i in r:mf
                        Fs[i, jj] = zero(Tv)
                    end
                    continue
                end
                τ = _house!(Fs, r, jj, mf, nrm)
                _apply_house_adj!(Fs, r, jj, mf, τ, (jj + 1):jend)
                tau[r] = τ
                hc[r] = jj
                ve[r] = mf
                jj <= np && (npiv += 1)
                r += 1
            end
        end
        nk = r - r0
        nk > 0 && jend < nf && _apply_block_adj!(Fs, r0, nk, rl, tau, hc, (jend + 1):nf)
        j = jend + 1
    end
    resize!(tau, r - 1)
    resize!(hc, r - 1)
    resize!(ve, r - 1)
    F.npiv[s] = npiv
    F.nrefl[s] = r - 1
    return nothing
end

# Compact the singleton rows of R (rows of A) for the substitution in solve:
# pivot, then the off-pivot entries with their columns of x; entries in empty
# columns (x = 0) are dropped.
function _compact_singletons!(F::SupernodalQRFactor{Tv}) where {Tv}
    sym = F.sym
    nsing = length(sym.scol)
    nzA = nonzeros(F.A)
    resize!(F.sptr, nsing + 1)
    resize!(F.spiv, nsing)
    empty!(F.scolx)
    empty!(F.sval)
    F.sptr[1] = 1
    @inbounds for l in 1:nsing
        i = sym.srow[l]
        for p in sym.brp[i]:(sym.brp[i + 1] - 1)
            cp = sym.bcol[p]
            v = nzA[sym.bpos[p]]
            if cp == -l
                F.spiv[l] = v
            elseif cp != 0
                push!(F.scolx, cp > 0 ? sym.qf[cp] : sym.scol[-cp])
                push!(F.sval, v)
            end
        end
        F.sptr[l + 1] = length(F.scolx) + 1
    end
    return F
end

# True iff every singleton pivot still clears the dead-column threshold
# (the singleton set is fixed by the analysis; new values can invalidate it).
function _singletons_ok(F::SupernodalQRFactor)
    nzA = nonzeros(F.A)
    @inbounds for p in F.sym.spos
        abs(nzA[p]) > F.tol || return false
    end
    return true
end

function _factor_core!(F::SupernodalQRFactor{Tv}) where {Tv}
    _set_tol!(F)
    sym = F.sym
    ns = length(sym.sstart) - 1
    maxmf = 0
    rnk = length(sym.scol)
    for s in 1:ns
        _factor_front!(F, s)
        maxmf = max(maxmf, length(F.frows[s]))
        rnk += F.npiv[s]
    end
    F.maxmf = maxmf
    F.rank = rnk
    _compact_singletons!(F)
    if size(F.Y, 1) < maxmf
        F.Y = Matrix{Tv}(undef, maxmf, size(F.Y, 2))
    end
    _find_dead!(F)
    _pack_R!(F)
    F.Kfac = sym.transposed ? _deficiency_system(F) : _empty_chol(Tv)
    return F
end

_empty_chol(::Type{Tv}) where {Tv} =
    LinearAlgebra.Cholesky(Matrix{Tv}(undef, 0, 0), 'U', LinearAlgebra.BlasInt(0))

# Pack the R rows of every front that touches a dead column (see `Rp`).
function _pack_R!(F::SupernodalQRFactor{Tv}) where {Tv}
    sym = F.sym
    ns = length(sym.sstart) - 1
    resize!(F.Rp, ns)
    resize!(F.Rcols, ns)
    deadidx = F.deadidx
    @inbounds for s in 1:ns
        c1 = sym.sstart[s]
        np = sym.sstart[s + 1] - c1
        U = sym.cols[s]
        nu = length(U)
        npv = F.npiv[s]
        hc = F.hcol[s]
        ndu = 0
        for t in 1:nu
            deadidx[U[t]] > 0 && (ndu += 1)
        end
        if npv == 0 || (npv == np && ndu == 0)
            F.Rp[s] = Matrix{Tv}(undef, 0, 0)
            F.Rcols[s] = Int[]
            continue
        end
        Fs = F.fronts[s]
        nlu = nu - ndu
        Rp = zeros(Tv, npv, npv + nlu)
        for a in 1:npv, q in 1:a          # row q reaches pivot column hc[a] iff q ≤ a
            Rp[q, a] = Fs[q, hc[a]]
        end
        rc = Vector{Int}(undef, nlu)
        k = 0
        for t in 1:nu
            deadidx[U[t]] > 0 && continue
            k += 1
            rc[k] = U[t]
            for q in 1:npv
                Rp[q, npv + k] = Fs[q, np + t]
            end
        end
        F.Rp[s] = Rp
        F.Rcols[s] = rc
    end
    return F
end

# Dead columns of B: every column that is not the pivot of a live R row.
function _find_dead!(F::SupernodalQRFactor)
    sym = F.sym
    deadidx = F.deadidx
    fill!(deadidx, -1)
    @inbounds for s in 1:(length(sym.sstart) - 1)
        c1 = sym.sstart[s]
        hc = F.hcol[s]
        for k in 1:F.npiv[s]
            deadidx[c1 + hc[k] - 1] = 0
        end
    end
    dead = F.dead
    empty!(dead)
    @inbounds for c in eachindex(deadidx)
        if deadidx[c] == -1
            push!(dead, c)
            deadidx[c] = length(dead)
        end
    end
    return dead
end
