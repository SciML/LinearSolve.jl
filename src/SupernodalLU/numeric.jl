# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# Numeric phase: left-looking supernodal LU on the symmetric pattern, the
# Schenk–Gärtner scheme (FGCS 20(3), 2004).
#
# Each supernode s owns columns c1:c2 (np wide) with update rows R (nu of them,
# shared by the whole block since L and Uᵀ live on one symmetric structure):
#
#   W[s] : (np+nu)×np dense panel.  Rows 1:np hold the getrf-style factored
#          diagonal block (unit-L11 below, U11 on/above the diagonal); rows
#          np+1:np+nu hold L21 = A21·U11⁻¹, indexed by R (global row ids).
#   Z[s] : np×nu dense panel holding U12 = L11⁻¹·A12, columns indexed by R.
#
# Left-looking updates: when supernode d finishes, it is threaded onto the
# linked list of the supernode owning its first update row.  When target s is
# assembled, every descendant d on its list contributes two GEMMs
#
#   ΔW = L_d[j1:end, :] · U_d[:, j1:j2]      (rows/cols of d hitting s)
#   ΔZ = L_d[j1:j2, :] · U_d[:, j2+1:end]
#
# scattered through a relative-index map, after which d advances to the list
# of the supernode owning its next update row.  This is the left-looking
# supernodal schedule (as in Schenk–Gärtner and CHOLMOD) rather than a
# multifrontal stack: no contribution blocks, updates land directly in the
# target panel.
#
# Pivoting follows the papers: partial pivoting restricted to the supernode diagonal
# block; when even the best in-block candidate is below ε‖A‖, the diagonal
# entry is kept and perturbed to ±ε‖A‖ (static pivoting), counted in
# `nperturbed` and compensated later by iterative refinement.
#
# Every workspace the numeric phase touches is preallocated on the
# factorization object, and the factorized matrix `V = M[qf,qf]` (with `M` the
# possibly matched/scaled input) plus its transpose are refreshed in place
# through precomputed position maps — so `snlu!` refactorization and `solve!`
# are allocation-free after the first factorization (the PureKLU guarantee).

"""
    SupernodalLUFactor

Factorization object of [`snlu`](@ref).  Satisfies `A[F.p, F.q] ≈ F.L * F.U`
with `L` unit lower triangular and `U` upper triangular.  `nperturbed(F)`
reports how many pivots required static perturbation.
"""
mutable struct SupernodalLUFactor{Tv, Ti <: Integer}
    sym::SymbolicAnalysis
    W::Vector{Matrix{Tv}}            # supernode panels: [diag LU block; L21]
    Z::Vector{Matrix{Tv}}            # supernode panels: U12
    prow::Vector{Int}                # factor row -> row of V
    p::Vector{Int}                   # factor row -> row of M  (= qf[prow])
    rowsfac::Vector{Vector{Int}}     # rows[s] relabelled to factor rows
    nperturbed::Int
    anorm::Float64                   # ‖M‖_max (NaN for non-Float64-convertible Tv)
    eps_pivot::Float64
    A::SparseMatrixCSC{Tv, Ti}       # original matrix (iterative refinement)
    work::Vector{Tv}                 # solve workspace (length n)
    gbuf::Vector{Tv}                 # gemv/gemm scratch
    # MC64 matching + scaling preprocessing (identity when disabled): the
    # factorized matrix is M = (diag(Rs)·A·diag(Cs))[rowperm, :]
    rowperm::Vector{Int}
    Rs::Vector{Float64}
    Cs::Vector{Float64}
    matched::Bool
    # ---- preallocated numeric state (zero-allocation refactor/solve) ----
    V::SparseMatrixCSC{Tv, Ti}       # M[qf,qf]; values refreshed in place
    Vt::SparseMatrixCSC{Tv, Ti}      # transpose of V (fixed structure)
    Vpos::Vector{Int}                # V.nzval[p] = scale·A.nzval[Vpos[p]]
    Vscale::Vector{Float64}          # per-entry Rs·Cs scale; empty if unmatched
    Tpos::Vector{Int}                # Vt.nzval[q] = V.nzval[Tpos[q]]
    relmap::Vector{Int}              # global row -> local panel position
    ipiv::Vector{Int}                # block pivot scratch
    rowfac::Vector{Int}              # invperm(prow) scratch
    solve_scratch::Matrix{Tv}        # multi-RHS solve workspace (grown on demand)
    ir_r::Vector{Tv}                 # iterative-refinement residual
    ir_dx::Vector{Tv}                # iterative-refinement correction
    btmp::Vector{Tv}                 # in-place ldiv! RHS copy
    threaded::Bool                   # use the etree-parallel numeric phase
end

nperturbed(F::SupernodalLUFactor) = F.nperturbed

@inline _pert(d::Real, epsv::Real) = d < 0 ? -oftype(d, epsv) : oftype(d, epsv)
@inline _pert(d::Complex, epsv::Real) =
    iszero(d) ? oftype(d, complex(epsv)) : d / abs(d) * epsv

# Informational Float64 view of a generic magnitude; value types with no exact
# Float64 conversion (e.g. ForwardDiff.Dual) report NaN — thresholds themselves
# stay in the native arithmetic.
_scalarval(x::Union{AbstractFloat, Integer, Rational}) = Float64(x)
_scalarval(x::Real) = NaN
_scalarval(x) = NaN

# Blocked dense LU of the np×np diagonal block of W (rows 1:np), partial
# pivoting restricted to the block, static perturbation below `epsv` (in the
# native |Tv| arithmetic).  Row swaps stay within W's np columns; the caller
# replays them on the U12 panel afterwards with `_apply_ipiv!` (equivalent,
# since this kernel never reads that panel).  `ipiv[j]` records the row
# swapped with j at step j (LAPACK convention); must come in as ipiv[j] == j.
function _block_lu!(
        W::Matrix{Tv}, np::Int, epsv, ipiv::AbstractVector{Int}
    ) where {Tv}
    npert = 0
    nb = 48
    @inbounds for kb in 1:nb:np
        ke = min(kb + nb - 1, np)
        for j in kb:ke
            pj = j
            mx = abs(W[j, j])
            for r in (j + 1):np
                a = abs(W[r, j])
                if a > mx
                    mx = a
                    pj = r
                end
            end
            if mx < epsv
                W[j, j] = _pert(W[j, j], epsv)
                npert += 1
            elseif pj != j
                for c in 1:np
                    W[j, c], W[pj, c] = W[pj, c], W[j, c]
                end
                ipiv[j] = pj
            end
            piv = W[j, j]
            for r in (j + 1):np
                W[r, j] /= piv
            end
            for c in (j + 1):ke
                wjc = W[j, c]
                iszero(wjc) && continue
                @simd for r in (j + 1):np
                    W[r, c] = muladd(-W[r, j], wjc, W[r, c])
                end
            end
        end
        if ke < np
            ldiv!(
                UnitLowerTriangular(view(W, kb:ke, kb:ke)),
                view(W, kb:ke, (ke + 1):np)
            )
            mul!(
                view(W, (ke + 1):np, (ke + 1):np),
                view(W, (ke + 1):np, kb:ke), view(W, kb:ke, (ke + 1):np),
                -one(Tv), one(Tv)
            )
        end
    end
    return npert
end

# Replay the sequential block row swaps on the U12 panel.
@inline function _apply_ipiv!(Z::Matrix, ipiv::AbstractVector{Int}, np::Int, nu::Int)
    nu == 0 && return nothing
    @inbounds for j in 1:np
        pj = ipiv[j]
        if pj != j
            for c in 1:nu
                Z[j, c], Z[pj, c] = Z[pj, c], Z[j, c]
            end
        end
    end
    return nothing
end

# ---- overridable dense-kernel hooks ---------------------------------------
# The package extension (RecursiveFactorization + TriangularSolve, the same
# components LinearSolve.jl's default dense LU prefers at supernode block
# sizes) overrides these for Float32/Float64 panels; the defaults run the
# built-in blocked kernel and stdlib BLAS.  A plain LAPACK-getrf fast path was
# measured and is NOT used: at supernode block sizes the mandatory block
# backup + call overhead cancels getrf's gains (±5 % vs the built-in kernel),
# while RecursiveFactorization is a genuine win.  `scratch` is free workspace
# the getrf hook may resize (backs up the block so a too-small pivot can fall
# back to the static-perturbation kernel).

@inline function _backup_block!(scratch::Vector{Tv}, W::Matrix{Tv}, np::Int) where {Tv}
    length(scratch) < np * np && resize!(scratch, np * np)
    k = 0
    @inbounds for c in 1:np, r in 1:np
        k += 1
        scratch[k] = W[r, c]
    end
    return nothing
end

@inline function _restore_block!(W::Matrix{Tv}, scratch::Vector{Tv}, np::Int) where {Tv}
    k = 0
    @inbounds for c in 1:np, r in 1:np
        k += 1
        W[r, c] = scratch[k]
    end
    return nothing
end

# Accept an optimistically-factored block iff every pivot clears the static
# threshold; otherwise restore it and rerun the perturbing kernel.
@inline function _accept_or_perturb!(
        W::Matrix{Tv}, np::Int, epsv, ipiv::AbstractVector{Int},
        scratch::Vector{Tv}
    ) where {Tv}
    ok = true
    @inbounds for j in 1:np
        if !(abs(W[j, j]) >= epsv)
            ok = false
            break
        end
    end
    ok && return 0
    _restore_block!(W, scratch, np)
    @inbounds for j in 1:np
        ipiv[j] = j
    end
    return _block_lu!(W, np, epsv, ipiv)
end

function _block_getrf!(
        W::Matrix{Tv}, np::Int, epsv, ipiv::AbstractVector{Int},
        scratch::Vector{Tv}
    ) where {Tv}
    return _block_lu!(W, np, epsv, ipiv)
end


# L21 := L21 · U11⁻¹
function _panel_rdiv!(W::Matrix{Tv}, np::Int, len::Int) where {Tv}
    rdiv!(view(W, (np + 1):len, 1:np), UpperTriangular(view(W, 1:np, 1:np)))
    return nothing
end

# U12 := L11⁻¹ · U12
function _panel_ldiv!(W::Matrix{Tv}, np::Int, Z::Matrix{Tv}) where {Tv}
    ldiv!(UnitLowerTriangular(view(W, 1:np, 1:np)), Z)
    return nothing
end

# Transpose of V with a position map: Vt.nzval[q] == V.nzval[Tpos[q]] forever
# (structure is static), so refactorization refreshes Vt by one gather pass.
function _transpose_map(V::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    n = size(V, 2)
    Vp = getcolptr(V)
    Vi = rowvals(V)
    Vx = nonzeros(V)
    nz = nnz(V)
    Vtp = zeros(Ti, n + 1)
    @inbounds for p in 1:nz
        Vtp[Vi[p] + 1] += 1
    end
    Vtp[1] = 1
    @inbounds for i in 1:n
        Vtp[i + 1] += Vtp[i]
    end
    cursor = Vtp[1:n]
    Vti = Vector{Ti}(undef, nz)
    Vtx = Vector{Tv}(undef, nz)
    Tpos = Vector{Int}(undef, nz)
    @inbounds for j in 1:n
        for p in Vp[j]:(Vp[j + 1] - 1)
            i = Vi[p]
            q = cursor[i]
            cursor[i] = q + 1
            Vti[q] = j
            Vtx[q] = Vx[p]
            Tpos[q] = p
        end
    end
    return SparseMatrixCSC(n, n, Vtp, Vti, Vtx), Tpos
end

# Position map from the user's A into V = M[qf,qf], where M is A row-permuted
# by `rowperm` and scaled by Rs/Cs (identity permutation/scaling when
# unmatched).  V entry p sources A.nzval[Vpos[p]] times Vscale[p].
function _build_vpos!(
        F::SupernodalLUFactor{Tv, Ti}, A::SparseMatrixCSC{Tv, Ti}
    ) where {Tv, Ti}
    n = F.sym.n
    qf = F.sym.qf
    V = F.V
    Vp = getcolptr(V)
    Vi = rowvals(V)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    σ = F.rowperm
    σinv = invperm(σ)
    qfinv = invperm(qf)
    resize!(F.Vpos, nnz(V))
    domatch = F.matched
    resize!(F.Vscale, domatch ? nnz(V) : 0)
    posmap = zeros(Int, n)                    # V row -> V nz position (per col)
    @inbounds for j in 1:n
        for p in Vp[j]:(Vp[j + 1] - 1)
            posmap[Vi[p]] = p
        end
        acol = qf[j]
        cj = domatch ? F.Cs[acol] : 1.0
        for pA in Ap[acol]:(Ap[acol + 1] - 1)
            arow = Ai[pA]
            p = posmap[qfinv[σinv[arow]]]
            F.Vpos[p] = pA
            domatch && (F.Vscale[p] = F.Rs[arow] * cj)
        end
    end
    return F
end

# Refresh V and Vt values from A through the precomputed maps (no allocation).
function _load_values!(F::SupernodalLUFactor{Tv}, A::SparseMatrixCSC{Tv}) where {Tv}
    Ax = nonzeros(A)
    Vx = nonzeros(F.V)
    Vtx = nonzeros(F.Vt)
    Vpos = F.Vpos
    Tpos = F.Tpos
    if isempty(F.Vscale)
        @inbounds @simd for p in eachindex(Vx)
            Vx[p] = Ax[Vpos[p]]
        end
    else
        Vscale = F.Vscale
        @inbounds @simd for p in eachindex(Vx)
            Vx[p] = Vscale[p] * Ax[Vpos[p]]
        end
    end
    @inbounds @simd for q in eachindex(Vtx)
        Vtx[q] = Vx[Tpos[q]]
    end
    return F
end

# Numeric factorization over the preallocated state (V/Vt already loaded).
# Allocation-free in the serial path; reusable for refactorization with new
# values.  The threaded path (opt-in, engages for large enough problems)
# follows the same static contribution schedule and pivot sequence as
# serial; entry values can differ only by the floating-point rounding of
# BLAS-parallel sums (it manages BLAS threads across its two phases).
function _factor_core!(F::SupernodalLUFactor{Tv}) where {Tv}
    n = F.sym.n
    nsuper = length(F.sym.sstart) - 1
    Vx = nonzeros(F.V)

    anorm = _maxabs(Vx)
    epsv = F.eps_pivot * anorm
    F.anorm = _scalarval(anorm)
    copyto!(F.prow, 1:n)

    if F.threaded && Threads.nthreads() > 1 && n >= 4096 && nsuper > 4
        _core_threaded!(F, epsv)
    else
        npert = 0
        @inbounds for s in 1:nsuper
            npert += _process_supernode!(F, s, epsv, F.relmap, F.ipiv, F.gbuf)
        end
        F.nperturbed = npert
    end

    # factor-row relabelling of every panel's update rows
    qf = F.sym.qf
    rows = F.sym.rows
    prow = F.prow
    rowfac = F.rowfac
    @inbounds for k in 1:n
        rowfac[prow[k]] = k
    end
    @inbounds for s in 1:nsuper
        R = rows[s]
        Rf = F.rowsfac[s]
        for t in eachindex(R)
            Rf[t] = rowfac[R[t]]
        end
    end
    @inbounds for k in 1:n
        F.p[k] = qf[prow[k]]
    end
    return F
end

# Assemble, update, and factor one supernode.  Reads finished descendants'
# panels (via the static schedule), writes only this supernode's panels and
# its own prow range — the unit of work for both serial and threaded phases.
# `relmap` must be all-zero on entry and is restored to all-zero.
function _process_supernode!(
        F::SupernodalLUFactor{Tv}, s::Int, epsv, relmap::Vector{Int},
        ipiv::Vector{Int}, buf::Vector{Tv}
    ) where {Tv}
    sym = F.sym
    sstart = sym.sstart
    rows = sym.rows
    V = F.V
    Vt = F.Vt
    Vp = getcolptr(V)
    Vi = rowvals(V)
    Vx = nonzeros(V)
    Vtp = getcolptr(Vt)
    Vti = rowvals(Vt)
    Vtx = nonzeros(Vt)
    prow = F.prow

    c1 = sstart[s]
    c2 = sstart[s + 1] - 1
    np = c2 - c1 + 1
    R = rows[s]
    nu = length(R)
    len = np + nu
    Ws = F.W[s]
    Zs = F.Z[s]
    fill!(Ws, zero(Tv))
    fill!(Zs, zero(Tv))

    @inbounds begin
        for a in 1:np
            relmap[c1 + a - 1] = a
        end
        for t in 1:nu
            relmap[R[t]] = np + t
        end

        # assemble original entries of V
        for a in 1:np
            j = c1 + a - 1
            for pp in Vp[j]:(Vp[j + 1] - 1)
                i = Vi[pp]
                if i >= c1
                    li = relmap[i]
                    li != 0 && (Ws[li, a] += Vx[pp])
                end
            end
            for pp in Vtp[j]:(Vtp[j + 1] - 1)   # row j of V: strict-right entries
                k = Vti[pp]
                if k > c2
                    lk = relmap[k]
                    lk != 0 && (Zs[a, lk - np] += Vtx[pp])
                end
            end
        end

        # left-looking updates from the static contribution schedule
        for k in sym.ccp[s]:(sym.ccp[s + 1] - 1)
            _apply_contribution!(F, sym.cd[k], sym.cj1[k], c1, c2, np, Ws, Zs, relmap, buf)
        end

        # factor the diagonal block (restricted pivoting + static perturbation)
        ipv = view(ipiv, 1:np)
        for a in 1:np
            ipv[a] = a
        end
        npert = _block_getrf!(Ws, np, epsv, ipv, buf)
        _apply_ipiv!(Zs, ipv, np, nu)
        # fold the block row swaps (sequential, LAPACK-style) into prow; the
        # touched range c1:c2 belongs exclusively to this supernode
        for a in 1:np
            k = ipv[a]
            if k != a
                prow[c1 + a - 1], prow[c1 + k - 1] = prow[c1 + k - 1], prow[c1 + a - 1]
            end
        end

        if nu > 0
            _panel_rdiv!(Ws, np, len)
            _panel_ldiv!(Ws, np, Zs)
        end

        for a in 1:np
            relmap[c1 + a - 1] = 0
        end
        for t in 1:nu
            relmap[R[t]] = 0
        end
    end
    return npert
end

# One descendant's two-GEMM contribution to target supernode s (columns
# c1:c2), starting at offset j1 of the descendant's update-row list.
@inline function _apply_contribution!(
        F::SupernodalLUFactor{Tv}, d::Int, j1::Int, c1::Int, c2::Int, np::Int,
        Ws::Matrix{Tv}, Zs::Matrix{Tv}, relmap::Vector{Int}, buf::Vector{Tv}
    ) where {Tv}
    rows = F.sym.rows
    sstart = F.sym.sstart
    Rd = rows[d]
    lend = length(Rd)
    npd = sstart[d + 1] - sstart[d]
    j2 = j1
    @inbounds while j2 <= lend && Rd[j2] <= c2
        j2 += 1
    end
    j2 -= 1
    w = j2 - j1 + 1
    m1 = lend - j1 + 1
    Wd = F.W[d]
    Zd = F.Z[d]
    Ld = view(Wd, (npd + j1):(npd + lend), 1:npd)          # rows Rd[j1:end]
    # ΔW: rows Rd[j1:end] × cols Rd[j1:j2] of the target panel
    length(buf) < m1 * w && resize!(buf, max(m1 * w, 2 * length(buf)))
    C1 = reshape(view(buf, 1:(m1 * w)), m1, w)
    mul!(C1, Ld, view(Zd, :, j1:j2))
    @inbounds for c in 1:w
        a = Rd[j1 + c - 1] - c1 + 1                         # target column
        for r in 1:m1
            Ws[relmap[Rd[j1 + r - 1]], a] -= C1[r, c]
        end
    end
    # ΔZ: rows Rd[j1:j2] (in-block) × cols Rd[j2+1:end] of U12
    m2 = lend - j2
    if m2 > 0
        length(buf) < w * m2 && resize!(buf, max(w * m2, 2 * length(buf)))
        C2 = reshape(view(buf, 1:(w * m2)), w, m2)
        mul!(C2, view(Wd, (npd + j1):(npd + j2), 1:npd), view(Zd, :, (j2 + 1):lend))
        @inbounds for c in 1:m2
            zc = relmap[Rd[j2 + c]] - np                    # target U column
            for r in 1:w
                Zs[Rd[j1 + r - 1] - c1 + 1, zc] -= C2[r, c]
            end
        end
    end
    return nothing
end

# Threaded numeric phase: subtree-frontier parallelism on the supernodal
# elimination tree (the PureUMFPACK scheme adapted to left-looking).  Because
# supernodes are postordered, every subtree is a contiguous supernode range
# whose contributions stay inside the range — so an antichain of subtrees can
# be factored fully independently.  The frontier is grown by repeatedly
# splitting the heaviest splittable subtree until ~4·nthreads subtrees exist;
# the split-off ancestors (the "top" of the tree) run serially afterwards, in
# ascending order, once all their descendants are done.  Only a handful of
# tasks and channel operations exist regardless of problem size, each worker
# owns one private (relmap, ipiv, buf) workspace, and every supernode still
# applies its contributions in static-schedule order — same pivot sequence
# and structure as serial, with entry-level rounding differences only from
# BLAS-parallel sums.
function _core_threaded!(F::SupernodalLUFactor{Tv}, epsv) where {Tv}
    sym = F.sym
    rows = sym.rows
    snof = sym.snof
    sstart = sym.sstart
    nsuper = length(sstart) - 1

    parent = Vector{Int}(undef, nsuper)
    first_ = collect(1:nsuper)                 # subtree(s) = first_[s]:s
    wk = Vector{Float64}(undef, nsuper)        # per-supernode flop proxy
    @inbounds for s in 1:nsuper
        parent[s] = isempty(rows[s]) ? 0 : snof[rows[s][1]]
        np = Float64(sstart[s + 1] - sstart[s])
        len = np + length(rows[s])
        wk[s] = np * len * len
    end
    SW = copy(wk)                              # subtree work totals
    @inbounds for s in 1:nsuper
        p = parent[s]
        if p != 0
            first_[p] = min(first_[p], first_[s])
            SW[p] += SW[s]
        end
    end
    # children lists in flat CSR form
    ccount = zeros(Int, nsuper)
    @inbounds for s in 1:nsuper
        p = parent[s]
        p != 0 && (ccount[p] += 1)
    end
    cptr = Vector{Int}(undef, nsuper + 1)
    cptr[1] = 1
    @inbounds for s in 1:nsuper
        cptr[s + 1] = cptr[s] + ccount[s]
    end
    clist = Vector{Int}(undef, cptr[nsuper + 1] - 1)
    ccur = cptr[1:nsuper]
    @inbounds for s in 1:nsuper
        p = parent[s]
        if p != 0
            clist[ccur[p]] = s
            ccur[p] = ccur[p] + 1
        end
    end

    T = Threads.nthreads()
    frontier = [s for s in 1:nsuper if parent[s] == 0]
    istop = falses(nsuper)
    while length(frontier) < 4 * T
        besti = 0
        bestw = -1.0
        for (i, s) in enumerate(frontier)
            if cptr[s + 1] > cptr[s] && SW[s] > bestw
                bestw = SW[s]
                besti = i
            end
        end
        besti == 0 && break                    # nothing splittable left
        s = frontier[besti]
        frontier[besti] = frontier[end]
        pop!(frontier)
        istop[s] = true
        append!(frontier, view(clist, cptr[s]:(cptr[s + 1] - 1)))
    end
    subtrees = [(first_[s], s) for s in frontier]
    sort!(subtrees; by = t -> -SW[t[2]])       # heaviest first for balance

    # BLAS thread management, always at task barriers (never while our own
    # BLAS calls are in flight): 1 thread during the task-parallel subtree
    # phase (avoids oversubscription), up to `T` threads for the serial top
    # of the tree — that is where the large root separators live, and their
    # big GEMM/TRSM calls parallelize well inside BLAS.  Restored afterwards.
    # Caveat (documented on `snlu`): if *other* user tasks run BLAS
    # concurrently with a threaded factorization, skip `threaded=true`.
    nb0 = BLAS.get_num_threads()
    npert_par = Threads.Atomic{Int}(0)
    try
        if !isempty(subtrees)
            BLAS.set_num_threads(1)
            q = Channel{Tuple{Int, Int}}(length(subtrees))
            foreach(t -> put!(q, t), subtrees)
            close(q)
            nworkers = min(T, length(subtrees))
            @sync for _ in 1:nworkers
                Threads.@spawn begin
                    relmap = zeros(Int, sym.n)
                    ipiv = Vector{Int}(undef, sym.n)
                    buf = Vector{Tv}(undef, 64)
                    acc = 0
                    for (lo, hi) in q
                        for s in lo:hi
                            acc += _process_supernode!(F, s, epsv, relmap, ipiv, buf)
                        end
                    end
                    Threads.atomic_add!(npert_par, acc)
                end
            end
        end
        # top of the tree: one task, ascending (all subtree work is done),
        # with BLAS-level parallelism inside the big panels
        BLAS.set_num_threads(max(nb0, T))
        acc = 0
        @inbounds for s in 1:nsuper
            istop[s] && (acc += _process_supernode!(F, s, epsv, F.relmap, F.ipiv, F.gbuf))
        end
        F.nperturbed = npert_par[] + acc
    finally
        BLAS.set_num_threads(nb0)
    end
    return F
end

# Native-arithmetic max |x| (keeps ForwardDiff.Dual etc. in their own type).
function _maxabs(x::AbstractVector{Tv}) where {Tv}
    isempty(x) && return abs(one(Tv))
    m = abs(x[1])
    @inbounds for k in 2:length(x)
        a = abs(x[k])
        a > m && (m = a)
    end
    iszero(m) && return abs(one(Tv))
    return m
end

# Build B = (diag(r)·A·diag(c))[rowperm, :] — the matrix the panels factorize
# when MC64 matching/scaling preprocessing is active.
function _matched_matrix(A::SparseMatrixCSC{Tv}, ms::MatchScale) where {Tv}
    B = A[ms.rowperm, :]
    n = size(B, 2)
    Bp = getcolptr(B)
    Bi = rowvals(B)
    Bx = nonzeros(B)
    @inbounds for j in 1:n
        cj = ms.c[j]
        for p in Bp[j]:(Bp[j + 1] - 1)
            Bx[p] *= ms.r[ms.rowperm[Bi[p]]] * cj
        end
    end
    return B
end
