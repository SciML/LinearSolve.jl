# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# Symbolic analysis for the multifrontal QR of B (B = A, or Aᴴ for
# minimum-norm wide solves).  BᴴB is never formed — its pattern can be
# quadratic in nnz(B) — everything works on B's own pattern:
#
#   1. column singletons (B = A only): a column with a single entry among the
#      remaining rows pivots on that row at zero cost; the row is removed and
#      the search repeats (Davis 2011, after the usual singleton/BTF
#      preprocessing).  They form an upper-triangular block R11 solved by
#      substitution; the multifrontal QR handles what remains.
#   2. fill-reducing column order: approximate minimum degree on a quotient
#      graph built from B's rows — long rows as initial elements (each row
#      *is* the clique it contributes to BᴴB, the idea behind COLAMD), short
#      rows as explicit edges — run by the vendored AMD core from that preset
#      state.
#   3. column elimination tree + postorder on the *star* reduction of BᴴB:
#      every row of B contributes the edges (f, j) from its leftmost column f
#      to each of its other columns j.  Eliminating f first turns the star
#      into the row's clique, so the filled graph — hence the etree and the
#      structure of R — is that of BᴴB, at O(nnz(B)) size.  A postorder keeps
#      every row's columns on the etree path above its leftmost column, so the
#      reduction stays valid after reordering.
#   4. column counts of R by row-subtree counting (Gilbert, Ng & Peyton 1994)
#      without forming any column structure, fundamental supernodes
#      (SupernodalLU's routine), relaxed amalgamation by a QR front cost
#      model, and the structure of each supernode only, by climbing the
#      supernodal tree.
#   5. the assembly tree: each row of B is assembled into the front that owns
#      its leftmost column (Matstoms 1994; Davis 2011).

# Row-wise access to the pattern of A: row i holds columns
# tcol[tp[i]:tp[i+1]-1] (ascending) at positions tpos[...] of nonzeros(A).
function _row_access(A::SparseMatrixCSC)
    m, n = size(A)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    nz = nnz(A)
    tp = zeros(Int, m + 1)
    @inbounds for p in 1:nz
        tp[Ai[p] + 1] += 1
    end
    tp[1] = 1
    @inbounds for i in 1:m
        tp[i + 1] += tp[i]
    end
    cur = tp[1:m]
    tcol = Vector{Int}(undef, nz)
    tpos = Vector{Int}(undef, nz)
    @inbounds for j in 1:n
        for p in Ap[j]:(Ap[j + 1] - 1)
            i = Ai[p]
            k = cur[i]
            tcol[k] = j
            tpos[k] = p
            cur[i] = k + 1
        end
    end
    return tp, tcol, tpos
end

# Column singletons of B = A (Davis 2011): repeatedly take a column with
# exactly one entry among the rows not yet used, if that entry exceeds `tol`
# in magnitude, and retire its row.  Returns the singleton rows, columns and
# the positions of the pivots in nonzeros(A), in discovery order — with rows
# and columns in that order the singleton block is upper triangular.
function _singletons(
        A::SparseMatrixCSC, tp::Vector{Int}, tcol::Vector{Int}, tol::Real
    )
    m, n = size(A)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    Ax = nonzeros(A)
    alive = trues(m)
    cnt = Vector{Int}(undef, n)
    queue = Int[]
    @inbounds for j in 1:n
        cnt[j] = Ap[j + 1] - Ap[j]
        cnt[j] == 1 && push!(queue, j)
    end
    taken = falses(n)
    srow = Int[]
    scol = Int[]
    spos = Int[]
    h = 1
    @inbounds while h <= length(queue)
        j = queue[h]
        h += 1
        (taken[j] || cnt[j] != 1) && continue
        pj = 0
        for p in Ap[j]:(Ap[j + 1] - 1)
            if alive[Ai[p]]
                pj = p
                break
            end
        end
        abs(Ax[pj]) > tol || continue
        i = Ai[pj]
        push!(srow, i)
        push!(scol, j)
        push!(spos, pj)
        taken[j] = true
        alive[i] = false
        for p in tp[i]:(tp[i + 1] - 1)
            c = tcol[p]
            cnt[c] -= 1
            cnt[c] == 1 && !taken[c] && push!(queue, c)
        end
    end
    return srow, scol, spos
end

# Rows of B with at most this many entries are expanded into explicit
# edges for the column ordering; longer rows stay quotient-graph elements.
const _ORDER_SHORT_ROW = 16

# Column ordering of B (columns 1:nB, pattern given both column- and
# row-wise) for QR, without forming BᴴB: approximate minimum degree, run by
# the vendored AMD core from a preset quotient graph in which
#   - each long row of B (> _ORDER_SHORT_ROW entries) is an initial element:
#     the clique it contributes to BᴴB, stored as its column list (the idea
#     behind COLAMD), so long rows never cost more than their length;
#   - short rows are expanded into explicit column-column edges, which keeps
#     AMD's approximate degrees sharp where many short rows overlap — their
#     total size is at most _ORDER_SHORT_ROW · nnz(B).
# Initial degrees are exact BᴴB degrees, counted by marker union with an
# early exit, and columns whose degree exceeds the dense threshold (as do
# rows with more entries than it) are left out and ordered last.
function _col_order(
        mB::Int, nB::Int, bcp::Vector{Int}, bci::Vector{Int},
        brp::Vector{Int}, brc::Vector{Int}
    )
    nB == 0 && return Int[]
    dense = max(16, round(Int, 10 * sqrt(nB)))
    rowok = falses(mB)
    @inbounds for i in 1:mB
        rowok[i] = 2 <= brp[i + 1] - brp[i] <= dense
    end
    # exact BᴴB degree of each column (stopping once past `dense`); dense
    # columns are excluded, the rest become the variables 1:nv
    var = zeros(Int, nB)
    xdeg = zeros(Int, nB)
    mark = zeros(Int, nB)
    densecols = Int[]
    nv = 0
    @inbounds for j in 1:nB
        c = 0
        mark[j] = j
        for p in bcp[j]:(bcp[j + 1] - 1)
            i = bci[p]
            rowok[i] || continue
            for q in brp[i]:(brp[i + 1] - 1)
                u = brc[q]
                if mark[u] != j
                    mark[u] = j
                    c += 1
                end
            end
            c > dense && break
        end
        xdeg[j] = c
        if c > dense
            push!(densecols, j)
        else
            nv += 1
            var[j] = nv
        end
    end
    vcol = Vector{Int}(undef, nv)
    @inbounds for j in 1:nB
        var[j] > 0 && (vcol[var[j]] = j)
    end
    # rows by number of variable columns: < 2 contribute nothing, long ones
    # become elements nv+1:N
    rlen = zeros(Int, mB)
    elt = zeros(Int, mB)
    ne = 0
    @inbounds for i in 1:mB
        rowok[i] || continue
        c = 0
        for p in brp[i]:(brp[i + 1] - 1)
            var[brc[p]] > 0 && (c += 1)
        end
        rlen[i] = c >= 2 ? c : 0
        if rlen[i] > _ORDER_SHORT_ROW
            ne += 1
            elt[i] = nv + ne
        end
    end
    N = nv + ne
    # explicit adjacency from short rows (deduplicated), element counts
    fill!(mark, 0)
    vadjp = Vector{Int}(undef, nv + 1)
    vadj = Int[]
    nelv = zeros(Int, nv)
    @inbounds for v in 1:nv
        j = vcol[v]
        mark[v] = v
        vadjp[v] = length(vadj) + 1
        for p in bcp[j]:(bcp[j + 1] - 1)
            i = bci[p]
            rlen[i] >= 2 || continue
            if elt[i] != 0
                nelv[v] += 1
                continue
            end
            for q in brp[i]:(brp[i + 1] - 1)
                u = var[brc[q]]
                if u > 0 && mark[u] != v
                    mark[u] = v
                    push!(vadj, u)
                end
            end
        end
    end
    vadjp[nv + 1] = length(vadj) + 1
    # quotient graph: a variable's list is its elements, then its neighbours;
    # an element's list is its variables (0-based node ids, AMD convention)
    Len = zeros(Int, N)
    @inbounds for v in 1:nv
        Len[v] = nelv[v] + (vadjp[v + 1] - vadjp[v])
    end
    @inbounds for i in 1:mB
        elt[i] != 0 && (Len[elt[i]] = rlen[i])
    end
    total = sum(Len; init = 0)
    iwlen = total + total ÷ 5 + 2N + 1
    Pe = Vector{Int}(undef, N)
    Iw = Vector{Int}(undef, iwlen)
    pos = 0
    @inbounds for x in 1:N
        Pe[x] = pos
        pos += Len[x]
    end
    cur = copy(Pe)
    @inbounds for i in 1:mB
        e = elt[i]
        e == 0 && continue
        for p in brp[i]:(brp[i + 1] - 1)
            v = var[brc[p]]
            v == 0 && continue
            Iw[cur[e] + 1] = v - 1
            cur[e] += 1
            Iw[cur[v] + 1] = e - 1
            cur[v] += 1
        end
    end
    Degree = Vector{Int}(undef, N)
    Elen = Vector{Int}(undef, N)
    @inbounds for v in 1:nv
        for p in vadjp[v]:(vadjp[v + 1] - 1)
            Iw[cur[v] + 1] = vadj[p] - 1
            cur[v] += 1
        end
        Degree[v] = min(xdeg[vcol[v]], nv - 1)
        Elen[v] = nelv[v]
    end
    @inbounds for e in (nv + 1):N
        Degree[e] = Len[e]
        Elen[e] = AMD._flip(1 + Len[e])
    end
    Nv = ones(Int, N)
    W = ones(Int, N)
    Next = Vector{Int}(undef, N)
    Last = Vector{Int}(undef, N)
    Head = Vector{Int}(undef, N)
    N > 0 && AMD.amd_2!(
        N, Pe, Iw, Len, iwlen, pos, Nv, Next, Last, Head, Elen, Degree, W;
        dense_alpha = -1.0, aggressive = true, preset = true
    )
    q = Vector{Int}(undef, nB)
    k = 0
    @inbounds for t in 1:N
        x = Last[t] + 1
        if x <= nv
            k += 1
            q[k] = vcol[x]
        end
    end
    for j in densecols
        k += 1
        q[k] = j
    end
    k == nB || error("column ordering dropped columns ($k of $nB)")
    return q
end

# Star reduction of BᴴB in the column numbering `pinv` (original column c is
# column pinv[c], 0 = not a multifrontal column): off-diagonal symmetric
# pattern, sorted, deduplicated.
function _star_pattern(
        mB::Int, nB::Int, brp::Vector{Int}, brc::Vector{Int}, pinv::Vector{Int}
    )
    deg = zeros(Int, nB + 1)
    @inbounds for i in 1:mB
        brp[i + 1] - brp[i] >= 2 || continue
        f = typemax(Int)
        for q in brp[i]:(brp[i + 1] - 1)
            f = min(f, pinv[brc[q]])
        end
        for q in brp[i]:(brp[i + 1] - 1)
            j = pinv[brc[q]]
            j == f && continue
            deg[f + 1] += 1
            deg[j + 1] += 1
        end
    end
    deg[1] = 1
    @inbounds for j in 1:nB
        deg[j + 1] += deg[j]
    end
    cur = deg[1:nB]
    adj = Vector{Int}(undef, deg[nB + 1] - 1)
    @inbounds for i in 1:mB
        brp[i + 1] - brp[i] >= 2 || continue
        f = typemax(Int)
        for q in brp[i]:(brp[i + 1] - 1)
            f = min(f, pinv[brc[q]])
        end
        for q in brp[i]:(brp[i + 1] - 1)
            j = pinv[brc[q]]
            j == f && continue
            adj[cur[f]] = j
            cur[f] += 1
            adj[cur[j]] = f
            cur[j] += 1
        end
    end
    # deduplicate (rows sharing a leftmost column repeat edges)
    mark = zeros(Int, nB)
    cp = Vector{Int}(undef, nB + 1)
    ri = Int[]
    cp[1] = 1
    @inbounds for j in 1:nB
        for p in deg[j]:(deg[j + 1] - 1)
            k = adj[p]
            if mark[k] != j
                mark[k] = j
                push!(ri, k)
            end
        end
        cp[j + 1] = length(ri) + 1
    end
    return permute_pattern(cp, ri, collect(1:nB), nB)
end

# Column counts of the Cholesky factor (diagonal included) of a postordered
# symmetric pattern, without forming any structure: |struct(L[:, j])| is the
# number of row subtrees containing j.  Each row subtree T_i is encoded by
# point weights whose subtree sums are its indicator — +1 at each leaf, -1 at
# the least common ancestor of consecutive leaves (in postorder), -1 at
# parent(i) — so all counts are one postorder accumulation (Gilbert, Ng &
# Peyton 1994).  k is a leaf of T_i iff no earlier neighbour of i lies in k's
# subtree, i.e. first[k] exceeds the largest `first` seen for row i; LCAs come
# from union-find over completed nodes.
function _col_counts(cp::Vector{Int}, ri::Vector{Int}, parent::Vector{Int}, n::Int)
    w = zeros(Int, n)
    first = collect(1:n)
    @inbounds for j in 1:n
        p = parent[j]
        if p != 0
            w[p] -= 1
            first[p] = min(first[p], first[j])
        end
    end
    maxfirst = zeros(Int, n)
    prevleaf = zeros(Int, n)
    uf = collect(1:n)
    @inbounds for k in 1:n
        fk = first[k]
        p = cp[k]
        pend = cp[k + 1]
        while p <= pend
            # neighbours i > k of k, then the diagonal i = k
            if p < pend
                i = ri[p]
                p += 1
                i > k || continue
            else
                i = k
                p += 1
            end
            if fk > maxfirst[i]
                w[k] += 1
                maxfirst[i] = fk
                pl = prevleaf[i]
                if pl != 0
                    r = pl
                    while uf[r] != r
                        uf[r] = uf[uf[r]]
                        r = uf[r]
                    end
                    w[r] -= 1
                end
                prevleaf[i] = k
            end
        end
        parent[k] != 0 && (uf[k] = parent[k])
    end
    @inbounds for j in 1:n
        parent[j] != 0 && (w[parent[j]] += w[j])
    end
    return w
end

# Structure of each supernode — the columns right of its pivot block that its
# R rows reach, i.e. struct(L[:, last column]) — by row subtrees climbed in
# the supernodal tree: row i lies in the structure of every supernode on the
# paths from its neighbours k < i up to i's own supernode.  Ascending i makes
# every list come out sorted.
function _super_structure(
        cp::Vector{Int}, ri::Vector{Int}, snof::Vector{Int},
        sparent::Vector{Int}, ns::Int, n::Int
    )
    cols = [Int[] for _ in 1:ns]
    smark = zeros(Int, ns)
    @inbounds for i in 1:n
        si = snof[i]
        smark[si] = i
        for p in cp[i]:(cp[i + 1] - 1)
            k = ri[p]
            k < i || continue
            s = snof[k]
            while s != 0 && smark[s] != i
                push!(cols[s], i)
                smark[s] = i
                s = sparent[s]
            end
        end
    end
    return cols
end

# Householder work (multiply-adds, up to a constant) of the QR of an m × n
# dense front: Σ_{i<k} (m - i)(n - i), k = min(m, n).
@inline function _front_cost(m::Int, n::Int)
    k = min(m, n)
    k <= 0 && return 0.0
    mf = Float64(m)
    nf = Float64(n)
    kf = Float64(k)
    return kf * mf * nf - (mf + nf) * kf * (kf - 1) / 2 + (kf - 1) * kf * (2kf - 1) / 6
end

# QR amalgamation constants, tuned on SuiteSparse least-squares, square and
# LP matrices: merge when the merged front costs no more than the separate
# ones plus a fixed per-front overhead (_AMALG_GAMMA multiply-adds), with
# stored entries weighted by _AMALG_BETA (they cost every solve), or when the
# merged pivot block has at most _AMALG_TINY columns.
const _AMALG_GAMMA = 1.0e4
const _AMALG_BETA = 100.0
const _AMALG_TINY = 4

# Relaxed amalgamation for QR fronts.  Like CHOLMOD's rule a supernode is
# merged into the next one when that is its etree parent (so column ranges
# stay contiguous), and decisions are made top down so a parent can absorb a
# chain of trailing children.  The decision, though, uses the QR fronts
# themselves: the estimated front height (rows of B assigned to it plus the
# children's contribution blocks, assuming full rank) and width give the
# Householder work plus the stored R/V entries (weighted, since every solve
# streams them) of the separate and of the merged fronts.  Unlike
# Cholesky-count padding this sees that a front with few rows gains nothing
# from more pivot columns — its R rows only get wider — which is what wide
# and rank-limited problems produce.
function _qr_amalgamate(
        sstart::Vector{Int}, cnt::Vector{Int}, lrows::Vector{Int},
        parent::Vector{Int}, n::Int, maxsuper::Int
    )
    ns = length(sstart) - 1
    first_ = sstart[1:ns]
    last_ = [sstart[s + 1] - 1 for s in 1:ns]
    np = [sstart[s + 1] - sstart[s] for s in 1:ns]
    nu = [cnt[sstart[s + 1] - 1] - 1 for s in 1:ns]
    snodeof = Vector{Int}(undef, n)
    mf = zeros(Int, ns)
    @inbounds for s in 1:ns, j in sstart[s]:(sstart[s + 1] - 1)
        mf[s] += lrows[j]
        snodeof[j] = s
    end
    # front heights of the unmerged supernodes, bottom up (children first)
    cm = zeros(Int, ns)
    @inbounds for s in 1:ns
        cm[s] = min(max(mf[s] - np[s], 0), nu[s])
        p = parent[last_[s]]
        p != 0 && (mf[snodeof[p]] += cm[s])
    end
    # merge decisions top down, as in CHOLMOD's relaxed amalgamation, so a
    # parent can absorb a chain of trailing children
    alive = trues(ns)
    merged_into = collect(1:ns)
    @inbounds for s in (ns - 1):-1:1
        t = s + 1
        while !alive[t]
            t = merged_into[t]
        end
        first_[t] == last_[s] + 1 || continue
        pc = parent[last_[s]]
        (pc >= first_[t] && pc <= last_[t]) || continue
        npm = np[s] + np[t]
        npm <= maxsuper || continue
        mfm = mf[s] + mf[t] - cm[s]
        nfs = np[s] + nu[s]
        nft = np[t] + nu[t]
        nfm = npm + nu[t]
        sep = _front_cost(mf[s], nfs) + _front_cost(mf[t], nft) +
            _AMALG_BETA * (min(mf[s], nfs) * nfs + min(mf[t], nft) * nft)
        mrg = _front_cost(mfm, nfm) + _AMALG_BETA * min(mfm, nfm) * nfm
        (npm <= _AMALG_TINY || mrg <= sep + _AMALG_GAMMA) || continue
        first_[t] = first_[s]
        np[t] = npm
        mf[t] = mfm
        alive[s] = false
        merged_into[s] = t
    end
    out = Int[]
    @inbounds for s in 1:ns
        alive[s] && push!(out, first_[s])
    end
    push!(out, n + 1)
    return out
end

"""
    QRSymbolic

Result of the analysis phase for the multifrontal QR of `B` (`B = A`, or
`B = Aᴴ` when `transposed`): the column singletons, the final order `qf` of
the remaining (multifrontal) columns, their column elimination tree, the
(amalgamated) supernode partition, the assembly tree, and the assignment of
rows of `B` to fronts.  Front `s` owns multifrontal columns
`sstart[s]:sstart[s+1]-1` and additionally spans `cols[s]` — the columns
right of the pivot block that R's rows in this front reach.

Singleton `l` pivots row `srow[l]` on column `scol[l]` (entry `spos[l]` of
`nonzeros(A)`); in discovery order these form an upper-triangular block.
`colpos[c]` maps column `c` of `B` to its multifrontal position (> 0), to
`-l` for singleton `l`, or to 0 for the columns `ecol` that have no entries
outside the singleton rows (their solution components are zero).

Rows of `B` are accessed through `brp`/`bcol`/`bpos`: row `i` has its
entries in columns `bcol[brp[i]:brp[i+1]-1]` (as `colpos` values), stored at
positions `bpos[...]` of `nonzeros(A)` (conjugated when `transposed`), so a
numeric refactorization only re-reads values.
"""
struct QRSymbolic
    m::Int                      # rows of B
    n::Int                      # columns of B
    nmf::Int                    # multifrontal columns (n minus singletons)
    transposed::Bool            # B = Aᴴ (wide A: minimum-norm solves)
    srow::Vector{Int}           # singleton rows / columns / pivot positions
    scol::Vector{Int}
    spos::Vector{Int}
    ecol::Vector{Int}           # columns empty outside the singleton rows (x = 0)
    colpos::Vector{Int}         # column of B -> multifrontal position, -singleton, 0 if empty
    qf::Vector{Int}             # multifrontal column k is column qf[k] of B
    parent::Vector{Int}         # column elimination tree (multifrontal numbering)
    sstart::Vector{Int}         # front s pivots columns sstart[s]:sstart[s+1]-1
    cols::Vector{Vector{Int}}   # columns right of the pivot block in front s
    snof::Vector{Int}           # column -> front
    sparent::Vector{Int}        # assembly tree (0 = root)
    chp::Vector{Int}            # children of s: chld[chp[s]:chp[s+1]-1]
    chld::Vector{Int}
    frp::Vector{Int}            # rows of B assembled in s: fri[frp[s]:frp[s+1]-1]
    fri::Vector{Int}
    brp::Vector{Int}            # row access into B, see above
    bcol::Vector{Int}
    bpos::Vector{Int}
    mfest::Vector{Int}          # front heights assuming full rank
    maxnp::Int
    maxnu::Int
    # analysis options, kept so a refactorization can redo the analysis
    ordering::Symbol
    relax::Bool
    maxsuper::Int
    singletons::Bool
end

# Default dead-column tolerance, SPQR's rule 20 (m + n) eps max_j ‖B[:, j]‖.
function _default_tol(A::SparseMatrixCSC{Tv}, transposed::Bool) where {Tv}
    Tr = real(float(Tv))
    mA, nA = size(A)
    c2 = zeros(Tr, transposed ? mA : nA)
    Ai = rowvals(A)
    Ax = nonzeros(A)
    Ap = getcolptr(A)
    @inbounds for j in 1:nA, p in Ap[j]:(Ap[j + 1] - 1)
        c2[transposed ? Ai[p] : j] += abs2(Ax[p])
    end
    mx = isempty(c2) ? zero(Tr) : sqrt(maximum(c2))
    return 20 * (mA + nA) * eps(Tr) * mx
end

"""
    snqr_symbolic(A; ordering=:amd, relax=true, maxsuper=512, singletons=true,
                  transposed=false, tol=nothing) -> QRSymbolic

Analysis phase of the multifrontal QR of `B = A` (or `Aᴴ` when
`transposed`): column singletons (`B = A` only; pivots must exceed the
dead-column threshold `tol`, see [`snqr`](@ref)), fill-reducing column
ordering of the rest (`:amd` — minimum degree on the row cliques of `B`, no
BᴴB formed — or `:natural`), column elimination tree + postorder, column
counts, supernode detection with relaxed amalgamation, and the assembly tree
with each row of `B` assigned to the front of its leftmost column.
"""
function snqr_symbolic(
        A::SparseMatrixCSC; ordering::Symbol = :amd, relax::Bool = true,
        maxsuper::Int = 512, singletons::Bool = true, transposed::Bool = false,
        tol = nothing
    )
    ordering in (:amd, :natural) || throw(ArgumentError("unknown ordering $ordering"))
    mA, nA = size(A)
    Ap = Vector{Int}(getcolptr(A))
    Ai = Vector{Int}(rowvals(A))
    tp, tcol, tpos = _row_access(A)
    if transposed
        # B = Aᴴ: columns of B are rows of A, rows of B are columns of A
        mB, nB = nA, mA
        bcp, bci = tp, tcol
        brp, brc, brpos = Ap, Ai, collect(1:nnz(A))
    else
        mB, nB = mA, nA
        bcp, bci = Ap, Ai
        brp, brc, brpos = tp, tcol, tpos
    end
    # 1. singletons
    srow, scol, spos = if singletons && !transposed
        t = tol === nothing ? _default_tol(A, false) : tol
        _singletons(A, tp, tcol, t)
    else
        Int[], Int[], Int[]
    end
    nsing = length(scol)
    colpos = zeros(Int, nB)
    for (l, c) in enumerate(scol)
        colpos[c] = -l
    end
    rowalive = trues(mB)
    for i in srow
        rowalive[i] = false
    end
    # multifrontal columns, numbered 1:nmf in column order; columns left
    # without entries once the singleton rows are gone are structurally dead
    # (x = 0) and skip the multifrontal phase altogether
    mcol = Int[]
    ecol = Int[]
    cmap = zeros(Int, nB)
    @inbounds for c in 1:nB
        colpos[c] == 0 || continue
        live = false
        for p in bcp[c]:(bcp[c + 1] - 1)
            if rowalive[bci[p]]
                live = true
                break
            end
        end
        if live
            push!(mcol, c)
            cmap[c] = length(mcol)
        else
            push!(ecol, c)
        end
    end
    nmf = length(mcol)
    # pattern of the remaining block B22 (alive rows × multifrontal columns),
    # column- and row-wise; singleton rows are empty in it
    bcp2 = Vector{Int}(undef, nmf + 1)
    bci2 = Int[]
    bcp2[1] = 1
    @inbounds for k in 1:nmf
        c = mcol[k]
        for p in bcp[c]:(bcp[c + 1] - 1)
            rowalive[bci[p]] && push!(bci2, bci[p])
        end
        bcp2[k + 1] = length(bci2) + 1
    end
    brp2 = Vector{Int}(undef, mB + 1)
    brc2 = Int[]
    brp2[1] = 1
    @inbounds for i in 1:mB
        if rowalive[i]
            for p in brp[i]:(brp[i + 1] - 1)
                push!(brc2, cmap[brc[p]])
            end
        end
        brp2[i + 1] = length(brc2) + 1
    end
    # 2-3. ordering, etree, postorder (all in B22 column numbering)
    q = ordering === :amd ? _col_order(mB, nmf, bcp2, bci2, brp2, brc2) : collect(1:nmf)
    cp1, ri1 = _star_pattern(mB, nmf, brp2, brc2, invperm(q))
    parent1 = etree_sym(cp1, ri1, nmf)
    post = postorder_tree(parent1)
    qf2 = q[post]
    cpF, riF = permute_pattern(cp1, ri1, post, nmf)
    parentF = etree_sym(cpF, riF, nmf)
    # 4. counts, supernodes, per-supernode structure.  SupernodalLU's
    # supernode routines only take lengths of the column structures, so
    # ranges of the right lengths stand in for them.
    cnt = _col_counts(cpF, riF, parentF, nmf)
    lens = [Base.OneTo(c - 1) for c in cnt]
    sstart = fundamental_supernodes(parentF, lens, nmf)
    if relax
        # rows of B22 by leftmost column (final numbering): the rows each
        # front assembles directly
        pinv2 = invperm(qf2)
        lrows = zeros(Int, nmf)
        @inbounds for i in 1:mB
            brp2[i + 1] > brp2[i] || continue
            lm = typemax(Int)
            for p in brp2[i]:(brp2[i + 1] - 1)
                lm = min(lm, pinv2[brc2[p]])
            end
            lrows[lm] += 1
        end
        sstart = _qr_amalgamate(sstart, cnt, lrows, parentF, nmf, maxsuper)
    end
    ns = length(sstart) - 1
    snof = Vector{Int}(undef, nmf)
    @inbounds for s in 1:ns, j in sstart[s]:(sstart[s + 1] - 1)
        snof[j] = s
    end
    sparent = zeros(Int, ns)
    @inbounds for s in 1:ns
        p = parentF[sstart[s + 1] - 1]
        sparent[s] = p == 0 ? 0 : snof[p]
    end
    cols = _super_structure(cpF, riF, snof, sparent, ns, nmf)
    chp, chld = _children_csr(sparent)
    # final numbering: multifrontal position k is column qf[k] of B
    qf = mcol[qf2]
    @inbounds for k in 1:nmf
        colpos[qf[k]] = k
    end
    # 5. row access in final numbering + leftmost-column front assignment
    bcol = Vector{Int}(undef, length(brc))
    rfront = zeros(Int, mB)
    fcnt = zeros(Int, ns + 1)
    @inbounds for i in 1:mB
        lm = typemax(Int)
        for p in brp[i]:(brp[i + 1] - 1)
            c = colpos[brc[p]]
            bcol[p] = c
            c > 0 && (lm = min(lm, c))
        end
        if rowalive[i] && lm != typemax(Int)
            rfront[i] = snof[lm]
            fcnt[snof[lm] + 1] += 1
        end
    end
    fcnt[1] = 1
    @inbounds for s in 1:ns
        fcnt[s + 1] += fcnt[s]
    end
    frp = copy(fcnt)
    cur = fcnt[1:ns]
    fri = Vector{Int}(undef, frp[ns + 1] - 1)
    @inbounds for i in 1:mB
        s = rfront[i]
        s == 0 && continue
        fri[cur[s]] = i
        cur[s] += 1
    end
    # front heights assuming no dead columns (children precede parents)
    mfest = zeros(Int, ns)
    cm = zeros(Int, ns)
    maxnp = 0
    maxnu = 0
    @inbounds for s in 1:ns
        np = sstart[s + 1] - sstart[s]
        nu = length(cols[s])
        mf = frp[s + 1] - frp[s]
        for k in chp[s]:(chp[s + 1] - 1)
            mf += cm[chld[k]]
        end
        mfest[s] = mf
        cm[s] = mf <= np ? 0 : min(mf - np, nu)
        maxnp = max(maxnp, np)
        maxnu = max(maxnu, nu)
    end
    return QRSymbolic(
        mB, nB, nmf, transposed, srow, scol, spos, ecol, colpos, qf, parentF, sstart,
        cols, snof, sparent, chp, chld, frp, fri, brp, bcol, brpos, mfest,
        maxnp, maxnu, ordering, relax, maxsuper, singletons
    )
end
