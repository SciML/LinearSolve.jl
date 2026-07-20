# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# Symbolic analysis for the supernodal factorization, Schenk–Gärtner-style: everything
# runs on the symmetric pattern of A + Aᵀ, so L and Uᵀ share one structure and
# the numeric phase needs no dynamic fill.
#
# Pipeline:
#   1. fill-reducing order (AMD or nested dissection)     src/ordering.jl
#   2. elimination tree of the permuted pattern + postorder -> final order qf
#   3. per-column structure of L (row indices below the diagonal)
#   4. fundamental supernodes + relaxed amalgamation (CHOLMOD-style rule)
#
# The algorithms are implemented from their published descriptions: the
# elimination tree via union-find with path halving (Liu 1990; Tarjan-style
# compression), postorder by an explicit two-stack DFS over counting-sorted
# child arrays, and L's column structure by the row-subtree characterization
# of Gilbert–Ng–Peyton (1994): row i appears in exactly the columns on the
# etree paths from i's below-diagonal neighbours up toward i.  No per-column
# sorting is ever needed — every structure comes out ordered by construction.

# Elimination tree of a symmetric off-diagonal pattern.  For each vertex k
# (ascending) and each neighbour i < k, climb from i to the current root of
# i's partially-built subtree and attach it under k.  The climb uses
# path-halving union-find (`uf[x]` points toward the subtree root), so the
# total cost is effectively O(nnz · α).  parent[k] == 0 marks a root.
function etree_sym(cp::Vector{Int}, ri::Vector{Int}, n::Int)
    parent = zeros(Int, n)
    uf = collect(1:n)                     # union-find: x -> toward its root
    @inbounds for k in 1:n
        for p in cp[k]:(cp[k + 1] - 1)
            i = ri[p]
            i < k || continue
            # find root of i's subtree with path halving
            r = i
            while uf[r] != r
                uf[r] = uf[uf[r]]
                r = uf[r]
            end
            if r != k
                parent[r] = k
                uf[r] = k                  # union into k's growing subtree
            end
        end
        uf[k] = k
    end
    return parent
end

# Children of every node as a CSR-style array pair (counting sort by parent):
# children of v are chld[chp[v]:chp[v+1]-1], in ascending order.
function _children_csr(parent::Vector{Int})
    n = length(parent)
    chp = zeros(Int, n + 2)
    @inbounds for v in 1:n
        p = parent[v]
        p != 0 && (chp[p + 2] += 1)
    end
    chp[1] = 1
    chp[2] = 1
    @inbounds for v in 1:n
        chp[v + 2] += chp[v + 1]
    end
    chld = Vector{Int}(undef, chp[n + 2] - 1)
    @inbounds for v in 1:n                # ascending v ⇒ children stay sorted
        p = parent[v]
        if p != 0
            chld[chp[p + 1]] = v
            chp[p + 1] += 1
        end
    end
    return chp, chld                       # use chp[v]:chp[v+1]-1 after shift
end

# Depth-first postorder of the forest `parent`, children visited in ascending
# order.  Explicit node stack + child-cursor stack over the CSR child arrays
# (nothing is mutated, no sibling links).
function postorder_tree(parent::Vector{Int})
    n = length(parent)
    chp, chld = _children_csr(parent)
    post = Vector{Int}(undef, n)
    nstack = Vector{Int}(undef, n)
    cstack = Vector{Int}(undef, n)
    k = 0
    @inbounds for r in 1:n
        parent[r] == 0 || continue
        top = 1
        nstack[1] = r
        cstack[1] = chp[r]
        while top > 0
            v = nstack[top]
            c = cstack[top]
            if c < chp[v + 1]
                cstack[top] = c + 1
                top += 1
                nstack[top] = chld[c]
                cstack[top] = chp[chld[c]]
            else
                k += 1
                post[k] = v
                top -= 1
            end
        end
    end
    return post
end

# Pattern of B[perm, perm] for an off-diagonal symmetric pattern, produced
# with sorted columns by a single bucket pass: because the pattern is
# symmetric, emitting entry (min-side) buckets in ascending destination-row
# order is exactly a transpose pass, which sorts every column for free — no
# comparison sort anywhere.
function permute_pattern(cp::Vector{Int}, ri::Vector{Int}, perm::Vector{Int}, n::Int)
    pinv = invperm(perm)
    nz = length(ri)
    cpN = zeros(Int, n + 1)
    @inbounds for j in 1:n                # new-column sizes (symmetric: |col|
        cpN[pinv[j] + 1] = cp[j + 1] - cp[j]   # is permutation-invariant)
    end
    cpN[1] = 1
    @inbounds for j in 1:n
        cpN[j + 1] += cpN[j]
    end
    cursor = cpN[1:n]
    riN = Vector{Int}(undef, nz)
    # walk old columns in the order of their new ROW index; scattering entry
    # (i,j) -> (pinv[i], pinv[j]) into bucket pinv[j] then fills each new
    # column's rows in ascending order
    @inbounds for inew in 1:n
        iold = perm[inew]
        for p in cp[iold]:(cp[iold + 1] - 1)
            jnew = pinv[ri[p]]
            riN[cursor[jnew]] = inew
            cursor[jnew] += 1
        end
    end
    return cpN, riN
end

# Row structure of each column of L (strictly below the diagonal) for a
# postordered pattern, by row subtrees (Gilbert–Ng–Peyton): row i's columns
# are the nodes visited climbing the etree from each neighbour j < i of
# vertex i, stopping at nodes already claimed for row i.  Scanning i in
# ascending order appends rows to each column list in ascending order, so
# the structures come out sorted with no sorting step.
function col_structure(cp::Vector{Int}, ri::Vector{Int}, parent::Vector{Int}, n::Int)
    colstruct = [Int[] for _ in 1:n]
    claimed = zeros(Int, n)               # last row that claimed this column
    @inbounds for i in 1:n
        claimed[i] = i                    # the climb for row i stops before i
        for p in cp[i]:(cp[i + 1] - 1)
            j = ri[p]
            j < i || continue
            while claimed[j] != i         # climb toward i, claiming columns
                push!(colstruct[j], i)
                claimed[j] = i
                j = parent[j]
                j == 0 && break
            end
        end
    end
    return colstruct
end

"""
    SymbolicAnalysis

Result of the analysis phase: final elimination order `qf`, elimination tree,
the (amalgamated) supernode partition, and the static left-looking update
schedule.  `rows[s]` is the update-row set of supernode `s` — the row indices
of its panel below the pivot block, in the `qf` numbering.

The schedule (`ccp`/`cd`/`cj1`, CSR over target supernodes) lists, for every
target supernode `s`, the descendants `d` whose update rows intersect `s`'s
columns and the offset `j1` into `rows[d]` where that intersection starts.
It is purely structural, so both the serial and the threaded numeric phases
apply contributions in the same (ascending-descendant) order — same pivot
sequence and fill — and refactorization needs no per-factor list bookkeeping.
"""
struct SymbolicAnalysis
    n::Int
    qf::Vector{Int}          # final order: column k of the factor is A[:, qf[k]]
    parent::Vector{Int}      # elimination tree (qf numbering)
    sstart::Vector{Int}      # supernode s owns columns sstart[s]:sstart[s+1]-1
    rows::Vector{Vector{Int}}  # update rows (> last column) of each supernode
    snof::Vector{Int}        # column -> supernode
    nnzL::Int                # entries in L panels incl. unit diagonal & padding
    ccp::Vector{Int}         # contribution schedule: target s gets ccp[s]:ccp[s+1]-1
    cd::Vector{Int}          # ... from descendant cd[k]
    cj1::Vector{Int}         # ... starting at rows[cd[k]][cj1[k]]
end

# Static left-looking contribution schedule: split each supernode's update-row
# list into runs by owning target supernode.
function _contrib_schedule(
        sstart::Vector{Int}, rows::Vector{Vector{Int}}, snof::Vector{Int}
    )
    nsuper = length(sstart) - 1
    cnt = zeros(Int, nsuper)
    @inbounds for d in 1:nsuper
        R = rows[d]
        j = 1
        while j <= length(R)
            t = snof[R[j]]
            cnt[t] += 1
            c2t = sstart[t + 1] - 1
            while j <= length(R) && R[j] <= c2t
                j += 1
            end
        end
    end
    ccp = Vector{Int}(undef, nsuper + 1)
    ccp[1] = 1
    @inbounds for s in 1:nsuper
        ccp[s + 1] = ccp[s] + cnt[s]
    end
    ntot = ccp[nsuper + 1] - 1
    cd = Vector{Int}(undef, ntot)
    cj1 = Vector{Int}(undef, ntot)
    cur = ccp[1:nsuper]
    @inbounds for d in 1:nsuper
        R = rows[d]
        j = 1
        while j <= length(R)
            t = snof[R[j]]
            k = cur[t]
            cur[t] = k + 1
            cd[k] = d
            cj1[k] = j
            c2t = sstart[t + 1] - 1
            while j <= length(R) && R[j] <= c2t
                j += 1
            end
        end
    end
    return ccp, cd, cj1
end

# Largest half-bandwidth of the symmetric pattern, or -1 as soon as it exceeds
# `bwmax` (early abort — banded detection only needs narrow bands).
function _pattern_bandwidth(cp::Vector{Int}, ri::Vector{Int}, n::Int, bwmax::Int)
    bw = 0
    @inbounds for j in 1:n
        for p in cp[j]:(cp[j + 1] - 1)
            d = abs(ri[p] - j)
            if d > bw
                d > bwmax && return -1
                bw = d
            end
        end
    end
    return bw
end

# Fundamental supernodes: column j extends the current supernode iff j-1 is
# its only child in the etree and struct(j-1) = {j} ∪ struct(j).
function fundamental_supernodes(
        parent::Vector{Int}, colstruct::Vector{Vector{Int}}, n::Int
    )
    nchild = zeros(Int, n)
    @inbounds for j in 1:n
        p = parent[j]
        p != 0 && (nchild[p] += 1)
    end
    sstart = Int[]
    @inbounds for j in 1:n
        extend = j > 1 && parent[j - 1] == j && nchild[j] == 1 &&
            length(colstruct[j - 1]) == length(colstruct[j]) + 1
        extend || push!(sstart, j)
    end
    push!(sstart, n + 1)
    return sstart
end

# Relaxed amalgamation à la CHOLMOD: repeatedly merge a supernode into the
# next one when the next one is its etree parent supernode and starts at the
# following column, if the merged panel stays small or introduces few explicit
# zeros.  For such adjacent parent merges the merged update-row set is exactly
# the parent's update-row set, which keeps the bookkeeping O(1) per attempt.
function amalgamate(
        sstart::Vector{Int}, colstruct::Vector{Vector{Int}},
        parent::Vector{Int}, n::Int;
        nrelax0::Int = 8, nrelax1::Int = 32, nrelax2::Int = 96,
        zrelax0::Float64 = 0.8, zrelax1::Float64 = 0.2, zrelax2::Float64 = 0.05,
        maxsuper::Int = 512
    )
    ns = length(sstart) - 1
    # per (current) supernode, in column order
    first_ = [sstart[s] for s in 1:ns]
    last_ = [sstart[s + 1] - 1 for s in 1:ns]
    nrows = [length(colstruct[sstart[s + 1] - 1]) for s in 1:ns]  # |update rows|
    nzero = zeros(Int, ns)          # explicit zeros accumulated in the panel
    total = Vector{Int}(undef, ns)  # panel entries (L side, incl. pivot block)
    @inbounds for s in 1:ns
        np = last_[s] - first_[s] + 1
        total[s] = np * np + np * nrows[s]     # full square pivot block + rect
    end
    merged_into = collect(1:ns)     # union-find-ish forward pointer
    alive = trues(ns)
    @inbounds for s in (ns - 1):-1:1
        t = s + 1
        while !alive[t]
            t = merged_into[t]
        end
        # merge candidate: t must start right after s and be s's etree parent
        # supernode (parent of s's last column lies in t's column range)
        first_[t] == last_[s] + 1 || continue
        pc = parent[last_[s]]
        (pc >= first_[t] && pc <= last_[t]) || continue
        npS = last_[s] - first_[s] + 1
        npT = last_[t] - first_[t] + 1
        npM = npS + npT
        npM <= maxsuper || continue
        # merged panel: columns first_[s]:last_[t], update rows = t's update
        # rows; per-column heights for s's columns grow accordingly
        totM = total[t] + npS * (npM + nrows[t])
        zM = nzero[s] + nzero[t] + (npS * (npM + nrows[t]) - total[s])
        z = zM / max(totM, 1)
        ok = npM <= nrelax0 ||
            (npM <= nrelax1 && z <= zrelax0) ||
            (npM <= nrelax2 && z <= zrelax1) ||
            z <= zrelax2
        ok || continue
        # merge s into t (t keeps its identity; ranges/zeros absorb s)
        first_[t] = first_[s]
        nzero[t] = zM
        total[t] = totM
        alive[s] = false
        merged_into[s] = t
    end
    out = Int[]
    @inbounds for s in 1:ns
        alive[s] && push!(out, first_[s])
    end
    sort!(out)
    push!(out, n + 1)
    return out
end

"""
    snlu_symbolic(A; ordering=:amd, relax=true, maxsuper=512) -> SymbolicAnalysis

Analysis phase: fill-reducing ordering (`:amd`, `:nd`, or `:natural`),
elimination tree + postorder, supernode detection with relaxed amalgamation,
and the static left-looking contribution schedule.  With the default `:amd`
ordering, matrices whose symmetric pattern is a densely-populated narrow band
skip AMD and keep the natural ordering (banded fast path — the natural order
is already fill-optimal there).
"""
function snlu_symbolic(
        A::SparseMatrixCSC; ordering::Symbol = :amd, relax::Bool = true,
        maxsuper::Int = 512
    )
    n = size(A, 2)
    size(A, 1) == n || throw(DimensionMismatch("matrix must be square"))
    cp, ri = sym_pattern(A)
    q = if ordering === :amd
        # banded fast path: half-bandwidth bw with n >= 4(bw+1) and the band
        # mostly populated -> natural ordering is fill-optimal, skip AMD
        bw = n >= 8 ? _pattern_bandwidth(cp, ri, n, max(n ÷ 4 - 1, 0)) : -1
        if bw >= 0 && 2 * (length(ri) + n) >= n * (2 * bw + 1)
            collect(1:n)
        else
            _amd_on_pattern(cp, ri, n)
        end
    elseif ordering === :nd
        nd_order_sym(A)
    elseif ordering === :natural
        collect(1:n)
    else
        throw(ArgumentError("unknown ordering $ordering"))
    end
    cp1, ri1 = permute_pattern(cp, ri, q, n)
    parent1 = etree_sym(cp1, ri1, n)
    post = postorder_tree(parent1)
    qf = q[post]
    cpF, riF = permute_pattern(cp1, ri1, post, n)
    parentF = etree_sym(cpF, riF, n)
    colstruct = col_structure(cpF, riF, parentF, n)
    sstart = fundamental_supernodes(parentF, colstruct, n)
    if relax
        sstart = amalgamate(sstart, colstruct, parentF, n; maxsuper = maxsuper)
    end
    ns = length(sstart) - 1
    rows = Vector{Vector{Int}}(undef, ns)
    snof = Vector{Int}(undef, n)
    nnzL = 0
    @inbounds for s in 1:ns
        c1 = sstart[s]
        c2 = sstart[s + 1] - 1
        rows[s] = colstruct[c2]
        np = c2 - c1 + 1
        nnzL += np * np + np * length(rows[s])
        for j in c1:c2
            snof[j] = s
        end
    end
    ccp, cd, cj1 = _contrib_schedule(sstart, rows, snof)
    return SymbolicAnalysis(n, qf, parentF, sstart, rows, snof, nnzL, ccp, cd, cj1)
end
