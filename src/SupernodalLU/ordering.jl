# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# Fill-reducing orderings on the symmetric pattern of A + Aᵀ.
#
# The method's reference implementations default to nested dissection
# (METIS), with minimum degree as the fallback.  Here both are pure Julia: `amd_order_sym` drives the
# vendored SuiteSparse-AMD port, and `nd_order_sym` is a BFS-bisection nested
# dissection (level-set separators, pseudo-peripheral roots) that switches to
# AMD on small subgraphs.

# Off-diagonal symmetric pattern of A + Aᵀ as 1-based CSC (sorted row
# indices).  Three O(nnz) passes, no sparse addition and no comparison sort:
# (1) bucket-transpose A's pattern, (2) merge col j of A with row j of A
# under a last-claimant marker (dedup, unsorted), (3) one more bucket pass —
# a transpose of a symmetric pattern — which sorts every column.
function sym_pattern(A::SparseMatrixCSC)
    n = size(A, 2)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    nz = nnz(A)
    # (1) pattern transpose of A
    tp = zeros(Int, n + 1)
    @inbounds for p in 1:nz
        tp[Ai[p] + 1] += 1
    end
    tp[1] = 1
    @inbounds for i in 1:n
        tp[i + 1] += tp[i]
    end
    tcur = tp[1:n]
    ti = Vector{Int}(undef, nz)
    @inbounds for j in 1:n
        for p in Ap[j]:(Ap[j + 1] - 1)
            i = Ai[p]
            ti[tcur[i]] = j
            tcur[i] = tcur[i] + 1
        end
    end
    # (2) per-column union of A[:,j] and Aᵀ[:,j], diagonal dropped
    mark = zeros(Int, n)
    cpU = Vector{Int}(undef, n + 1)
    riU = Vector{Int}(undef, 2 * nz)
    k = 0
    cpU[1] = 1
    @inbounds for j in 1:n
        mark[j] = j
        for p in Ap[j]:(Ap[j + 1] - 1)
            i = Ai[p]
            if mark[i] != j
                mark[i] = j
                k += 1
                riU[k] = i
            end
        end
        for p in tp[j]:(tp[j + 1] - 1)
            i = ti[p]
            if mark[i] != j
                mark[i] = j
                k += 1
                riU[k] = i
            end
        end
        cpU[j + 1] = k + 1
    end
    resize!(riU, k)
    # (3) sort all columns at once: transpose pass on the symmetric pattern
    return permute_pattern(cpU, riU, collect(1:n), n)
end

# AMD on an off-diagonal symmetric 1-based pattern.  The vendored port mirrors
# amd_order.c, so indices are converted to 0-based around the call.
function _amd_on_pattern(cp::Vector{Int}, ri::Vector{Int}, n::Int)
    n == 0 && return Int[]
    Ap0 = Vector{Int}(undef, n + 1)
    @inbounds for j in 1:(n + 1)
        Ap0[j] = cp[j] - 1
    end
    Ai0 = Vector{Int}(undef, length(ri))
    @inbounds for p in eachindex(ri)
        Ai0[p] = ri[p] - 1
    end
    P0 = Vector{Int}(undef, n)
    status, _ = AMD.amd_order!(n, Ap0, Ai0, P0)
    status >= 0 || error("AMD ordering failed with status $status")
    return P0 .+ 1
end

"""
    amd_order_sym(A) -> perm

Approximate-minimum-degree ordering of the symmetric pattern of `A + Aᵀ`
(pure Julia, SuiteSparse-AMD port).  `perm[k]` is the original index eliminated
at step `k`.
"""
function amd_order_sym(A::SparseMatrixCSC)
    cp, ri = sym_pattern(A)
    return _amd_on_pattern(cp, ri, size(A, 2))
end

# ---------------------------------------------------------------------------
# Nested dissection
# ---------------------------------------------------------------------------

# BFS over the subgraph `verts` (global ids, membership given by sub[v] != 0),
# starting from `root`.  Fills `level[v]` (1-based level, global array reused
# across calls via the `mark` epoch) and returns (order of visit, nlevels,
# last vertex visited).  Unreached vertices of the component are not our
# problem — the caller splits per connected component first.
function _bfs!(
        cp::Vector{Int}, ri::Vector{Int}, root::Int,
        sub::Vector{Int}, level::Vector{Int}, queue::Vector{Int}
    )
    head = 1
    tail = 1
    queue[1] = root
    level[root] = 1
    nvis = 1
    nlev = 1
    @inbounds while head <= tail
        v = queue[head]
        head += 1
        lv = level[v]
        for p in cp[v]:(cp[v + 1] - 1)
            w = ri[p]
            if sub[w] != 0 && level[w] == 0
                level[w] = lv + 1
                nlev = max(nlev, lv + 1)
                tail += 1
                queue[tail] = w
                nvis += 1
            end
        end
    end
    return nvis, nlev, queue[tail]
end

# One nested-dissection split of the subgraph `verts`: BFS from a
# pseudo-peripheral root, cut at the level whose cumulative size is closest to
# half, take that whole level set as the separator.  Appends `part1`, `part2`,
# `sep` (global ids).  Vertices are removed from `sub` membership as they are
# classified.
function _nd_order!(
        perm::Vector{Int}, pk::Base.RefValue{Int},
        cp::Vector{Int}, ri::Vector{Int},
        verts::Vector{Int}, sub::Vector{Int}, level::Vector{Int},
        queue::Vector{Int}, leafsize::Int, depth::Int
    )
    nv = length(verts)
    if nv == 0
        return nothing
    end
    if nv <= leafsize || depth > 64
        _nd_leaf!(perm, pk, cp, ri, verts, sub)
        return nothing
    end
    for v in verts
        level[v] = 0
    end
    # BFS may not reach the whole vertex set (disconnected subgraph): peel one
    # component at a time.
    root = verts[1]
    nvis, nlev, far = _bfs!(cp, ri, root, sub, level, queue)
    if nvis < nv
        comp = Int[]
        rest = Int[]
        for v in verts
            (level[v] != 0 ? comp : rest) |> c -> push!(c, v)
        end
        _nd_order!(perm, pk, cp, ri, comp, sub, level, queue, leafsize, depth)
        _nd_order!(perm, pk, cp, ri, rest, sub, level, queue, leafsize, depth)
        return nothing
    end
    # pseudo-peripheral: restart BFS from the farthest vertex found
    if nlev > 2
        for v in verts
            level[v] = 0
        end
        _, nlev, _ = _bfs!(cp, ri, far, sub, level, queue)
    end
    if nlev <= 2
        # graph is (near-)complete from this root; no useful separator
        _nd_leaf!(perm, pk, cp, ri, verts, sub)
        return nothing
    end
    # cumulative level sizes -> split level (levels 1..cut-1 | cut | cut+1..)
    lsz = zeros(Int, nlev)
    for v in verts
        lsz[level[v]] += 1
    end
    half = nv ÷ 2
    acc = 0
    cut = 2
    for l in 1:(nlev - 1)
        acc += lsz[l]
        if acc >= half
            cut = max(2, min(l + 1, nlev - 1))
            break
        end
    end
    part1 = Int[]
    part2 = Int[]
    sep = Int[]
    for v in verts
        lv = level[v]
        if lv < cut
            push!(part1, v)
        elseif lv > cut
            push!(part2, v)
        else
            push!(sep, v)
        end
    end
    # order both parts first (they are independent once sep is removed), then
    # the separator last — the ND elimination order.
    for v in sep
        sub[v] = 0
    end
    _nd_order!(perm, pk, cp, ri, part1, sub, level, queue, leafsize, depth + 1)
    _nd_order!(perm, pk, cp, ri, part2, sub, level, queue, leafsize, depth + 1)
    for v in sep                      # restore membership for the leaf order
        sub[v] = 1
    end
    _nd_leaf!(perm, pk, cp, ri, sep, sub)
    return nothing
end

# Order a leaf subgraph with AMD on its induced pattern and emit into perm.
function _nd_leaf!(
        perm::Vector{Int}, pk::Base.RefValue{Int},
        cp::Vector{Int}, ri::Vector{Int}, verts::Vector{Int}, sub::Vector{Int}
    )
    nv = length(verts)
    nv == 0 && return nothing
    if nv <= 2
        for v in verts
            perm[pk[]] = v
            pk[] += 1
            sub[v] = 0
        end
        return nothing
    end
    lid = Dict{Int, Int}()
    for (k, v) in enumerate(verts)
        lid[v] = k
    end
    cpl = Vector{Int}(undef, nv + 1)
    ril = Int[]
    cpl[1] = 1
    for (k, v) in enumerate(verts)
        for p in cp[v]:(cp[v + 1] - 1)
            w = ri[p]
            if sub[w] != 0 && haskey(lid, w)
                push!(ril, lid[w])
            end
        end
        cpl[k + 1] = length(ril) + 1
    end
    p = _amd_on_pattern(cpl, ril, nv)
    for k in p
        perm[pk[]] = verts[k]
        pk[] += 1
    end
    for v in verts
        sub[v] = 0
    end
    return nothing
end

"""
    nd_order_sym(A; leafsize=200) -> perm

Nested-dissection ordering of the symmetric pattern of `A + Aᵀ` (pure Julia):
recursive BFS-bisection with level-set separators, AMD on subgraphs of at most
`leafsize` vertices.  This mirrors the method's preferred reordering
strategy (nested dissection), traded down from METIS-quality separators for
dependency-free simplicity.
"""
function nd_order_sym(A::SparseMatrixCSC; leafsize::Int = 200)
    n = size(A, 2)
    cp, ri = sym_pattern(A)
    perm = Vector{Int}(undef, n)
    pk = Ref(1)
    sub = ones(Int, n)
    level = zeros(Int, n)
    queue = Vector{Int}(undef, n)
    _nd_order!(perm, pk, cp, ri, collect(1:n), sub, level, queue, leafsize, 1)
    pk[] == n + 1 || error("nested dissection dropped vertices (internal error)")
    return perm
end
