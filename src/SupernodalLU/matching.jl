# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# Maximum-weighted (max-product) bipartite matching with dual-variable
# scalings — the MC64 job-5 preprocessing the Schenk–Gärtner method applies
# to unsymmetric matrices (Duff & Koster 2001; Olschowka & Neumaier 1996).  Implemented as
# shortest augmenting paths with potentials (sparse Jonker–Volgenant style),
# written fresh from the published algorithm.
#
# Finds a row permutation σ maximizing ∏ |A[σ(j), j]| and scalings r, c such
# that B = diag(r)·A·diag(c) permuted by σ has |B[σ(j),j]| = 1 and all
# |B[i,j]| ≤ 1.  Factorizing that matrix makes static pivoting almost never
# fire, which is exactly why the method turns matching on by default for
# general unsymmetric systems.

struct MatchScale
    rowperm::Vector{Int}   # σ: column j is matched to row rowperm[j]
    r::Vector{Float64}     # row scalings
    c::Vector{Float64}     # column scalings
    ok::Bool               # false: structurally singular, fall back to identity
end

function _identity_matchscale(n::Int)
    return MatchScale(collect(1:n), ones(n), ones(n), false)
end

"""
    mc64_matching(A) -> MatchScale

Max-product matching + scaling of sparse square `A`.  On structurally
singular matrices returns the identity assignment with `ok = false`.
"""
function mc64_matching(A::SparseMatrixCSC{Tv}) where {Tv}
    n = size(A, 2)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    Ax = nonzeros(A)
    nz = nnz(A)

    # entry costs: w = log(colmax / |a|)  (>= 0, 0 for the largest entry).
    # Magnitudes go through `_costabs`, a Float64 barrier that keeps the
    # combinatorial side of the matching in Float64 for any value type
    # (overloaded for ForwardDiff.Dual in the package extension).
    w = Vector{Float64}(undef, nz)
    logcmax = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        cm = 0.0
        for p in Ap[j]:(Ap[j + 1] - 1)
            a = _costabs(Ax[p])
            a > cm && (cm = a)
        end
        (cm == 0 || !isfinite(cm)) && return _identity_matchscale(n)
        logcmax[j] = log(cm)
        for p in Ap[j]:(Ap[j + 1] - 1)
            a = _costabs(Ax[p])
            w[p] = a == 0 ? Inf : logcmax[j] - log(a)
        end
    end

    u = zeros(n)                   # column potentials
    v = zeros(n)                   # row potentials
    colmatch = zeros(Int, n)       # column j -> matched row
    rowmatch = zeros(Int, n)       # row i -> matched column

    # cheap initial assignment: zero-reduced-cost entries
    @inbounds for j in 1:n
        best = Inf
        for p in Ap[j]:(Ap[j + 1] - 1)
            w[p] < best && (best = w[p])
        end
        u[j] = best
    end
    @inbounds for j in 1:n
        for p in Ap[j]:(Ap[j + 1] - 1)
            i = Ai[p]
            if rowmatch[i] == 0 && w[p] - u[j] <= 0.0 + 1.0e-14
                rowmatch[i] = j
                colmatch[j] = i
                break
            end
        end
    end

    # shortest augmenting path for every unmatched column
    d = fill(Inf, n)               # tentative distance to each row
    pred = zeros(Int, n)           # predecessor column on the path to row i
    done = falses(n)
    touched = Int[]
    heap = Vector{Tuple{Float64, Int}}()   # (dist, row) — lazy-deletion binheap

    @inbounds for j0 in 1:n
        colmatch[j0] != 0 && continue
        empty!(heap)
        for i in touched
            d[i] = Inf
            done[i] = false
            pred[i] = 0
        end
        empty!(touched)
        jcur = j0
        dcur = 0.0
        isink = 0
        dstar = Inf
        while true
            for p in Ap[jcur]:(Ap[jcur + 1] - 1)
                i = Ai[p]
                done[i] && continue
                isfinite(w[p]) || continue
                dnew = dcur + (w[p] - u[jcur] - v[i])
                if dnew < d[i] - 1.0e-15
                    d[i] == Inf && push!(touched, i)
                    d[i] = dnew
                    pred[i] = jcur
                    _heap_push!(heap, (dnew, i))
                end
            end
            imin = 0
            while !isempty(heap)
                dm, im = _heap_pop!(heap)
                if !done[im] && dm <= d[im] + 1.0e-15
                    imin = im
                    break
                end
            end
            imin == 0 && break                 # no augmenting path
            done[imin] = true                  # already on `touched` (d was set)
            if rowmatch[imin] == 0
                isink = imin
                dstar = d[imin]
                break
            end
            dcur = d[imin]
            jcur = rowmatch[imin]
        end
        isink == 0 && return _identity_matchscale(n)
        # dual updates for finalized rows and their matched columns
        for i in touched
            if done[i] && i != isink
                v[i] += d[i] - dstar
                u[rowmatch[i]] += dstar - d[i]
            end
        end
        u[j0] += dstar
        # augment along the predecessor chain
        i = isink
        while true
            j = pred[i]
            inext = colmatch[j]
            colmatch[j] = i
            rowmatch[i] = j
            j == j0 && break
            i = inext
        end
    end

    # scalings: r_i = exp(v_i), c_j = exp(u_j)/colmax_j gives |B| <= 1 with
    # ones on the matched diagonal
    r = Vector{Float64}(undef, n)
    c = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        r[i] = exp(clamp(v[i], -300.0, 300.0))
    end
    @inbounds for j in 1:n
        c[j] = exp(clamp(u[j] - logcmax[j], -300.0, 300.0))
    end
    rowperm = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        rowperm[j] = colmatch[j]
    end
    return MatchScale(rowperm, r, c, true)
end

# minimal binary min-heap on (dist, row) tuples with lazy deletion
@inline function _heap_push!(h::Vector{Tuple{Float64, Int}}, x::Tuple{Float64, Int})
    push!(h, x)
    k = length(h)
    @inbounds while k > 1
        p = k >> 1
        h[p][1] <= h[k][1] && break
        h[p], h[k] = h[k], h[p]
        k = p
    end
    return nothing
end

@inline function _heap_pop!(h::Vector{Tuple{Float64, Int}})
    @inbounds top = h[1]
    @inbounds h[1] = h[end]
    pop!(h)
    k = 1
    m = length(h)
    @inbounds while true
        l = 2k
        l > m && break
        c = (l < m && h[l + 1][1] < h[l][1]) ? l + 1 : l
        h[k][1] <= h[c][1] && break
        h[k], h[c] = h[c], h[k]
        k = c
    end
    return top
end

# Float64 view of |x| for the matching's combinatorial side.  Value types
# with no exact Float64 conversion (e.g. ForwardDiff.Dual) get an overload in
# the package extension that extracts the primal value.
@inline _costabs(x::Number) = Float64(abs(x))

# Quick check whether matching is worth it: any missing or relatively tiny
# structural diagonal (the situations where restricted pivoting on the raw
# matrix would have to perturb).  Runs in the native |Tv| arithmetic so it
# stays generic over value types.
function needs_matching(A::SparseMatrixCSC{Tv}) where {Tv}
    n = size(A, 2)
    Ap = getcolptr(A)
    Ai = rowvals(A)
    Ax = nonzeros(A)
    z = abs(zero(Tv))
    @inbounds for j in 1:n
        cm = z
        dj = z
        hasdiag = false
        for p in Ap[j]:(Ap[j + 1] - 1)
            a = abs(Ax[p])
            a > cm && (cm = a)
            if Ai[p] == j
                hasdiag = true
                dj = a
            end
        end
        iszero(cm) && continue
        (!hasdiag || dj < 1.0e-2 * cm) && return true
    end
    return false
end
