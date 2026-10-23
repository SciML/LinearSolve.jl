# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# High-level API: snqr / snqr! and factor extraction.

"""
    snqr(A; ordering=:amd, tol=nothing, wide=:basic, singletons=true,
         relax=true, maxsuper=512) -> SupernodalQRFactor

Multifrontal sparse Householder QR of `A` (any shape), giving least-squares
solutions.  By default `A` itself is factored: column singletons are peeled
off first, and the rest is factored as `A22[:, F.q] = Q R`; on wide or rank
deficient `A` the solution is a basic one (as SPQR's).

- `ordering` ∈ (`:amd`, `:natural`) — fill-reducing column ordering
  (minimum degree on the row cliques of `A`; AᴴA is never formed).
- `tol` — Heath dead-column threshold: pivot columns whose remaining 2-norm
  is at most `tol` are dropped (solution component zero); singleton pivots
  must exceed it too.  `nothing` (default) uses
  `20 (m + n) eps max_j ‖A[:, j]‖`; a negative value disables rank detection.
- `wide` — `:basic` (default) or `:minnorm`: for wide `A`, factor `Aᴴ`
  instead and return the minimum-norm least-squares solution.  This pays for
  the fill of factoring AAᴴ, which can be far more than factoring `A`.
- `singletons` — peel off column singletons before the multifrontal phase.
- `relax`, `maxsuper` — supernode amalgamation controls.
"""
function snqr(
        A::SparseMatrixCSC{Tv, Ti}; ordering::Symbol = :amd, tol = nothing,
        wide::Symbol = :basic, singletons::Bool = true, relax::Bool = true,
        maxsuper::Int = 512
    ) where {Tv, Ti <: Integer}
    wide in (:basic, :minnorm) || throw(ArgumentError("wide must be :basic or :minnorm, got $wide"))
    transposed = wide === :minnorm && size(A, 1) < size(A, 2)
    sym = snqr_symbolic(
        A; ordering = ordering, relax = relax, maxsuper = maxsuper,
        singletons = singletons, transposed = transposed, tol = tol
    )
    return snqr(sym, A; tol = tol)
end

"""
    snqr(sym::QRSymbolic, A; tol=nothing) -> SupernodalQRFactor

Numeric factorization reusing an analysis from [`snqr_symbolic`](@ref).
"""
function snqr(
        sym::QRSymbolic, A::SparseMatrixCSC{Tv, Ti}; tol = nothing
    ) where {Tv, Ti <: Integer}
    Tr = real(float(Tv))
    size(A) == (sym.transposed ? (sym.n, sym.m) : (sym.m, sym.n)) ||
        throw(DimensionMismatch("matrix/symbolic size mismatch"))
    ns = length(sym.sstart) - 1
    fronts = Vector{Matrix{Tv}}(undef, ns)
    taus = Vector{Vector{Tv}}(undef, ns)
    hcol = Vector{Vector{Int}}(undef, ns)
    vend = Vector{Vector{Int}}(undef, ns)
    frows = Vector{Vector{Int}}(undef, ns)
    for s in 1:ns
        np = sym.sstart[s + 1] - sym.sstart[s]
        nf = np + length(sym.cols[s])
        mf = sym.mfest[s]
        fronts[s] = Matrix{Tv}(undef, mf, nf)
        taus[s] = Vector{Tv}(undef, min(mf, nf))
        hcol[s] = Vector{Int}(undef, min(mf, nf))
        vend[s] = Vector{Int}(undef, min(mf, nf))
        frows[s] = Vector{Int}(undef, mf)
    end
    usertol = tol === nothing ? -one(Tr) : Tr(tol)
    maxmf = ns == 0 ? 0 : maximum(sym.mfest)
    F = SupernodalQRFactor{Tv, Tr, Ti}(
        sym, A, fronts, taus, hcol, vend, frows, zeros(Int, ns), zeros(Int, ns),
        0, zero(Tr), usertol, zeros(Int, sym.nmf),
        Int[], zeros(Int, sym.nmf), _empty_chol(Tv),
        Matrix{Tv}[], Vector{Int}[],
        Int[], Int[], Tv[], Tv[], Int[], Int[], Int[],
        Matrix{Tv}(undef, sym.m, 1), Matrix{Tv}(undef, sym.nmf, 1),
        Matrix{Tv}(undef, sym.nmf, 1), Matrix{Tv}(undef, maxmf, 1),
        Matrix{Tv}(undef, sym.maxnp, 1), Matrix{Tv}(undef, sym.maxnu, 1), maxmf
    )
    return _factor_core!(F)
end

"""
    snqr!(F::SupernodalQRFactor, A) -> F

Refactorize with new values on the SAME sparsity pattern, reusing the
analysis and all front storage.  If the new values make a column singleton's
pivot fall below the dead-column threshold, the analysis is redone (with the
same options) in place.
"""
function snqr!(F::SupernodalQRFactor{Tv}, A::SparseMatrixCSC{Tv}) where {Tv}
    size(A) == size(F) || throw(DimensionMismatch())
    if nnz(A) != length(F.sym.bpos)
        throw(ArgumentError("snqr! requires the same sparsity pattern as the analyzed matrix"))
    end
    F.A = A
    _set_tol!(F)
    if !_singletons_ok(F)
        sym = F.sym
        tol = F.usertol >= 0 ? F.usertol : nothing
        G = snqr(
            snqr_symbolic(
                A; ordering = sym.ordering, relax = sym.relax,
                maxsuper = sym.maxsuper, singletons = sym.singletons,
                transposed = sym.transposed, tol = tol
            ), A; tol = tol
        )
        for f in fieldnames(typeof(F))
            setfield!(F, f, getfield(G, f))
        end
        return F
    end
    return _factor_core!(F)
end

Base.size(F::SupernodalQRFactor) = size(F.A)
Base.size(F::SupernodalQRFactor, i::Integer) = size(F.A, i)

function Base.getproperty(F::SupernodalQRFactor, k::Symbol)
    if k === :q
        return getfield(F, :sym).qf
    elseif k === :R
        return _extract_R(F)
    else
        return getfield(F, k)
    end
end

Base.propertynames(F::SupernodalQRFactor) = (fieldnames(typeof(F))..., :q, :R)

"""
    nnz_factors(F::SupernodalQRFactor) -> Int

Stored entries of R: the singleton rows (rows of `A`) plus the rows of R
kept in the fronts (upper-trapezoidal counting, including explicit zeros from
amalgamation).
"""
function nnz_factors(F::SupernodalQRFactor)
    sym = getfield(F, :sym)
    tot = 0
    for i in sym.srow
        tot += sym.brp[i + 1] - sym.brp[i]
    end
    for s in 1:(length(sym.sstart) - 1)
        nf = sym.sstart[s + 1] - sym.sstart[s] + length(sym.cols[s])
        hc = getfield(F, :hcol)[s]
        for k in 1:getfield(F, :npiv)[s]
            tot += nf - hc[k] + 1
        end
    end
    return tot
end

# Assemble the sparse upper-triangular R of the multifrontal block (nmf × nmf
# in `qf` numbering, singletons excluded; dead columns give empty rows).
# Test/inspection path, not perf-critical.
function _extract_R(F::SupernodalQRFactor{Tv}) where {Tv}
    sym = getfield(F, :sym)
    Is = Int[]
    Js = Int[]
    Vs = Tv[]
    for s in 1:(length(sym.sstart) - 1)
        c1 = sym.sstart[s]
        np = sym.sstart[s + 1] - c1
        U = sym.cols[s]
        nf = np + length(U)
        Fs = getfield(F, :fronts)[s]
        hc = getfield(F, :hcol)[s]
        for k in 1:getfield(F, :npiv)[s]
            row = c1 + hc[k] - 1
            for t in hc[k]:nf
                push!(Is, row)
                push!(Js, t <= np ? c1 + t - 1 : U[t - np])
                push!(Vs, Fs[k, t])
            end
        end
    end
    return sparse(Is, Js, Vs, sym.nmf, sym.nmf)
end

function Base.show(io::IO, ::MIME"text/plain", F::SupernodalQRFactor{Tv}) where {Tv}
    m, n = size(F)
    ns = length(F.sym.sstart) - 1
    return print(
        io, "SupernodalQRFactor{$Tv}: $m × $n, $(length(F.sym.scol)) singletons, $ns fronts, ",
        "nnz(R) = $(nnz_factors(F)), rank $(F.rank)",
        F.sym.transposed ? " (factors Aᴴ)" : ""
    )
end
