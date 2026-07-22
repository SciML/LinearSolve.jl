# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# High-level API: snlu / snlu! and factor extraction.

"""
    snlu(A; ordering=:amd, eps_pivot=1e-8, matching=:auto, relax=true,
         maxsuper=512, check=true) -> SupernodalLUFactor

Factorize sparse square `A` with the supernodal left–right-looking LU
method of Schenk & Gärtner (2004, 2006): supernodal BLAS-3 LU on the
symmetric pattern of `A + Aᵀ`, pivoting restricted to supernode diagonal
blocks with static perturbation of pivots smaller than `eps_pivot * ‖A‖`.

- `ordering` ∈ (`:amd`, `:nd`, `:natural`) — fill-reducing ordering.
- `eps_pivot` — static pivoting threshold (the papers' ε, default `1e-8`).
- `matching` ∈ (`:auto`, `true`, `false`) — MC64-style maximum-weight matching
  + scaling preprocessing (the method's default for unsymmetric systems).
  `:auto` enables it only when the diagonal is structurally missing or
  relatively tiny somewhere; a structurally singular matching falls back to
  the identity.
- `relax`, `maxsuper` — supernode amalgamation controls.
- `check` — throw `SingularException` on a fully-degenerate factorization;
  perturbed pivots by themselves do **not** throw — inspect
  [`nperturbed`](@ref) and rely on the iterative refinement in
  [`solve`](@ref).
- `threaded` — opt-in supernodal-elimination-tree parallel numeric phase
  (engages with `Threads.nthreads() > 1` on large enough problems).  Runs
  independent subtrees on tasks with BLAS pinned to 1 thread, then boosts
  BLAS threads for the serial top of the tree (the big root separators),
  restoring the setting afterwards — so do not run other BLAS work
  concurrently with a threaded factorization.  Same pivot sequence as
  serial; entry rounding may differ via BLAS-parallel sums.
  Refactorization via [`snlu!`](@ref) inherits the setting.

Without matching the result satisfies `A[F.p, F.q] ≈ F.L * F.U`; with
matching the factorized matrix is `(diag(F.Rs)·A·diag(F.Cs))[F.rowperm, :]`.
"""
function snlu(
        A::SparseMatrixCSC{Tv, Ti}; ordering::Symbol = :amd,
        eps_pivot::Float64 = 1.0e-8, matching::Union{Symbol, Bool} = :auto,
        relax::Bool = true, maxsuper::Int = 512, check::Bool = true,
        threaded::Bool = false
    ) where {Tv, Ti <: Integer}
    domatch = matching === true || (matching === :auto && needs_matching(A))
    if domatch
        ms = mc64_matching(A)
        if ms.ok
            B = _matched_matrix(A, ms)
            sym = snlu_symbolic(B; ordering = ordering, relax = relax, maxsuper = maxsuper)
            return _snlu_build(
                sym, A, B, ms;
                eps_pivot = eps_pivot, check = check, threaded = threaded
            )
        end
    end
    sym = snlu_symbolic(A; ordering = ordering, relax = relax, maxsuper = maxsuper)
    F = _snlu_build(
        sym, A, A, nothing;
        eps_pivot = eps_pivot, check = check, threaded = threaded
    )
    # Self-healing retry: mass pivot perturbation means the ε·‖A‖ threshold
    # dwarfed legitimate pivots (huge global dynamic range) — a case the
    # per-column `needs_matching` heuristic cannot see.  Matching + scaling
    # normalizes every entry to ≤ 1, which is exactly the preprocessing the
    # method prescribes; retry once with it.  Only fires on pathological
    # inputs (e.g. Sandia/ASIC_100ks: 96 % of pivots perturbed without it,
    # residual 0.76 → 0 perturbed, residual 4e-12 with it).
    if matching === :auto && !domatch && 20 * F.nperturbed > F.sym.n
        ms = mc64_matching(A)
        if ms.ok
            B = _matched_matrix(A, ms)
            sym2 = snlu_symbolic(B; ordering = ordering, relax = relax, maxsuper = maxsuper)
            F2 = _snlu_build(
                sym2, A, B, ms;
                eps_pivot = eps_pivot, check = check, threaded = threaded
            )
            F2.nperturbed < F.nperturbed && return F2
        end
    end
    return F
end

"""
    snlu(sym::SymbolicAnalysis, A; eps_pivot=1e-8, check=true) -> SupernodalLUFactor

Numeric factorization reusing an analysis from [`snlu_symbolic`](@ref) (PARDISO
phase 22 given phase 11).
"""
function snlu(
        sym::SymbolicAnalysis, A::SparseMatrixCSC{Tv, Ti};
        eps_pivot::Float64 = 1.0e-8, check::Bool = true, threaded::Bool = false
    ) where {Tv, Ti <: Integer}
    return _snlu_build(
        sym, A, A, nothing;
        eps_pivot = eps_pivot, check = check, threaded = threaded
    )
end

# Allocate all numeric state (panels, V/Vt + position maps, every workspace)
# and run the first factorization.  `M` is the matrix the panels factorize
# (the matched/scaled matrix when `ms` is set, otherwise `A` itself).
function _snlu_build(
        sym::SymbolicAnalysis, A::SparseMatrixCSC{Tv, Ti},
        M::SparseMatrixCSC{Tv, Ti}, ms::Union{MatchScale, Nothing};
        eps_pivot::Float64 = 1.0e-8, check::Bool = true, threaded::Bool = false
    ) where {Tv, Ti <: Integer}
    n = sym.n
    size(A) == (n, n) || throw(DimensionMismatch("matrix/symbolic size mismatch"))
    nsuper = length(sym.sstart) - 1
    W = Vector{Matrix{Tv}}(undef, nsuper)
    Z = Vector{Matrix{Tv}}(undef, nsuper)
    rowsfac = Vector{Vector{Int}}(undef, nsuper)
    for s in 1:nsuper
        np = sym.sstart[s + 1] - sym.sstart[s]
        nu = length(sym.rows[s])
        W[s] = Matrix{Tv}(undef, np + nu, np)
        Z[s] = Matrix{Tv}(undef, np, nu)
        rowsfac[s] = Vector{Int}(undef, nu)
    end
    V = M[sym.qf, sym.qf]
    Vt, Tpos = _transpose_map(V)
    F = SupernodalLUFactor{Tv, Ti}(
        sym, W, Z, collect(1:n), collect(1:n), rowsfac, 0, NaN, eps_pivot,
        A, Vector{Tv}(undef, n), Vector{Tv}(undef, 64),
        ms === nothing ? collect(1:n) : ms.rowperm,
        ms === nothing ? ones(n) : ms.r,
        ms === nothing ? ones(n) : ms.c,
        ms !== nothing,
        V, Vt, Int[], Float64[], Tpos,
        zeros(Int, n), Vector{Int}(undef, n), Vector{Int}(undef, n),
        Matrix{Tv}(undef, n, 0),
        Vector{Tv}(undef, n), Vector{Tv}(undef, n), Vector{Tv}(undef, n),
        threaded
    )
    _build_vpos!(F, A)
    _factor_core!(F)
    check && F.nperturbed >= n && n > 0 && throw(SingularException(0))
    return F
end

"""
    snlu!(F::SupernodalLUFactor, A) -> F

Refactorize with new values on the SAME sparsity pattern (PARDISO phase 22
with reuse of phase-11 analysis, matching, and all numeric storage).
Allocation-free after the first factorization.
"""
function snlu!(F::SupernodalLUFactor{Tv}, A::SparseMatrixCSC{Tv}) where {Tv}
    size(A) == (F.sym.n, F.sym.n) || throw(DimensionMismatch())
    if nnz(A) != length(F.Vpos)
        throw(ArgumentError("snlu! requires the same sparsity pattern as the analyzed matrix"))
    end
    F.A = A
    _load_values!(F, A)
    return _factor_core!(F)
end

"""
    nnz_factors(F::SupernodalLUFactor) -> Int

Stored entries of L plus U (including the unit diagonal and any explicit
zeros from pattern symmetrization / supernode amalgamation — the entries the
solver actually stores and operates on).
"""
function nnz_factors(F::SupernodalLUFactor)
    sstart = F.sym.sstart
    tot = 0
    for s in 1:(length(sstart) - 1)
        np = sstart[s + 1] - sstart[s]
        nu = length(F.sym.rows[s])
        tot += np * (np + 1) + 2 * np * nu   # L and U halves of the panels
    end
    return tot
end

Base.size(F::SupernodalLUFactor) = (F.sym.n, F.sym.n)
Base.size(F::SupernodalLUFactor, i::Integer) = i <= 2 ? F.sym.n : 1

function Base.getproperty(F::SupernodalLUFactor{Tv, Ti}, k::Symbol) where {Tv, Ti}
    if k === :q
        return getfield(F, :sym).qf
    elseif k === :L || k === :U
        return _extract_factor(F, k)
    else
        return getfield(F, k)
    end
end

Base.propertynames(F::SupernodalLUFactor) = (fieldnames(typeof(F))..., :q, :L, :U)

# Assemble CSC L or U from the panels (test/inspection path, not perf-critical).
function _extract_factor(F::SupernodalLUFactor{Tv, Ti}, which::Symbol) where {Tv, Ti}
    sym = getfield(F, :sym)
    n = sym.n
    sstart = sym.sstart
    Is = Int[]
    Js = Int[]
    Vs = Tv[]
    for s in 1:(length(sstart) - 1)
        c1 = sstart[s]
        c2 = sstart[s + 1] - 1
        np = c2 - c1 + 1
        Rf = getfield(F, :rowsfac)[s]
        R = sym.rows[s]
        nu = length(R)
        Ws = getfield(F, :W)[s]
        Zs = getfield(F, :Z)[s]
        for a in 1:np
            j = c1 + a - 1
            if which === :L
                push!(Is, j); push!(Js, j); push!(Vs, one(Tv))
                for b in (a + 1):np
                    push!(Is, c1 + b - 1); push!(Js, j); push!(Vs, Ws[b, a])
                end
                for t in 1:nu
                    push!(Is, Rf[t]); push!(Js, j); push!(Vs, Ws[np + t, a])
                end
            else
                for b in 1:a
                    push!(Is, c1 + b - 1); push!(Js, j); push!(Vs, Ws[b, a])
                end
                for t in 1:nu
                    push!(Is, j); push!(Js, R[t]); push!(Vs, Zs[a, t])
                end
            end
        end
    end
    return sparse(Is, Js, Vs, n, n)
end

function Base.show(io::IO, ::MIME"text/plain", F::SupernodalLUFactor{Tv}) where {Tv}
    n = F.sym.n
    ns = length(F.sym.sstart) - 1
    return print(
        io, "SupernodalLUFactor{$Tv}: n = $n, $ns supernodes, ",
        "nnz(L+U) = $(nnz_factors(F)), $(F.nperturbed) perturbed pivots"
    )
end
