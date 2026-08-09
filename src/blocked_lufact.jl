# Blocked, partially pivoted LU for real strided float matrices, in plain
# Julia (`@inbounds` + `@simd ivdep` + `muladd` only — no LoopVectorization,
# no CPU-specific code). Same blocked/panel structure as LAPACK getrf
# (getf2 + laswp + trsm + gemm), which keeps the trailing matrix in cache and
# gives LLVM's auto-vectorizer long contiguous inner loops. Dispatch-selected
# specialization of `generic_lufact!`, so `GenericLUFactorization` uses it for
# `StridedMatrix{Float64/Float32}` with `RowMaximum` pivoting; every other
# eltype/pivot/container keeps the scalar `generic_lufact!` path.

const _BLOCKED_LU_UNBLOCKED_CUTOFF = 8
const _BLOCKED_LU_ROWBLOCK = 384

# Narrow panels win while the trailing matrix is cache-resident; wider ones
# amortize panel/trsm overhead once the Schur update dominates.
_blocked_lu_default_panel(minmn::Int) = minmn <= 160 ? 8 : 16

# Row-maximum pivot search in two passes: a `>`-select max reduction that
# vectorizes (NaN compares false, so NaNs are ignored exactly like the
# scalar stdlib/LAPACK search), then first-index-of-max, matching the
# stdlib's first-occurrence tie-breaking. `kp == k` with `amax == 0` marks an
# all-zero (or all-NaN) subcolumn.
@inline function _blocked_lu_find_pivot(A::AbstractMatrix{T}, k::Int, m::Int) where {T}
    amax = zero(T)
    @inbounds @simd for i in k:m
        absi = abs(A[i, k])
        amax = ifelse(absi > amax, absi, amax)
    end
    kp = k
    if !iszero(amax)
        @inbounds for i in k:m
            if abs(A[i, k]) == amax
                kp = i
                break
            end
        end
    end
    return kp, amax
end

# Unblocked factorization for the whole matrix at small sizes. Identical
# control flow to the scalar `generic_lufact!` RowMaximum path, with hoisted
# multipliers, `muladd`, and `@simd ivdep` on the column updates.
function _blocked_lu_unblocked!(A::AbstractMatrix{T}, ipiv, m::Int, n::Int) where {T}
    minmn = min(m, n)
    info = 0
    @inbounds for k in 1:minmn
        kp, _ = _blocked_lu_find_pivot(A, k, m)
        ipiv[k] = kp
        if !iszero(A[kp, k])
            if k != kp
                for j in 1:n
                    tmp = A[k, j]
                    A[k, j] = A[kp, j]
                    A[kp, j] = tmp
                end
            end
            Akkinv = inv(A[k, k])
            @simd ivdep for i in (k + 1):m
                A[i, k] *= Akkinv
            end
        elseif info == 0
            info = k
        end
        for j in (k + 1):n
            Akj = A[k, j]
            @simd ivdep for i in (k + 1):m
                A[i, j] = muladd(-A[i, k], Akj, A[i, j])
            end
        end
    end
    return info
end

# Panel factorization: unblocked LU of A[j0:m, j0:j1] with pivot search over
# all remaining rows. Row swaps are applied only inside the panel; the driver
# applies them to the rest of the matrix afterwards (getrf structure).
function _blocked_lu_panel!(
        A::AbstractMatrix{T}, ipiv, m::Int, j0::Int, j1::Int, info::Int
    ) where {T}
    @inbounds for k in j0:j1
        kp, _ = _blocked_lu_find_pivot(A, k, m)
        ipiv[k] = kp
        if !iszero(A[kp, k])
            if k != kp
                for j in j0:j1
                    tmp = A[k, j]
                    A[k, j] = A[kp, j]
                    A[kp, j] = tmp
                end
            end
            Akkinv = inv(A[k, k])
            @simd ivdep for i in (k + 1):m
                A[i, k] *= Akkinv
            end
        elseif info == 0
            info = k
        end
        for j in (k + 1):j1
            Akj = A[k, j]
            @simd ivdep for i in (k + 1):m
                A[i, j] = muladd(-A[i, k], Akj, A[i, j])
            end
        end
    end
    return info
end

# Apply the interchanges recorded in ipiv[k0:k1] to columns c0:c1, column-outer
# so each contiguous column stays cache-resident across all of the swaps.
function _blocked_lu_swap_rows!(
        A::AbstractMatrix, ipiv, k0::Int, k1::Int, c0::Int, c1::Int
    )
    c0 > c1 && return nothing
    @inbounds for j in c0:c1
        for k in k0:k1
            kp = ipiv[k]
            if kp != k
                tmp = A[k, j]
                A[k, j] = A[kp, j]
                A[kp, j] = tmp
            end
        end
    end
    return nothing
end

# U12 := L11 \ A12 with L11 the unit-lower triangle of A[j0:j1, j0:j1],
# forward substitution on columns c0:c1, four right-hand-side columns at a
# time so each L column load is amortized.
function _blocked_lu_trsm_unit_lower!(
        A::AbstractMatrix{T}, j0::Int, j1::Int, c0::Int, c1::Int
    ) where {T}
    @inbounds begin
        c = c0
        while c + 3 <= c1
            for k in j0:j1
                b1 = A[k, c]
                b2 = A[k, c + 1]
                b3 = A[k, c + 2]
                b4 = A[k, c + 3]
                @simd ivdep for i in (k + 1):j1
                    aik = A[i, k]
                    A[i, c] = muladd(-aik, b1, A[i, c])
                    A[i, c + 1] = muladd(-aik, b2, A[i, c + 1])
                    A[i, c + 2] = muladd(-aik, b3, A[i, c + 2])
                    A[i, c + 3] = muladd(-aik, b4, A[i, c + 3])
                end
            end
            c += 4
        end
        while c <= c1
            for k in j0:j1
                bk = A[k, c]
                @simd ivdep for i in (k + 1):j1
                    A[i, c] = muladd(-A[i, k], bk, A[i, c])
                end
            end
            c += 1
        end
    end
    return nothing
end

# Schur complement update A[i0:m, c0:c1] -= A[i0:m, j0:j1] * A[j0:j1, c0:c1]
# (C -= L21 * U12); ~all the flops at mid/large N live here. Register
# blocking: 4 columns of C by 4 pivot columns per inner loop (16 fused
# multiply-adds per vectorized i-iteration). Cache blocking: rows in chunks of
# `rowblock` so the active C tile stays L1-resident across the k-chunks. All
# columns touched in an inner loop are distinct, so `ivdep` is exact.
function _blocked_lu_schur_unpacked!(
        A::AbstractMatrix{T}, m::Int, j0::Int, j1::Int,
        c0::Int, c1::Int, rowblock::Int
    ) where {T}
    i0 = j1 + 1
    @inbounds begin
        ib = i0
        while ib <= m
            ie = min(ib + rowblock - 1, m)
            c = c0
            while c + 3 <= c1
                k = j0
                while k + 3 <= j1
                    b11 = A[k, c]
                    b21 = A[k + 1, c]
                    b31 = A[k + 2, c]
                    b41 = A[k + 3, c]
                    b12 = A[k, c + 1]
                    b22 = A[k + 1, c + 1]
                    b32 = A[k + 2, c + 1]
                    b42 = A[k + 3, c + 1]
                    b13 = A[k, c + 2]
                    b23 = A[k + 1, c + 2]
                    b33 = A[k + 2, c + 2]
                    b43 = A[k + 3, c + 2]
                    b14 = A[k, c + 3]
                    b24 = A[k + 1, c + 3]
                    b34 = A[k + 2, c + 3]
                    b44 = A[k + 3, c + 3]
                    @simd ivdep for i in ib:ie
                        a1 = A[i, k]
                        a2 = A[i, k + 1]
                        a3 = A[i, k + 2]
                        a4 = A[i, k + 3]
                        c1v = A[i, c]
                        c1v = muladd(-a1, b11, c1v)
                        c1v = muladd(-a2, b21, c1v)
                        c1v = muladd(-a3, b31, c1v)
                        c1v = muladd(-a4, b41, c1v)
                        A[i, c] = c1v
                        c2v = A[i, c + 1]
                        c2v = muladd(-a1, b12, c2v)
                        c2v = muladd(-a2, b22, c2v)
                        c2v = muladd(-a3, b32, c2v)
                        c2v = muladd(-a4, b42, c2v)
                        A[i, c + 1] = c2v
                        c3v = A[i, c + 2]
                        c3v = muladd(-a1, b13, c3v)
                        c3v = muladd(-a2, b23, c3v)
                        c3v = muladd(-a3, b33, c3v)
                        c3v = muladd(-a4, b43, c3v)
                        A[i, c + 2] = c3v
                        c4v = A[i, c + 3]
                        c4v = muladd(-a1, b14, c4v)
                        c4v = muladd(-a2, b24, c4v)
                        c4v = muladd(-a3, b34, c4v)
                        c4v = muladd(-a4, b44, c4v)
                        A[i, c + 3] = c4v
                    end
                    k += 4
                end
                while k <= j1
                    b1 = A[k, c]
                    b2 = A[k, c + 1]
                    b3 = A[k, c + 2]
                    b4 = A[k, c + 3]
                    @simd ivdep for i in ib:ie
                        a = A[i, k]
                        A[i, c] = muladd(-a, b1, A[i, c])
                        A[i, c + 1] = muladd(-a, b2, A[i, c + 1])
                        A[i, c + 2] = muladd(-a, b3, A[i, c + 2])
                        A[i, c + 3] = muladd(-a, b4, A[i, c + 3])
                    end
                    k += 1
                end
                c += 4
            end
            while c <= c1
                k = j0
                while k + 3 <= j1
                    b1 = A[k, c]
                    b2 = A[k + 1, c]
                    b3 = A[k + 2, c]
                    b4 = A[k + 3, c]
                    @simd ivdep for i in ib:ie
                        acc = A[i, c]
                        acc = muladd(-A[i, k], b1, acc)
                        acc = muladd(-A[i, k + 1], b2, acc)
                        acc = muladd(-A[i, k + 2], b3, acc)
                        acc = muladd(-A[i, k + 3], b4, acc)
                        A[i, c] = acc
                    end
                    k += 4
                end
                while k <= j1
                    bk = A[k, c]
                    @simd ivdep for i in ib:ie
                        A[i, c] = muladd(-A[i, k], bk, A[i, c])
                    end
                    k += 1
                end
                c += 1
            end
            ib = ie + 1
        end
    end
    return nothing
end

# Pack A[i0:m, j0:j1] column-major into `pack` with column stride ldp.
@inline function _blocked_lu_pack_panel!(
        pack::Vector{T}, A::AbstractMatrix{T}, i0::Int, m::Int,
        j0::Int, j1::Int, ldp::Int
    ) where {T}
    @inbounds for k in j0:j1
        off = (k - j0) * ldp - i0 + 1
        @simd ivdep for i in i0:m
            pack[off + i] = A[i, k]
        end
    end
    return nothing
end

# Same update with the L21 operand packed into a contiguous scratch with a
# padded stride. Packing matters because power-of-two leading dimensions
# (N = 512, 1024, ...) put every column on the same L1/L2 cache sets: the
# in-place kernel loses ~25% at N = 512 while LAPACK is immune because GEMM
# packs. Packing costs O(rows*jb) copies against O(rows*jb*n2) flops.
function _blocked_lu_schur_packed!(
        A::AbstractMatrix{T}, m::Int, j0::Int, j1::Int,
        c0::Int, c1::Int, rowblock::Int, pack::Vector{T}
    ) where {T}
    i0 = j1 + 1
    rows = m - i0 + 1
    jb = j1 - j0 + 1
    ldp = rows % 256 == 0 ? rows + 4 : rows
    length(pack) < ldp * jb && resize!(pack, ldp * jb)
    _blocked_lu_pack_panel!(pack, A, i0, m, j0, j1, ldp)
    @inbounds begin
        ib = i0
        while ib <= m
            ie = min(ib + rowblock - 1, m)
            c = c0
            while c + 3 <= c1
                k = j0
                while k + 3 <= j1
                    o1 = (k - j0) * ldp - i0 + 1
                    o2 = o1 + ldp
                    o3 = o2 + ldp
                    o4 = o3 + ldp
                    b11 = A[k, c]
                    b21 = A[k + 1, c]
                    b31 = A[k + 2, c]
                    b41 = A[k + 3, c]
                    b12 = A[k, c + 1]
                    b22 = A[k + 1, c + 1]
                    b32 = A[k + 2, c + 1]
                    b42 = A[k + 3, c + 1]
                    b13 = A[k, c + 2]
                    b23 = A[k + 1, c + 2]
                    b33 = A[k + 2, c + 2]
                    b43 = A[k + 3, c + 2]
                    b14 = A[k, c + 3]
                    b24 = A[k + 1, c + 3]
                    b34 = A[k + 2, c + 3]
                    b44 = A[k + 3, c + 3]
                    @simd ivdep for i in ib:ie
                        a1 = pack[o1 + i]
                        a2 = pack[o2 + i]
                        a3 = pack[o3 + i]
                        a4 = pack[o4 + i]
                        c1v = A[i, c]
                        c1v = muladd(-a1, b11, c1v)
                        c1v = muladd(-a2, b21, c1v)
                        c1v = muladd(-a3, b31, c1v)
                        c1v = muladd(-a4, b41, c1v)
                        A[i, c] = c1v
                        c2v = A[i, c + 1]
                        c2v = muladd(-a1, b12, c2v)
                        c2v = muladd(-a2, b22, c2v)
                        c2v = muladd(-a3, b32, c2v)
                        c2v = muladd(-a4, b42, c2v)
                        A[i, c + 1] = c2v
                        c3v = A[i, c + 2]
                        c3v = muladd(-a1, b13, c3v)
                        c3v = muladd(-a2, b23, c3v)
                        c3v = muladd(-a3, b33, c3v)
                        c3v = muladd(-a4, b43, c3v)
                        A[i, c + 2] = c3v
                        c4v = A[i, c + 3]
                        c4v = muladd(-a1, b14, c4v)
                        c4v = muladd(-a2, b24, c4v)
                        c4v = muladd(-a3, b34, c4v)
                        c4v = muladd(-a4, b44, c4v)
                        A[i, c + 3] = c4v
                    end
                    k += 4
                end
                while k <= j1
                    ok = (k - j0) * ldp - i0 + 1
                    b1 = A[k, c]
                    b2 = A[k, c + 1]
                    b3 = A[k, c + 2]
                    b4 = A[k, c + 3]
                    @simd ivdep for i in ib:ie
                        a = pack[ok + i]
                        A[i, c] = muladd(-a, b1, A[i, c])
                        A[i, c + 1] = muladd(-a, b2, A[i, c + 1])
                        A[i, c + 2] = muladd(-a, b3, A[i, c + 2])
                        A[i, c + 3] = muladd(-a, b4, A[i, c + 3])
                    end
                    k += 1
                end
                c += 4
            end
            while c <= c1
                k = j0
                while k + 3 <= j1
                    o1 = (k - j0) * ldp - i0 + 1
                    o2 = o1 + ldp
                    o3 = o2 + ldp
                    o4 = o3 + ldp
                    b1 = A[k, c]
                    b2 = A[k + 1, c]
                    b3 = A[k + 2, c]
                    b4 = A[k + 3, c]
                    @simd ivdep for i in ib:ie
                        acc = A[i, c]
                        acc = muladd(-pack[o1 + i], b1, acc)
                        acc = muladd(-pack[o2 + i], b2, acc)
                        acc = muladd(-pack[o3 + i], b3, acc)
                        acc = muladd(-pack[o4 + i], b4, acc)
                        A[i, c] = acc
                    end
                    k += 4
                end
                while k <= j1
                    ok = (k - j0) * ldp - i0 + 1
                    bk = A[k, c]
                    @simd ivdep for i in ib:ie
                        A[i, c] = muladd(-pack[ok + i], bk, A[i, c])
                    end
                    k += 1
                end
                c += 1
            end
            ib = ie + 1
        end
    end
    return nothing
end

@inline function _blocked_lu_schur!(
        A::AbstractMatrix{T}, m::Int, j0::Int, j1::Int,
        c0::Int, c1::Int, rowblock::Int, pack::Vector{T}
    ) where {T}
    rows = m - j1
    n2 = c1 - c0 + 1
    if rows >= 64 && n2 >= 32
        _blocked_lu_schur_packed!(A, m, j0, j1, c0, c1, rowblock, pack)
    else
        _blocked_lu_schur_unpacked!(A, m, j0, j1, c0, c1, rowblock)
    end
    return nothing
end

function _blocked_lufact!(
        A::AbstractMatrix{T}, ipiv, m::Int, n::Int, minmn::Int,
        nb::Int, rowblock::Int
    ) where {T}
    pack = T[]
    info = 0
    j0 = 1
    while j0 <= minmn
        jb = min(nb, minmn - j0 + 1)
        j1 = j0 + jb - 1
        info = _blocked_lu_panel!(A, ipiv, m, j0, j1, info)
        _blocked_lu_swap_rows!(A, ipiv, j0, j1, 1, j0 - 1)
        if j1 < n
            _blocked_lu_swap_rows!(A, ipiv, j0, j1, j1 + 1, n)
            _blocked_lu_trsm_unit_lower!(A, j0, j1, j1 + 1, n)
            if j1 < m
                _blocked_lu_schur!(A, m, j0, j1, j1 + 1, n, rowblock, pack)
            end
        end
        j0 += jb
    end
    return info
end

# The `GenericLUFactorization` fast path: real strided float matrices with
# `RowMaximum` pivoting take the blocked kernel; everything else falls through
# to the scalar method. Semantics match `generic_lufact!` with a provided
# `ipiv`: runs to completion with `info` = first zero pivot, `check = true`
# validates finiteness up front and throws `SingularException` unless
# `allowsingular`.
function generic_lufact!(
        A::StridedMatrix{T}, pivot::RowMaximum,
        ipiv::AbstractVector{<:Integer};
        check::Bool = true, allowsingular::Bool = false
    ) where {T <: Union{Float32, Float64}}
    Base.require_one_based_indexing(A, ipiv)
    if check && !all(isfinite, A)
        throw(ArgumentError("matrix contains Infs or NaNs"))
    end
    m, n = size(A)
    minmn = min(m, n)
    length(ipiv) >= minmn ||
        throw(ArgumentError("ipiv has length $(length(ipiv)), needs at least $minmn"))
    info = if minmn <= _BLOCKED_LU_UNBLOCKED_CUTOFF
        _blocked_lu_unblocked!(A, ipiv, m, n)
    else
        _blocked_lufact!(
            A, ipiv, m, n, minmn, _blocked_lu_default_panel(minmn),
            _BLOCKED_LU_ROWBLOCK
        )
    end
    check && !allowsingular && info > 0 &&
        throw(LinearAlgebra.SingularException(info))
    return LinearAlgebra.LU{T, typeof(A), typeof(ipiv)}(
        A, ipiv, convert(LinearAlgebra.BlasInt, info)
    )
end
