# Blocked, partially pivoted LU for real strided float matrices, in plain
# Julia: no LoopVectorization, no SIMD.jl, no CPU feature detection, the only
# non-scalar primitives being Base's `VecElement` tuples and LLVM's `fmuladd`.
# Same blocked/panel structure as LAPACK getrf (getf2 + laswp + trsm + gemm),
# which keeps the trailing matrix in cache. Dispatch-selected specialization of
# `generic_lufact!`, so `GenericLUFactorization` uses it for
# `StridedMatrix{Float64/Float32}` with `RowMaximum` pivoting; every other
# eltype/pivot/container keeps the scalar `generic_lufact!` path.

const _BLOCKED_LU_UNBLOCKED_CUTOFF = 8
const _BLOCKED_LU_ROWBLOCK = 384

# Narrow panels win while the trailing matrix is cache-resident; wider ones
# amortize panel/trsm overhead once the Schur update dominates.
_blocked_lu_default_panel(minmn::Int) = minmn <= 160 ? 8 : 16

function _blocked_lu_pack_size(m::Int, n::Int, minmn::Int, nb::Int)
    len = 0
    j0 = 1
    while j0 <= minmn
        jb = min(nb, minmn - j0 + 1)
        j1 = j0 + jb - 1
        rows = m - j1
        if rows >= 64 && n - j1 >= 32
            ldp = rows % 256 == 0 ? rows + 4 : rows
            len = max(len, ldp * jb)
        end
        j0 += jb
    end
    return len
end

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

# Pack -A[i0:m, j0:j1] column-major into `pack` with column stride ldp. The
# negation happens here, O(rows*jb) times, so the O(rows*jb*n2) update loops
# below are plain fused multiply-adds.
@inline function _blocked_lu_pack_panel!(
        pack::Vector{T}, A::AbstractMatrix{T}, i0::Int, m::Int,
        j0::Int, j1::Int, ldp::Int
    ) where {T}
    @inbounds for k in j0:j1
        off = (k - j0) * ldp - i0 + 1
        @simd ivdep for i in i0:m
            pack[off + i] = -A[i, k]
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
                        c1v = muladd(a1, b11, c1v)
                        c1v = muladd(a2, b21, c1v)
                        c1v = muladd(a3, b31, c1v)
                        c1v = muladd(a4, b41, c1v)
                        A[i, c] = c1v
                        c2v = A[i, c + 1]
                        c2v = muladd(a1, b12, c2v)
                        c2v = muladd(a2, b22, c2v)
                        c2v = muladd(a3, b32, c2v)
                        c2v = muladd(a4, b42, c2v)
                        A[i, c + 1] = c2v
                        c3v = A[i, c + 2]
                        c3v = muladd(a1, b13, c3v)
                        c3v = muladd(a2, b23, c3v)
                        c3v = muladd(a3, b33, c3v)
                        c3v = muladd(a4, b43, c3v)
                        A[i, c + 2] = c3v
                        c4v = A[i, c + 3]
                        c4v = muladd(a1, b14, c4v)
                        c4v = muladd(a2, b24, c4v)
                        c4v = muladd(a3, b34, c4v)
                        c4v = muladd(a4, b44, c4v)
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
                        A[i, c] = muladd(a, b1, A[i, c])
                        A[i, c + 1] = muladd(a, b2, A[i, c + 1])
                        A[i, c + 2] = muladd(a, b3, A[i, c + 2])
                        A[i, c + 3] = muladd(a, b4, A[i, c + 3])
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
                        acc = muladd(pack[o1 + i], b1, acc)
                        acc = muladd(pack[o2 + i], b2, acc)
                        acc = muladd(pack[o3 + i], b3, acc)
                        acc = muladd(pack[o4 + i], b4, acc)
                        A[i, c] = acc
                    end
                    k += 4
                end
                while k <= j1
                    ok = (k - j0) * ldp - i0 + 1
                    bk = A[k, c]
                    @simd ivdep for i in ib:ie
                        A[i, c] = muladd(pack[ok + i], bk, A[i, c])
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

# `@simd ivdep` gives LLVM one accumulator per C column, so every fused
# multiply-add pays for a load and a store of C. The microkernel below holds a
# `3W x 4` tile of C in vector registers across the whole k loop instead: 12
# accumulators, 3 panel vectors and one broadcast, exactly the 16 vector
# registers x86-64 has. Still dependency-free — `VecElement` tuples are Base's
# portable vector type and `llvm.fmuladd` is reachable through `ccall(...,
# llvmcall, ...)`. `fmuladd` rather than `fma` contracts to a hardware fma
# where the target has one and degrades to mul + add where it does not, never
# to a libm call.

# 256-bit tiles on x86-64 (AVX and up), 128-bit elsewhere; both are legal LLVM
# vector types on every target, which a narrower machine splits.
const _BLOCKED_LU_VEC_BYTES = Sys.ARCH === :x86_64 ? 32 : 16

for (T, sfx) in ((Float64, "f64"), (Float32, "f32")), W in (2, 4, 8, 16)
    V = NTuple{W, VecElement{T}}
    @eval @inline _blocked_lu_fma(a::$V, b::$V, c::$V) = ccall(
        $("llvm.fmuladd.v$(W)$(sfx)"), llvmcall, $V, ($V, $V, $V), a, b, c
    )
end

@inline _blocked_lu_vectype(::Type{T}) where {T} =
    NTuple{_BLOCKED_LU_VEC_BYTES ÷ sizeof(T), VecElement{T}}
@inline _blocked_lu_vload(::Type{V}, p::Ptr) where {V} = unsafe_load(Ptr{V}(p))
@inline _blocked_lu_vstore!(p::Ptr, v::V) where {V} = unsafe_store!(Ptr{V}(p), v)
@inline _blocked_lu_bcast(::Type{NTuple{W, VecElement{T}}}, x::T) where {W, T} =
    ntuple(_ -> VecElement(x), Val(W))

# Everything the tile below cannot cover — row and column remainders, and
# regions too small for a whole tile — one column and one row vector at a time.
@inline function _blocked_lu_micro_edge!(
        ::Type{V}, pA::Ptr{T}, ld::Int, pP::Ptr{T}, ldp::Int,
        i0::Int, ib::Int, ie::Int, j0::Int, j1::Int, c0::Int, c1::Int
    ) where {W, T, V <: NTuple{W, VecElement{T}}}
    ib > ie && return nothing
    sz = sizeof(T)
    lds = ld * sz
    ldps = ldp * sz
    for c in c0:c1
        pcol = pA + (c - 1) * lds
        pb0 = pcol + (j0 - 1) * sz
        i = ib
        while i + W - 1 <= ie
            q = pcol + (i - 1) * sz
            acc = _blocked_lu_vload(V, q)
            pk = pP + (i - i0) * sz
            pb = pb0
            for _ in j0:j1
                acc = _blocked_lu_fma(
                    _blocked_lu_vload(V, pk), _blocked_lu_bcast(V, unsafe_load(pb)), acc
                )
                pk += ldps
                pb += sz
            end
            _blocked_lu_vstore!(q, acc)
            i += W
        end
        while i <= ie
            q = pcol + (i - 1) * sz
            s = unsafe_load(q)
            pk = pP + (i - i0) * sz
            pb = pb0
            for _ in j0:j1
                s = muladd(unsafe_load(pk), unsafe_load(pb), s)
                pk += ldps
                pb += sz
            end
            unsafe_store!(q, s)
            i += 1
        end
    end
    return nothing
end

# C[ib:ie, c0:c1] += P * B over whole `3W x 4` tiles, P the packed (negated)
# L21 panel and B = A[j0:j1, c0:c1]. Returns the first column no tile covered.
@inline function _blocked_lu_micro_tile!(
        ::Type{V}, pA::Ptr{T}, ld::Int, pP::Ptr{T}, ldp::Int,
        i0::Int, ib::Int, ie::Int, j0::Int, j1::Int, c0::Int, c1::Int
    ) where {W, T, V <: NTuple{W, VecElement{T}}}
    sz = sizeof(T)
    lds = ld * sz
    ldps = ldp * sz
    vb = W * sz
    c = c0
    while c + 3 <= c1
        pcol = pA + (c - 1) * lds
        pb0 = pcol + (j0 - 1) * sz
        i = ib
        while i + 3W - 1 <= ie
            q1 = pcol + (i - 1) * sz
            q2 = q1 + lds
            q3 = q2 + lds
            q4 = q3 + lds
            t11 = _blocked_lu_vload(V, q1)
            t21 = _blocked_lu_vload(V, q1 + vb)
            t31 = _blocked_lu_vload(V, q1 + 2vb)
            t12 = _blocked_lu_vload(V, q2)
            t22 = _blocked_lu_vload(V, q2 + vb)
            t32 = _blocked_lu_vload(V, q2 + 2vb)
            t13 = _blocked_lu_vload(V, q3)
            t23 = _blocked_lu_vload(V, q3 + vb)
            t33 = _blocked_lu_vload(V, q3 + 2vb)
            t14 = _blocked_lu_vload(V, q4)
            t24 = _blocked_lu_vload(V, q4 + vb)
            t34 = _blocked_lu_vload(V, q4 + 2vb)
            pk = pP + (i - i0) * sz
            pb = pb0
            for _ in j0:j1
                p1 = _blocked_lu_vload(V, pk)
                p2 = _blocked_lu_vload(V, pk + vb)
                p3 = _blocked_lu_vload(V, pk + 2vb)
                b = _blocked_lu_bcast(V, unsafe_load(pb))
                t11 = _blocked_lu_fma(p1, b, t11)
                t21 = _blocked_lu_fma(p2, b, t21)
                t31 = _blocked_lu_fma(p3, b, t31)
                b = _blocked_lu_bcast(V, unsafe_load(pb + lds))
                t12 = _blocked_lu_fma(p1, b, t12)
                t22 = _blocked_lu_fma(p2, b, t22)
                t32 = _blocked_lu_fma(p3, b, t32)
                b = _blocked_lu_bcast(V, unsafe_load(pb + 2lds))
                t13 = _blocked_lu_fma(p1, b, t13)
                t23 = _blocked_lu_fma(p2, b, t23)
                t33 = _blocked_lu_fma(p3, b, t33)
                b = _blocked_lu_bcast(V, unsafe_load(pb + 3lds))
                t14 = _blocked_lu_fma(p1, b, t14)
                t24 = _blocked_lu_fma(p2, b, t24)
                t34 = _blocked_lu_fma(p3, b, t34)
                pk += ldps
                pb += sz
            end
            _blocked_lu_vstore!(q1, t11)
            _blocked_lu_vstore!(q1 + vb, t21)
            _blocked_lu_vstore!(q1 + 2vb, t31)
            _blocked_lu_vstore!(q2, t12)
            _blocked_lu_vstore!(q2 + vb, t22)
            _blocked_lu_vstore!(q2 + 2vb, t32)
            _blocked_lu_vstore!(q3, t13)
            _blocked_lu_vstore!(q3 + vb, t23)
            _blocked_lu_vstore!(q3 + 2vb, t33)
            _blocked_lu_vstore!(q4, t14)
            _blocked_lu_vstore!(q4 + vb, t24)
            _blocked_lu_vstore!(q4 + 2vb, t34)
            i += 3W
        end
        c += 4
    end
    return c
end

# Same update as `_blocked_lu_schur_packed!` through the microkernel. Rows stay
# cut into `rowblock` chunks so the packed panel the tiles stream over each
# column group stays cache-resident.
function _blocked_lu_schur_micro!(
        A::AbstractMatrix{T}, m::Int, j0::Int, j1::Int,
        c0::Int, c1::Int, rowblock::Int, pack::Vector{T}
    ) where {T}
    i0 = j1 + 1
    rows = m - i0 + 1
    jb = j1 - j0 + 1
    ldp = rows % 256 == 0 ? rows + 4 : rows
    _blocked_lu_pack_panel!(pack, A, i0, m, j0, j1, ldp)
    V = _blocked_lu_vectype(T)
    # Must equal the tile's row step; a mismatch silently double-applies or
    # skips rows rather than erroring.
    mr = 3 * (_BLOCKED_LU_VEC_BYTES ÷ sizeof(T))
    ld = stride(A, 2)
    GC.@preserve A pack begin
        pA = pointer(A)
        pP = pointer(pack)
        ib = i0
        while ib <= m
            ie = min(ib + rowblock - 1, m)
            rfull = ib + ((ie - ib + 1) ÷ mr) * mr - 1
            ct = _blocked_lu_micro_tile!(V, pA, ld, pP, ldp, i0, ib, rfull, j0, j1, c0, c1)
            _blocked_lu_micro_edge!(V, pA, ld, pP, ldp, i0, rfull + 1, ie, j0, j1, c0, c1)
            _blocked_lu_micro_edge!(V, pA, ld, pP, ldp, i0, ib, rfull, j0, j1, ct, c1)
            ib = ie + 1
        end
    end
    return nothing
end

# The microkernel addresses A through a raw pointer, so it needs the
# `StridedArray` pointer/stride contract and a unit row stride.
@inline function _blocked_lu_schur_wide!(
        A::AbstractMatrix{T}, m::Int, j0::Int, j1::Int,
        c0::Int, c1::Int, rowblock::Int, pack::Vector{T}
    ) where {T}
    _blocked_lu_schur_packed!(A, m, j0, j1, c0, c1, rowblock, pack)
    return nothing
end

@inline function _blocked_lu_schur_wide!(
        A::StridedMatrix{T}, m::Int, j0::Int, j1::Int,
        c0::Int, c1::Int, rowblock::Int, pack::Vector{T}
    ) where {T <: Union{Float32, Float64}}
    if stride(A, 1) == 1
        _blocked_lu_schur_micro!(A, m, j0, j1, c0, c1, rowblock, pack)
    else
        _blocked_lu_schur_packed!(A, m, j0, j1, c0, c1, rowblock, pack)
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
        _blocked_lu_schur_wide!(A, m, j0, j1, c0, c1, rowblock, pack)
    else
        _blocked_lu_schur_unpacked!(A, m, j0, j1, c0, c1, rowblock)
    end
    return nothing
end

function _blocked_lufact!(
        A::AbstractMatrix{T}, ipiv, m::Int, n::Int, minmn::Int,
        nb::Int, rowblock::Int, pack::Vector{T}
    ) where {T}
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

function _blocked_lufact!(
        A::AbstractMatrix{T}, ipiv, m::Int, n::Int, minmn::Int,
        nb::Int, rowblock::Int
    ) where {T}
    pack = Vector{T}(undef, _blocked_lu_pack_size(m, n, minmn, nb))
    return _blocked_lufact!(A, ipiv, m, n, minmn, nb, rowblock, pack)
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
    return _blocked_generic_lufact!(
        A, pivot, ipiv, nothing; check = check, allowsingular = allowsingular
    )
end

function generic_lufact!(
        A::StridedMatrix{T}, pivot::RowMaximum,
        ipiv::AbstractVector{<:Integer}, pack::Vector{T};
        check::Bool = true, allowsingular::Bool = false
    ) where {T <: Union{Float32, Float64}}
    return _blocked_generic_lufact!(
        A, pivot, ipiv, pack; check = check, allowsingular = allowsingular
    )
end

function _blocked_generic_lufact!(
        A::StridedMatrix{T}, pivot::RowMaximum,
        ipiv::AbstractVector{<:Integer}, pack::Union{Nothing, Vector{T}};
        check::Bool = true, allowsingular::Bool = false
    ) where {T <: Union{Float32, Float64}}
    require_one_based_indexing(A, ipiv)
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
        nb = _blocked_lu_default_panel(minmn)
        pack === nothing &&
            (pack = Vector{T}(undef, _blocked_lu_pack_size(m, n, minmn, nb)))
        _blocked_lufact!(
            A, ipiv, m, n, minmn, nb, _BLOCKED_LU_ROWBLOCK, pack
        )
    end
    check && !allowsingular && info > 0 &&
        throw(LinearAlgebra.SingularException(info))
    return LinearAlgebra.LU{T, typeof(A), typeof(ipiv)}(
        A, ipiv, convert(LinearAlgebra.BlasInt, info)
    )
end
