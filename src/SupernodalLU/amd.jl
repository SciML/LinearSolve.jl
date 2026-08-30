# SPDX-FileCopyrightText: 1996-2025 Timothy A. Davis, Patrick R. Amestoy, and Iain S. Duff
# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: BSD-3-Clause
#
# Vendored from PureKLU.jl (src/AMD.jl), itself a direct pure-Julia port of
# SuiteSparse AMD (BSD-3-Clause).  See LICENSE for the BSD-3-Clause text.

"""
    AMD

Direct pure-Julia port of SuiteSparse's AMD (Approximate Minimum Degree)
ordering algorithm. Matches the SuiteSparse implementation closely
enough to produce identical permutations on the same inputs.

The entry point is [`amd_order!`](@ref); supporting routines mirror
`amd_aat`, `amd_1`, `amd_2`, `amd_postorder` and `amd_post_tree` from
SuiteSparse.
"""
module AMD

const EMPTY = -1
@inline _flip(j) = -j - 2

const AMD_OK = 0
const AMD_OUT_OF_MEMORY = -1
const AMD_INVALID = -2
const AMD_OK_BUT_JUMBLED = 1

const AMD_DEFAULT_DENSE = 10.0
const AMD_DEFAULT_AGGRESSIVE = 1

"""
    amd_aat!(n, Ap, Ai, Len, Tp) -> nzaat

Compute the lengths of `A+A'` columns (excluding diagonal). `Len` is
written; `Tp` is scratch of size `n`. Mirrors `amd_aat.c`.
"""
function amd_aat!(
        n::Int, Ap::AbstractVector{Ti}, Ai::AbstractVector{Ti},
        Len::AbstractVector{Ti}, Tp::AbstractVector{Ti}
    ) where {Ti <: Integer}
    @inbounds for k in 1:n
        Len[k] = Ti(0)
    end
    nzdiag = 0
    nzboth = 0
    nz = Int(Ap[n + 1])
    @inbounds for k in 0:(n - 1)
        p = Int(Ap[k + 1])
        p2 = Int(Ap[k + 2])
        while p < p2
            j = Int(Ai[p + 1])
            if j < k
                Len[j + 1] += Ti(1)
                Len[k + 1] += Ti(1)
                p += 1
            elseif j == k
                p += 1
                nzdiag += 1
                break
            else
                break
            end
            pj2 = Int(Ap[j + 2])
            pj = Int(Tp[j + 1])
            done = false
            while pj < pj2
                i = Int(Ai[pj + 1])
                if i < k
                    Len[i + 1] += Ti(1)
                    Len[j + 1] += Ti(1)
                    pj += 1
                elseif i == k
                    pj += 1
                    nzboth += 1
                    done = true
                    break
                else
                    break
                end
            end
            Tp[j + 1] = Ti(pj)
            if done
                # continue outer while loop
            end
        end
        Tp[k + 1] = Ti(p)
    end
    @inbounds for j in 0:(n - 1)
        pj = Int(Tp[j + 1])
        pjend = Int(Ap[j + 2])
        while pj < pjend
            i = Int(Ai[pj + 1])
            Len[i + 1] += Ti(1)
            Len[j + 1] += Ti(1)
            pj += 1
        end
    end

    nzaat = 0
    @inbounds for k in 1:n
        nzaat += Int(Len[k])
    end
    return nzaat
end

@inline function _clear_flag(wflg::Int, wbig::Int, W::AbstractVector{Ti}, n::Int) where {Ti}
    if wflg < 2 || wflg >= wbig
        @inbounds for x in 1:n
            if W[x] != 0
                W[x] = Ti(1)
            end
        end
        return 2
    end
    return wflg
end

# Out-of-line garbage collection for the workspace `Iw`.  Rare path; keeping
# it noinline lets the surrounding hot loop stay tight.
@noinline function _amd_gc!(
        n::Int, me::Int, e::Int, p_in::Int, pj_in::Int,
        knt1::Int, knt2::Int, ln::Int, pme1::Int, pfree_in::Int,
        Pe::Vector{Ti}, Iw::Vector{Ti},
        Len::Vector{Ti}
    ) where {Ti <: Integer}
    p = p_in
    pj = pj_in
    pfree = pfree_in
    @inbounds begin
        Pe[me + 1] = Ti(p)
        Len[me + 1] -= Ti(knt1)
        if Len[me + 1] == 0
            Pe[me + 1] = Ti(EMPTY)
        end
        Pe[e + 1] = Ti(pj)
        Len[e + 1] = Ti(ln - knt2)
        if Len[e + 1] == 0
            Pe[e + 1] = Ti(EMPTY)
        end
        for j in 0:(n - 1)
            pn = Int(Pe[j + 1])
            if pn >= 0
                Pe[j + 1] = Iw[pn + 1]
                Iw[pn + 1] = Ti(_flip(j))
            end
        end
        psrc = 0
        pdst = 0
        pend = pme1 - 1
        while psrc <= pend
            j = _flip(Int(Iw[psrc + 1]))
            psrc += 1
            if j >= 0
                Iw[pdst + 1] = Pe[j + 1]
                Pe[j + 1] = Ti(pdst)
                pdst += 1
                lenj = Int(Len[j + 1])
                for _knt3 in 1:(lenj - 1)
                    Iw[pdst + 1] = Iw[psrc + 1]
                    pdst += 1; psrc += 1
                end
            end
        end
        p1_local = pdst
        psrc = pme1
        while psrc <= pfree - 1
            Iw[pdst + 1] = Iw[psrc + 1]
            pdst += 1; psrc += 1
        end
        pme1 = p1_local
        pfree = pdst
        pj = Int(Pe[e + 1])
        p = Int(Pe[me + 1])
    end
    return (p, pj, pfree, pme1)
end

"""
    amd_2!(n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W,
           dense_threshold, aggressive) -> nothing

Core AMD ordering routine. Operates on the A+A' representation built in
`Pe`/`Iw`/`Len`. Writes the permutation into `Last` and inverse
permutation into `Next`. Mirrors `amd_2.c`.
"""
function amd_2!(
        n::Int, Pe::Vector{Ti}, Iw::Vector{Ti}, Len::Vector{Ti},
        iwlen::Int, pfree_in::Int,
        Nv::Vector{Ti}, Next::Vector{Ti}, Last::Vector{Ti},
        Head::Vector{Ti}, Elen::Vector{Ti}, Degree::Vector{Ti},
        W::Vector{Ti};
        dense_alpha::Float64 = AMD_DEFAULT_DENSE,
        aggressive::Bool = AMD_DEFAULT_AGGRESSIVE != 0
    ) where {Ti <: Integer}

    pfree = pfree_in
    mindeg = 0
    ncmpa = 0
    nel = 0
    lemax = 0
    me = EMPTY
    lnz = 0.0

    dense = dense_alpha < 0 ? n - 2 : floor(Int, dense_alpha * sqrt(Float64(n)))
    dense = max(16, dense)
    dense = min(n, dense)

    @inbounds for i in 1:n
        Last[i] = Ti(EMPTY)
        Head[i] = Ti(EMPTY)
        Next[i] = Ti(EMPTY)
        Nv[i] = Ti(1)
        W[i] = Ti(1)
        Elen[i] = Ti(0)
        Degree[i] = Len[i]
    end

    wbig = typemax(Ti) - Ti(n)
    wflg = _clear_flag(0, Int(wbig), W, n)

    ndense = 0
    @inbounds for i in 0:(n - 1)
        deg = Int(Degree[i + 1])
        if deg == 0
            Elen[i + 1] = Ti(_flip(1))
            nel += 1
            Pe[i + 1] = Ti(EMPTY)
            W[i + 1] = Ti(0)
        elseif deg > dense
            ndense += 1
            Nv[i + 1] = Ti(0)
            Elen[i + 1] = Ti(EMPTY)
            nel += 1
            Pe[i + 1] = Ti(EMPTY)
        else
            inext = Int(Head[deg + 1])
            if inext != EMPTY
                Last[inext + 1] = Ti(i)
            end
            Next[i + 1] = Ti(inext)
            Head[deg + 1] = Ti(i)
        end
    end

    @inbounds while nel < n
        # --- pick minimum-degree pivot
        deg = mindeg
        while deg < n
            me = Int(Head[deg + 1])
            if me != EMPTY
                break
            end
            deg += 1
        end
        mindeg = deg

        inext = Int(Next[me + 1])
        if inext != EMPTY
            Last[inext + 1] = Ti(EMPTY)
        end
        Head[deg + 1] = Ti(inext)

        elenme = Int(Elen[me + 1])
        nvpiv = Int(Nv[me + 1])
        nel += nvpiv

        Nv[me + 1] = Ti(-nvpiv)
        degme = 0
        pme1 = 0
        pme2 = 0

        if elenme == 0
            pme1 = Int(Pe[me + 1])
            pme2 = pme1 - 1
            lenme = Int(Len[me + 1])
            for p in pme1:(pme1 + lenme - 1)
                i = Int(Iw[p + 1])
                nvi = Int(Nv[i + 1])
                if nvi > 0
                    degme += nvi
                    Nv[i + 1] = Ti(-nvi)
                    pme2 += 1
                    Iw[pme2 + 1] = Ti(i)

                    ilast = Int(Last[i + 1])
                    inext2 = Int(Next[i + 1])
                    if inext2 != EMPTY
                        Last[inext2 + 1] = Ti(ilast)
                    end
                    if ilast != EMPTY
                        Next[ilast + 1] = Ti(inext2)
                    else
                        Head[Int(Degree[i + 1]) + 1] = Ti(inext2)
                    end
                end
            end
        else
            p = Int(Pe[me + 1])
            pme1 = pfree
            slenme = Int(Len[me + 1]) - elenme

            for knt1 in 1:(elenme + 1)
                if knt1 > elenme
                    e = me
                    pj = p
                    ln = slenme
                else
                    e = Int(Iw[p + 1]); p += 1
                    pj = Int(Pe[e + 1])
                    ln = Int(Len[e + 1])
                end

                for knt2 in 1:ln
                    i = Int(Iw[pj + 1]); pj += 1
                    nvi = Int(Nv[i + 1])
                    if nvi > 0
                        if pfree >= iwlen
                            ncmpa += 1
                            p, pj, pfree, pme1 = _amd_gc!(
                                n, me, e, p, pj, knt1, knt2, ln,
                                pme1, pfree, Pe, Iw, Len
                            )
                        end

                        degme += nvi
                        Nv[i + 1] = Ti(-nvi)
                        Iw[pfree + 1] = Ti(i)
                        pfree += 1

                        ilast = Int(Last[i + 1])
                        inext2 = Int(Next[i + 1])
                        if inext2 != EMPTY
                            Last[inext2 + 1] = Ti(ilast)
                        end
                        if ilast != EMPTY
                            Next[ilast + 1] = Ti(inext2)
                        else
                            Head[Int(Degree[i + 1]) + 1] = Ti(inext2)
                        end
                    end
                end

                if e != me
                    Pe[e + 1] = Ti(_flip(me))
                    W[e + 1] = Ti(0)
                end
            end
            pme2 = pfree - 1
        end

        Degree[me + 1] = Ti(degme)
        Pe[me + 1] = Ti(pme1)
        Len[me + 1] = Ti(pme2 - pme1 + 1)
        Elen[me + 1] = Ti(_flip(nvpiv + degme))

        wflg = _clear_flag(wflg, Int(wbig), W, n)

        # --- Scan 1 ---
        for pme in pme1:pme2
            i = Int(Iw[pme + 1])
            eln = Int(Elen[i + 1])
            if eln > 0
                nvi = -Int(Nv[i + 1])
                wnvi = wflg - nvi
                pibase = Int(Pe[i + 1])
                for pp in pibase:(pibase + eln - 1)
                    e = Int(Iw[pp + 1])
                    we = Int(W[e + 1])
                    if we >= wflg
                        we -= nvi
                    elseif we != 0
                        we = Int(Degree[e + 1]) + wnvi
                    end
                    W[e + 1] = Ti(we)
                end
            end
        end

        # --- Scan 2 (degree update + supervariable hashing) ---
        for pme in pme1:pme2
            i = Int(Iw[pme + 1])
            p1 = Int(Pe[i + 1])
            p2 = p1 + Int(Elen[i + 1]) - 1
            pn = p1
            hash = UInt(0)
            deg = 0

            if aggressive
                for pp in p1:p2
                    e = Int(Iw[pp + 1])
                    we = Int(W[e + 1])
                    if we != 0
                        dext = we - wflg
                        if dext > 0
                            deg += dext
                            Iw[pn + 1] = Ti(e); pn += 1
                            hash += UInt(e)
                        else
                            Pe[e + 1] = Ti(_flip(me))
                            W[e + 1] = Ti(0)
                        end
                    end
                end
            else
                for pp in p1:p2
                    e = Int(Iw[pp + 1])
                    we = Int(W[e + 1])
                    if we != 0
                        dext = we - wflg
                        deg += dext
                        Iw[pn + 1] = Ti(e); pn += 1
                        hash += UInt(e)
                    end
                end
            end

            elen_new = pn - p1 + 1
            Elen[i + 1] = Ti(elen_new)

            p3 = pn
            p4 = p1 + Int(Len[i + 1])
            for pp in (p2 + 1):(p4 - 1)
                j = Int(Iw[pp + 1])
                nvj = Int(Nv[j + 1])
                if nvj > 0
                    deg += nvj
                    Iw[pn + 1] = Ti(j); pn += 1
                    hash += UInt(j)
                end
            end

            if elen_new == 1 && p3 == pn
                Pe[i + 1] = Ti(_flip(me))
                nvi = -Int(Nv[i + 1])
                degme -= nvi
                nvpiv += nvi
                nel += nvi
                Nv[i + 1] = Ti(0)
                Elen[i + 1] = Ti(EMPTY)
            else
                degi = Int(Degree[i + 1])
                Degree[i + 1] = Ti(deg < degi ? deg : degi)

                Iw[pn + 1] = Iw[p3 + 1]
                Iw[p3 + 1] = Iw[p1 + 1]
                Iw[p1 + 1] = Ti(me)
                Len[i + 1] = Ti(pn - p1 + 1)

                hash = hash % UInt(n)
                hashi = Int(hash)
                j = Int(Head[hashi + 1])
                if j <= EMPTY
                    Next[i + 1] = Ti(_flip(j))
                    Head[hashi + 1] = Ti(_flip(i))
                else
                    Next[i + 1] = Last[j + 1]
                    Last[j + 1] = Ti(i)
                end
                Last[i + 1] = Ti(hashi)
            end
        end

        Degree[me + 1] = Ti(degme)
        if degme > lemax
            lemax = degme
        end
        wflg += lemax
        wflg = _clear_flag(wflg, Int(wbig), W, n)

        # --- Supervariable detection ---
        for pme in pme1:pme2
            i = Int(Iw[pme + 1])
            if Nv[i + 1] < 0
                hashi = Int(Last[i + 1])
                j = Int(Head[hashi + 1])
                if j == EMPTY
                    i = EMPTY
                elseif j < EMPTY
                    i = _flip(j)
                    Head[hashi + 1] = Ti(EMPTY)
                else
                    i = Int(Last[j + 1])
                    Last[j + 1] = Ti(EMPTY)
                end

                while i != EMPTY && Next[i + 1] != Ti(EMPTY)
                    ln = Int(Len[i + 1])
                    eln = Int(Elen[i + 1])
                    pi = Int(Pe[i + 1])
                    wflg_t = Ti(wflg)
                    for pp in (pi + 1):(pi + ln - 1)
                        W[Int(Iw[pp + 1]) + 1] = wflg_t
                    end

                    jlast = i
                    j = Int(Next[i + 1])

                    while j != EMPTY
                        ok = (Int(Len[j + 1]) == ln) && (Int(Elen[j + 1]) == eln)
                        if ok
                            pj0 = Int(Pe[j + 1])
                            ppend = pj0 + ln - 1
                            pp = pj0 + 1
                            while pp <= ppend
                                if W[Int(Iw[pp + 1]) + 1] != wflg_t
                                    ok = false
                                    break
                                end
                                pp += 1
                            end
                        end
                        if ok
                            Pe[j + 1] = Ti(_flip(i))
                            Nv[i + 1] += Nv[j + 1]
                            Nv[j + 1] = Ti(0)
                            Elen[j + 1] = Ti(EMPTY)
                            j = Int(Next[j + 1])
                            Next[jlast + 1] = Ti(j)
                        else
                            jlast = j
                            j = Int(Next[j + 1])
                        end
                    end

                    wflg += 1
                    i = Int(Next[i + 1])
                end
            end
        end

        # --- Restore degree lists, remove non-principal supervars ---
        p = pme1
        nleft = n - nel
        for pme in pme1:pme2
            i = Int(Iw[pme + 1])
            nvi = -Int(Nv[i + 1])
            if nvi > 0
                Nv[i + 1] = Ti(nvi)
                deg = Int(Degree[i + 1]) + degme - nvi
                cap = nleft - nvi
                if deg > cap
                    deg = cap
                end
                inext2 = Int(Head[deg + 1])
                if inext2 != EMPTY
                    Last[inext2 + 1] = Ti(i)
                end
                Next[i + 1] = Ti(inext2)
                Last[i + 1] = Ti(EMPTY)
                Head[deg + 1] = Ti(i)
                if deg < mindeg
                    mindeg = deg
                end
                Degree[i + 1] = Ti(deg)
                Iw[p + 1] = Ti(i); p += 1
            end
        end

        Nv[me + 1] = Ti(nvpiv)
        Len[me + 1] = Ti(p - pme1)
        if Len[me + 1] == 0
            Pe[me + 1] = Ti(EMPTY)
            W[me + 1] = Ti(0)
        end
        if elenme != 0
            pfree = p
        end

        # Accumulate nonzeros in L excluding the diagonal (amd_2.c Info[AMD_LNZ]).
        # The new element has nvpiv pivots; its contribution block including the
        # `ndense` dense rows/columns is (degme+ndense)-by-(degme+ndense).
        f = Float64(nvpiv)
        r = Float64(degme + ndense)
        lnz += f * r + (f - 1.0) * f / 2.0
    end

    # --- Post-ordering setup: restore Pe and Elen by FLIP-ing ---
    @inbounds for i in 1:n
        Pe[i] = Ti(_flip(Int(Pe[i])))
    end
    @inbounds for i in 1:n
        Elen[i] = Ti(_flip(Int(Elen[i])))
    end

    # --- Path compression: variables with Nv[i]=0 traverse to element ---
    @inbounds for i in 0:(n - 1)
        if Nv[i + 1] == 0
            j = Int(Pe[i + 1])
            if j == EMPTY
                continue
            end
            while Nv[j + 1] == 0
                j = Int(Pe[j + 1])
            end
            e = j
            j = i
            while Nv[j + 1] == 0
                jnext = Int(Pe[j + 1])
                Pe[j + 1] = Ti(e)
                j = jnext
            end
        end
    end

    # --- Postorder the assembly tree, output goes in W ---
    amd_postorder!(n, Pe, Nv, Elen, W, Head, Next, Last)

    # --- Build output permutation in Last, inverse in Next ---
    @inbounds for k in 1:n
        Head[k] = Ti(EMPTY)
        Next[k] = Ti(EMPTY)
    end
    @inbounds for e in 0:(n - 1)
        k = Int(W[e + 1])
        if k != EMPTY
            Head[k + 1] = Ti(e)
        end
    end

    nel = 0
    @inbounds for k in 0:(n - 1)
        e = Int(Head[k + 1])
        if e == EMPTY
            break
        end
        Next[e + 1] = Ti(nel)
        nel += Int(Nv[e + 1])
    end

    @inbounds for i in 0:(n - 1)
        if Nv[i + 1] == 0
            e = Int(Pe[i + 1])
            if e != EMPTY
                Next[i + 1] = Next[e + 1]
                Next[e + 1] += Ti(1)
            else
                Next[i + 1] = Ti(nel); nel += 1
            end
        end
    end

    @inbounds for i in 0:(n - 1)
        k = Int(Next[i + 1])
        Last[k + 1] = Ti(i)
    end
    return lnz
end

"""
    amd_postorder!(nn, Parent, Nv, Fsize, Order, Child, Sibling, Stack)

Postorder the assembly tree. `Order[i]` receives the postorder index of
node `i`. Mirrors `amd_postorder.c`.
"""
function amd_postorder!(
        nn::Int, Parent::Vector{Ti}, Nv::Vector{Ti}, Fsize::Vector{Ti},
        Order::Vector{Ti}, Child::Vector{Ti}, Sibling::Vector{Ti},
        Stack::Vector{Ti}
    ) where {Ti <: Integer}
    @inbounds for j in 1:nn
        Child[j] = Ti(EMPTY)
        Sibling[j] = Ti(EMPTY)
    end

    @inbounds for j in (nn - 1):-1:0
        if Nv[j + 1] > 0
            parent = Int(Parent[j + 1])
            if parent != EMPTY
                Sibling[j + 1] = Child[parent + 1]
                Child[parent + 1] = Ti(j)
            end
        end
    end

    @inbounds for i in 0:(nn - 1)
        if Nv[i + 1] > 0 && Child[i + 1] != Ti(EMPTY)
            fprev = EMPTY
            maxfrsize = EMPTY
            bigfprev = EMPTY
            bigf = EMPTY
            f = Int(Child[i + 1])
            while f != EMPTY
                frsize = Int(Fsize[f + 1])
                if frsize >= maxfrsize
                    maxfrsize = frsize
                    bigfprev = fprev
                    bigf = f
                end
                fprev = f
                f = Int(Sibling[f + 1])
            end
            fnext = Int(Sibling[bigf + 1])
            if fnext != EMPTY
                if bigfprev == EMPTY
                    Child[i + 1] = Ti(fnext)
                else
                    Sibling[bigfprev + 1] = Ti(fnext)
                end
                Sibling[bigf + 1] = Ti(EMPTY)
                Sibling[fprev + 1] = Ti(bigf)
            end
        end
    end

    @inbounds for i in 1:nn
        Order[i] = Ti(EMPTY)
    end

    k = 0
    @inbounds for i in 0:(nn - 1)
        if Parent[i + 1] == Ti(EMPTY) && Nv[i + 1] > 0
            # A childless root is its own (single-node) postorder: assigning
            # Order[i]=k inline is byte-identical to amd_post_tree!(i,...) for
            # that case and skips the call.  Matrices like the arrow head leave
            # hundreds of childless roots, so this elides hundreds of calls.
            if Child[i + 1] == Ti(EMPTY)
                Order[i + 1] = Ti(k)
                k += 1
            else
                k = amd_post_tree!(i, k, Child, Sibling, Order, Stack)
            end
        end
    end
    return nothing
end

"""
    amd_post_tree!(root, k, Child, Sibling, Order, Stack) -> k

Non-recursive postorder of a single tree rooted at `root`, starting
numbering at `k`. Mirrors `amd_post_tree.c`.
"""
function amd_post_tree!(
        root::Int, k_in::Int, Child::Vector{Ti}, Sibling::Vector{Ti},
        Order::Vector{Ti}, Stack::Vector{Ti}
    ) where {Ti <: Integer}
    k = k_in
    head = 0
    @inbounds Stack[1] = Ti(root)
    @inbounds while head >= 0
        i = Int(Stack[head + 1])
        if Child[i + 1] != Ti(EMPTY)
            f = Int(Child[i + 1])
            while f != EMPTY
                head += 1
                f = Int(Sibling[f + 1])
            end
            h = head
            f = Int(Child[i + 1])
            while f != EMPTY
                Stack[h + 1] = Ti(f)
                h -= 1
                f = Int(Sibling[f + 1])
            end
            Child[i + 1] = Ti(EMPTY)
        else
            head -= 1
            Order[i + 1] = Ti(k); k += 1
        end
    end
    return k
end

"""
    amd_1!(n, Ap, Ai, P, Pinv, Len, slen; dense_alpha, aggressive)

Build the A+A' representation in workspace and dispatch to `amd_2!`.
Mirrors `amd_1.c`.
"""
function amd_1!(
        n::Int, Ap::AbstractVector{Ti}, Ai::AbstractVector{Ti},
        P::Vector{Ti}, Pinv::Vector{Ti}, Len::Vector{Ti},
        slen::Int;
        dense_alpha::Float64 = AMD_DEFAULT_DENSE,
        aggressive::Bool = AMD_DEFAULT_AGGRESSIVE != 0
    ) where {Ti <: Integer}

    iwlen = slen - 6n
    # Allocate dedicated Vector{Ti} buffers up front; pass them by name to
    # amd_2!.  Using slices of `S` (via `view` + `collect`) duplicated all
    # data and forced an extra dispatch layer.
    Pe = Vector{Ti}(undef, n)
    Nv = Vector{Ti}(undef, n)
    Head = Vector{Ti}(undef, n)
    Elen = Vector{Ti}(undef, n)
    Degree = Vector{Ti}(undef, n)
    W = Vector{Ti}(undef, n)
    Iw = Vector{Ti}(undef, iwlen)

    # Use Nv and W as workspace for Sp and Tp during matrix construction
    Sp = Nv
    Tp = W

    pfree = 0
    @inbounds for j in 0:(n - 1)
        Pe[j + 1] = Ti(pfree)
        Sp[j + 1] = Ti(pfree)
        pfree += Int(Len[j + 1])
    end

    @inbounds for k in 0:(n - 1)
        p = Int(Ap[k + 1])
        p2 = Int(Ap[k + 2])
        while p < p2
            j = Int(Ai[p + 1])
            if j < k
                Iw[Int(Sp[j + 1]) + 1] = Ti(k)
                Sp[j + 1] += Ti(1)
                Iw[Int(Sp[k + 1]) + 1] = Ti(j)
                Sp[k + 1] += Ti(1)
                p += 1
            elseif j == k
                p += 1
                break
            else
                break
            end
            pj2 = Int(Ap[j + 2])
            pj = Int(Tp[j + 1])
            done = false
            while pj < pj2
                i = Int(Ai[pj + 1])
                if i < k
                    Iw[Int(Sp[i + 1]) + 1] = Ti(j)
                    Sp[i + 1] += Ti(1)
                    Iw[Int(Sp[j + 1]) + 1] = Ti(i)
                    Sp[j + 1] += Ti(1)
                    pj += 1
                elseif i == k
                    pj += 1
                    done = true
                    break
                else
                    break
                end
            end
            Tp[j + 1] = Ti(pj)
            if done
            end
        end
        Tp[k + 1] = Ti(p)
    end

    @inbounds for j in 0:(n - 1)
        pj = Int(Tp[j + 1])
        pjend = Int(Ap[j + 2])
        while pj < pjend
            i = Int(Ai[pj + 1])
            Iw[Int(Sp[i + 1]) + 1] = Ti(j)
            Sp[i + 1] += Ti(1)
            Iw[Int(Sp[j + 1]) + 1] = Ti(i)
            Sp[j + 1] += Ti(1)
            pj += 1
        end
    end

    # Pinv and P are output buffers (size n)
    lnz = amd_2!(
        n, Pe, Iw, Len, iwlen, pfree,
        Nv, Pinv, P, Head, Elen, Degree, W;
        dense_alpha, aggressive
    )
    return lnz
end

"""
    amd_order!(n, Ap, Ai, P; dense_alpha=10.0, aggressive=true) -> (status, lnz)

Compute the AMD ordering of the symmetric pattern of `A+A'`. `P` is the
output permutation. Mirrors `amd_order.c`. Returns `(AMD_OK, lnz)` on
success, where `lnz` is the number of off-diagonal nonzeros in `L`
(SuiteSparse's `Info[AMD_LNZ]`). On failure returns `(status, 0.0)`.
"""
function amd_order!(
        n::Int, Ap::AbstractVector{Ti}, Ai::AbstractVector{Ti},
        P::Vector{Ti};
        dense_alpha::Float64 = AMD_DEFAULT_DENSE,
        aggressive::Bool = AMD_DEFAULT_AGGRESSIVE != 0
    ) where {Ti <: Integer}
    n <= 0 && return (AMD_INVALID, 0.0)
    if n == 0
        return (AMD_OK, 0.0)
    end

    nz = Int(Ap[n + 1])
    if nz < 0
        return (AMD_INVALID, 0.0)
    end

    Len = Vector{Ti}(undef, n)
    Pinv = Vector{Ti}(undef, n)

    # Caller is expected to have validated the matrix.  Assume sorted-and-deduped.
    Tp = Pinv  # use Pinv as scratch
    nzaat = amd_aat!(n, Ap, Ai, Len, Tp)

    slen = nzaat + nzaat ÷ 5
    slen += 7 * n

    lnz = amd_1!(
        n, Ap, Ai, P, Pinv, Len, slen;
        dense_alpha, aggressive
    )
    return (AMD_OK, lnz)
end

end # module
