function SuiteSparse_config_printf_func_get()
    ccall((:SuiteSparse_config_printf_func_get, libsuitesparseconfig), Ptr{Cvoid}, ())
end

function SuiteSparse_config_malloc_func_get()
    ccall((:SuiteSparse_config_malloc_func_get, libsuitesparseconfig), Ptr{Cvoid}, ())
end

function SuiteSparse_config_calloc_func_get()
    ccall((:SuiteSparse_config_calloc_func_get, libsuitesparseconfig), Ptr{Cvoid}, ())
end

function SuiteSparse_config_realloc_func_get()
    ccall((:SuiteSparse_config_realloc_func_get, libsuitesparseconfig), Ptr{Cvoid}, ())
end

function SuiteSparse_config_free_func_get()
    ccall((:SuiteSparse_config_free_func_get, libsuitesparseconfig), Ptr{Cvoid}, ())
end

function SuiteSparse_config_hypot_func_get()
    ccall((:SuiteSparse_config_hypot_func_get, libsuitesparseconfig), Ptr{Cvoid}, ())
end

function SuiteSparse_config_divcomplex_func_get()
    ccall((:SuiteSparse_config_divcomplex_func_get, libsuitesparseconfig), Ptr{Cvoid}, ())
end

function SuiteSparse_config_malloc_func_set(malloc_func)
    ccall((:SuiteSparse_config_malloc_func_set, libsuitesparseconfig),
        Cvoid, (Ptr{Cvoid},), malloc_func)
end

function SuiteSparse_config_calloc_func_set(calloc_func)
    ccall((:SuiteSparse_config_calloc_func_set, libsuitesparseconfig),
        Cvoid, (Ptr{Cvoid},), calloc_func)
end

function SuiteSparse_config_realloc_func_set(realloc_func)
    ccall((:SuiteSparse_config_realloc_func_set, libsuitesparseconfig),
        Cvoid, (Ptr{Cvoid},), realloc_func)
end

function SuiteSparse_config_free_func_set(free_func)
    ccall((:SuiteSparse_config_free_func_set, libsuitesparseconfig),
        Cvoid, (Ptr{Cvoid},), free_func)
end

function SuiteSparse_config_printf_func_set(printf_func)
    ccall((:SuiteSparse_config_printf_func_set, libsuitesparseconfig),
        Cvoid, (Ptr{Cvoid},), printf_func)
end

function SuiteSparse_config_hypot_func_set(hypot_func)
    ccall((:SuiteSparse_config_hypot_func_set, libsuitesparseconfig),
        Cvoid, (Ptr{Cvoid},), hypot_func)
end

function SuiteSparse_config_divcomplex_func_set(divcomplex_func)
    ccall((:SuiteSparse_config_divcomplex_func_set, libsuitesparseconfig),
        Cvoid, (Ptr{Cvoid},), divcomplex_func)
end

function SuiteSparse_config_malloc(s)
    ccall((:SuiteSparse_config_malloc, libsuitesparseconfig), Ptr{Cvoid}, (Csize_t,), s)
end

function SuiteSparse_config_calloc(n, s)
    ccall((:SuiteSparse_config_calloc, libsuitesparseconfig),
        Ptr{Cvoid}, (Csize_t, Csize_t), n, s)
end

function SuiteSparse_config_realloc(arg1, s)
    ccall((:SuiteSparse_config_realloc, libsuitesparseconfig),
        Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t), arg1, s)
end

function SuiteSparse_config_free(arg1)
    ccall((:SuiteSparse_config_free, libsuitesparseconfig), Cvoid, (Ptr{Cvoid},), arg1)
end

function SuiteSparse_config_hypot(x, y)
    ccall((:SuiteSparse_config_hypot, libsuitesparseconfig),
        Cdouble, (Cdouble, Cdouble), x, y)
end

function SuiteSparse_config_divcomplex(xr, xi, yr, yi, zr, zi)
    ccall((:SuiteSparse_config_divcomplex, libsuitesparseconfig), Cint,
        (Cdouble, Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}),
        xr, xi, yr, yi, zr, zi)
end

function SuiteSparse_start()
    ccall((:SuiteSparse_start, libsuitesparseconfig), Cvoid, ())
end

function SuiteSparse_finish()
    ccall((:SuiteSparse_finish, libsuitesparseconfig), Cvoid, ())
end

function SuiteSparse_malloc(nitems, size_of_item)
    ccall((:SuiteSparse_malloc, libsuitesparseconfig),
        Ptr{Cvoid}, (Csize_t, Csize_t), nitems, size_of_item)
end

function SuiteSparse_calloc(nitems, size_of_item)
    ccall((:SuiteSparse_calloc, libsuitesparseconfig),
        Ptr{Cvoid}, (Csize_t, Csize_t), nitems, size_of_item)
end

function SuiteSparse_realloc(nitems_new, nitems_old, size_of_item, p, ok)
    ccall((:SuiteSparse_realloc, libsuitesparseconfig), Ptr{Cvoid},
        (Csize_t, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cint}),
        nitems_new, nitems_old, size_of_item, p, ok)
end

function SuiteSparse_free(p)
    ccall((:SuiteSparse_free, libsuitesparseconfig), Ptr{Cvoid}, (Ptr{Cvoid},), p)
end

function SuiteSparse_tic(tic)
    ccall((:SuiteSparse_tic, libsuitesparseconfig), Cvoid, (Ptr{Cdouble},), tic)
end

function SuiteSparse_toc(tic)
    ccall((:SuiteSparse_toc, libsuitesparseconfig), Cdouble, (Ptr{Cdouble},), tic)
end

function SuiteSparse_time()
    ccall((:SuiteSparse_time, libsuitesparseconfig), Cdouble, ())
end

function SuiteSparse_hypot(x, y)
    ccall((:SuiteSparse_hypot, libsuitesparseconfig), Cdouble, (Cdouble, Cdouble), x, y)
end

function SuiteSparse_divcomplex(ar, ai, br, bi, cr, ci)
    ccall((:SuiteSparse_divcomplex, libsuitesparseconfig), Cint,
        (Cdouble, Cdouble, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{Cdouble}),
        ar, ai, br, bi, cr, ci)
end

function SuiteSparse_version(version)
    ccall((:SuiteSparse_version, libsuitesparseconfig), Cint, (Ptr{Cint},), version)
end

function SuiteSparse_BLAS_library()
    ccall((:SuiteSparse_BLAS_library, libsuitesparseconfig), Ptr{Cchar}, ())
end

function SuiteSparse_BLAS_integer_size()
    ccall((:SuiteSparse_BLAS_integer_size, libsuitesparseconfig), Csize_t, ())
end

function amd_order(n, Ap, Ai, P, Control, Info)
    ccall((:amd_order, libamd), Cint,
        (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}),
        n, Ap, Ai, P, Control, Info)
end

function amd_l_order(n, Ap, Ai, P, Control, Info)
    ccall((:amd_l_order, libamd), Cint,
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}),
        n, Ap, Ai, P, Control, Info)
end

function amd_2(
        n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W, Control, Info)
    ccall((:amd_2, libamd),
        Cvoid,
        (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Int32, Int32,
            Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
            Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}),
        n,
        Pe,
        Iw,
        Len,
        iwlen,
        pfree,
        Nv,
        Next,
        Last,
        Head,
        Elen,
        Degree,
        W,
        Control,
        Info)
end

function amd_l2(
        n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W, Control, Info)
    ccall((:amd_l2, libamd),
        Cvoid,
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Int64, Int64,
            Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
            Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}),
        n,
        Pe,
        Iw,
        Len,
        iwlen,
        pfree,
        Nv,
        Next,
        Last,
        Head,
        Elen,
        Degree,
        W,
        Control,
        Info)
end

function amd_valid(n_row, n_col, Ap, Ai)
    ccall((:amd_valid, libamd), Cint,
        (Int32, Int32, Ptr{Int32}, Ptr{Int32}), n_row, n_col, Ap, Ai)
end

function amd_l_valid(n_row, n_col, Ap, Ai)
    ccall((:amd_l_valid, libamd), Cint,
        (Int64, Int64, Ptr{Int64}, Ptr{Int64}), n_row, n_col, Ap, Ai)
end

function amd_defaults(Control)
    ccall((:amd_defaults, libamd), Cvoid, (Ptr{Cdouble},), Control)
end

function amd_l_defaults(Control)
    ccall((:amd_l_defaults, libamd), Cvoid, (Ptr{Cdouble},), Control)
end

function amd_control(Control)
    ccall((:amd_control, libamd), Cvoid, (Ptr{Cdouble},), Control)
end

function amd_l_control(Control)
    ccall((:amd_l_control, libamd), Cvoid, (Ptr{Cdouble},), Control)
end

function amd_info(Info)
    ccall((:amd_info, libamd), Cvoid, (Ptr{Cdouble},), Info)
end

function amd_l_info(Info)
    ccall((:amd_l_info, libamd), Cvoid, (Ptr{Cdouble},), Info)
end

function colamd_recommended(nnz, n_row, n_col)
    ccall((:colamd_recommended, libamd), Csize_t, (Int32, Int32, Int32), nnz, n_row, n_col)
end

function colamd_l_recommended(nnz, n_row, n_col)
    ccall(
        (:colamd_l_recommended, libamd), Csize_t, (Int64, Int64, Int64), nnz, n_row, n_col)
end

function colamd_set_defaults(knobs)
    ccall((:colamd_set_defaults, libamd), Cvoid, (Ptr{Cdouble},), knobs)
end

function colamd_l_set_defaults(knobs)
    ccall((:colamd_l_set_defaults, libamd), Cvoid, (Ptr{Cdouble},), knobs)
end

function colamd(n_row, n_col, Alen, A, p, knobs, stats)
    ccall((:colamd, libamd), Cint,
        (Int32, Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}),
        n_row, n_col, Alen, A, p, knobs, stats)
end

function colamd_l(n_row, n_col, Alen, A, p, knobs, stats)
    ccall((:colamd_l, libamd), Cint,
        (Int64, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}),
        n_row, n_col, Alen, A, p, knobs, stats)
end

function symamd(n, A, p, perm, knobs, stats, allocate, release)
    ccall((:symamd, libamd),
        Cint,
        (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
            Ptr{Cdouble}, Ptr{Int32}, Ptr{Cvoid}, Ptr{Cvoid}),
        n,
        A,
        p,
        perm,
        knobs,
        stats,
        allocate,
        release)
end

function symamd_l(n, A, p, perm, knobs, stats, allocate, release)
    ccall((:symamd_l, libamd),
        Cint,
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
            Ptr{Cdouble}, Ptr{Int64}, Ptr{Cvoid}, Ptr{Cvoid}),
        n,
        A,
        p,
        perm,
        knobs,
        stats,
        allocate,
        release)
end

function colamd_report(stats)
    ccall((:colamd_report, libamd), Cvoid, (Ptr{Int32},), stats)
end

function colamd_l_report(stats)
    ccall((:colamd_l_report, libamd), Cvoid, (Ptr{Int64},), stats)
end

function symamd_report(stats)
    ccall((:symamd_report, libamd), Cvoid, (Ptr{Int32},), stats)
end

function symamd_l_report(stats)
    ccall((:symamd_l_report, libamd), Cvoid, (Ptr{Int64},), stats)
end

function btf_maxtrans(nrow, ncol, Ap, Ai, maxwork, work, Match, Work)
    ccall((:btf_maxtrans, libbtf),
        Int32,
        (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Cdouble,
            Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}),
        nrow,
        ncol,
        Ap,
        Ai,
        maxwork,
        work,
        Match,
        Work)
end

function btf_l_maxtrans(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:btf_l_maxtrans, libbtf),
        Int64,
        (Int64, Int64, Ptr{Int64}, Ptr{Int64}, Cdouble,
            Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8)
end

function btf_strongcomp(n, Ap, Ai, Q, P, R, Work)
    ccall((:btf_strongcomp, libbtf), Int32,
        (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}),
        n, Ap, Ai, Q, P, R, Work)
end

function btf_l_strongcomp(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:btf_l_strongcomp, libbtf), Int64,
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}),
        arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function btf_order(n, Ap, Ai, maxwork, work, P, Q, R, nmatch, Work)
    ccall((:btf_order, libbtf),
        Int32,
        (Int32, Ptr{Int32}, Ptr{Int32}, Cdouble, Ptr{Cdouble},
            Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}),
        n,
        Ap,
        Ai,
        maxwork,
        work,
        P,
        Q,
        R,
        nmatch,
        Work)
end

function btf_l_order(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    ccall((:btf_l_order, libbtf),
        Int64,
        (Int64, Ptr{Int64}, Ptr{Int64}, Cdouble, Ptr{Cdouble},
            Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10)
end

mutable struct klu_symbolic
    symmetry::Cdouble
    est_flops::Cdouble
    lnz::Cdouble
    unz::Cdouble
    Lnz::Ptr{Cdouble}
    n::Int32
    nz::Int32
    P::Ptr{Int32}
    Q::Ptr{Int32}
    R::Ptr{Int32}
    nzoff::Int32
    nblocks::Int32
    maxblock::Int32
    ordering::Int32
    do_btf::Int32
    structural_rank::Int32
    klu_symbolic() = new()
end

mutable struct klu_l_symbolic
    symmetry::Cdouble
    est_flops::Cdouble
    lnz::Cdouble
    unz::Cdouble
    Lnz::Ptr{Cdouble}
    n::Int64
    nz::Int64
    P::Ptr{Int64}
    Q::Ptr{Int64}
    R::Ptr{Int64}
    nzoff::Int64
    nblocks::Int64
    maxblock::Int64
    ordering::Int64
    do_btf::Int64
    structural_rank::Int64
    klu_l_symbolic() = new()
end

mutable struct klu_numeric
    n::Int32
    nblocks::Int32
    lnz::Int32
    unz::Int32
    max_lnz_block::Int32
    max_unz_block::Int32
    Pnum::Ptr{Int32}
    Pinv::Ptr{Int32}
    Lip::Ptr{Int32}
    Uip::Ptr{Int32}
    Llen::Ptr{Int32}
    Ulen::Ptr{Int32}
    LUbx::Ptr{Ptr{Cvoid}}
    LUsize::Ptr{Csize_t}
    Udiag::Ptr{Cvoid}
    Rs::Ptr{Cdouble}
    worksize::Csize_t
    Work::Ptr{Cvoid}
    Xwork::Ptr{Cvoid}
    Iwork::Ptr{Int32}
    Offp::Ptr{Int32}
    Offi::Ptr{Int32}
    Offx::Ptr{Cvoid}
    nzoff::Int32
    klu_numeric() = new()
end

mutable struct klu_l_numeric
    n::Int64
    nblocks::Int64
    lnz::Int64
    unz::Int64
    max_lnz_block::Int64
    max_unz_block::Int64
    Pnum::Ptr{Int64}
    Pinv::Ptr{Int64}
    Lip::Ptr{Int64}
    Uip::Ptr{Int64}
    Llen::Ptr{Int64}
    Ulen::Ptr{Int64}
    LUbx::Ptr{Ptr{Cvoid}}
    LUsize::Ptr{Csize_t}
    Udiag::Ptr{Cvoid}
    Rs::Ptr{Cdouble}
    worksize::Csize_t
    Work::Ptr{Cvoid}
    Xwork::Ptr{Cvoid}
    Iwork::Ptr{Int64}
    Offp::Ptr{Int64}
    Offi::Ptr{Int64}
    Offx::Ptr{Cvoid}
    nzoff::Int64
    klu_l_numeric() = new()
end

mutable struct klu_common_struct
    tol::Cdouble
    memgrow::Cdouble
    initmem_amd::Cdouble
    initmem::Cdouble
    maxwork::Cdouble
    btf::Cint
    ordering::Cint
    scale::Cint
    user_order::Ptr{Cvoid}
    user_data::Ptr{Cvoid}
    halt_if_singular::Cint
    status::Cint
    nrealloc::Cint
    structural_rank::Int32
    numerical_rank::Int32
    singular_col::Int32
    noffdiag::Int32
    flops::Cdouble
    rcond::Cdouble
    condest::Cdouble
    rgrowth::Cdouble
    work::Cdouble
    memusage::Csize_t
    mempeak::Csize_t
    klu_common_struct() = new()
end

const klu_common = klu_common_struct

mutable struct klu_l_common_struct
    tol::Cdouble
    memgrow::Cdouble
    initmem_amd::Cdouble
    initmem::Cdouble
    maxwork::Cdouble
    btf::Cint
    ordering::Cint
    scale::Cint
    user_order::Ptr{Cvoid}
    user_data::Ptr{Cvoid}
    halt_if_singular::Cint
    status::Cint
    nrealloc::Cint
    structural_rank::Int64
    numerical_rank::Int64
    singular_col::Int64
    noffdiag::Int64
    flops::Cdouble
    rcond::Cdouble
    condest::Cdouble
    rgrowth::Cdouble
    work::Cdouble
    memusage::Csize_t
    mempeak::Csize_t
    klu_l_common_struct() = new()
end

const klu_l_common = klu_l_common_struct

function klu_defaults(Common)
    ccall((:klu_defaults, libklu), Cint, (Ptr{klu_common},), Common)
end

function klu_l_defaults(Common)
    ccall((:klu_l_defaults, libklu), Cint, (Ptr{klu_l_common},), Common)
end

function klu_analyze(n, Ap, Ai, Common)
    ccall((:klu_analyze, libklu), Ptr{klu_symbolic},
        (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{klu_common}), n, Ap, Ai, Common)
end

function klu_l_analyze(arg1, arg2, arg3, Common)
    ccall((:klu_l_analyze, libklu), Ptr{klu_l_symbolic},
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{klu_l_common}), arg1, arg2, arg3, Common)
end

function klu_analyze_given(n, Ap, Ai, P, Q, Common)
    ccall((:klu_analyze_given, libklu), Ptr{klu_symbolic},
        (Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{klu_common}),
        n, Ap, Ai, P, Q, Common)
end

function klu_l_analyze_given(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:klu_l_analyze_given, libklu), Ptr{klu_l_symbolic},
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{klu_l_common}),
        arg1, arg2, arg3, arg4, arg5, arg6)
end

function klu_factor(Ap, Ai, Ax, Symbolic, Common)
    ccall((:klu_factor, libklu), Ptr{klu_numeric},
        (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{klu_symbolic}, Ptr{klu_common}),
        Ap, Ai, Ax, Symbolic, Common)
end

function klu_z_factor(Ap, Ai, Ax, Symbolic, Common)
    ccall((:klu_z_factor, libklu), Ptr{klu_numeric},
        (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{klu_symbolic}, Ptr{klu_common}),
        Ap, Ai, Ax, Symbolic, Common)
end

function klu_l_factor(arg1, arg2, arg3, arg4, arg5)
    ccall((:klu_l_factor, libklu), Ptr{klu_l_numeric},
        (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{klu_l_symbolic}, Ptr{klu_l_common}),
        arg1, arg2, arg3, arg4, arg5)
end

function klu_zl_factor(arg1, arg2, arg3, arg4, arg5)
    ccall((:klu_zl_factor, libklu), Ptr{klu_l_numeric},
        (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{klu_l_symbolic}, Ptr{klu_l_common}),
        arg1, arg2, arg3, arg4, arg5)
end

function klu_solve(Symbolic, Numeric, ldim, nrhs, B, Common)
    ccall((:klu_solve, libklu), Cint,
        (Ptr{klu_symbolic}, Ptr{klu_numeric}, Int32, Int32, Ptr{Cdouble}, Ptr{klu_common}),
        Symbolic, Numeric, ldim, nrhs, B, Common)
end

function klu_z_solve(Symbolic, Numeric, ldim, nrhs, B, Common)
    ccall((:klu_z_solve, libklu), Cint,
        (Ptr{klu_symbolic}, Ptr{klu_numeric}, Int32, Int32, Ptr{Cdouble}, Ptr{klu_common}),
        Symbolic, Numeric, ldim, nrhs, B, Common)
end

function klu_l_solve(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:klu_l_solve, libklu),
        Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Int64,
            Int64, Ptr{Cdouble}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6)
end

function klu_zl_solve(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:klu_zl_solve, libklu),
        Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Int64,
            Int64, Ptr{Cdouble}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6)
end

function klu_tsolve(Symbolic, Numeric, ldim, nrhs, B, Common)
    ccall((:klu_tsolve, libklu), Cint,
        (Ptr{klu_symbolic}, Ptr{klu_numeric}, Int32, Int32, Ptr{Cdouble}, Ptr{klu_common}),
        Symbolic, Numeric, ldim, nrhs, B, Common)
end

function klu_z_tsolve(Symbolic, Numeric, ldim, nrhs, B, conj_solve, Common)
    ccall((:klu_z_tsolve, libklu),
        Cint,
        (Ptr{klu_symbolic}, Ptr{klu_numeric}, Int32,
            Int32, Ptr{Cdouble}, Cint, Ptr{klu_common}),
        Symbolic,
        Numeric,
        ldim,
        nrhs,
        B,
        conj_solve,
        Common)
end

function klu_l_tsolve(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:klu_l_tsolve, libklu),
        Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Int64,
            Int64, Ptr{Cdouble}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6)
end

function klu_zl_tsolve(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:klu_zl_tsolve, libklu),
        Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Int64,
            Int64, Ptr{Cdouble}, Cint, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7)
end

function klu_refactor(Ap, Ai, Ax, Symbolic, Numeric, Common)
    ccall((:klu_refactor, libklu),
        Cint,
        (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble},
            Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}),
        Ap,
        Ai,
        Ax,
        Symbolic,
        Numeric,
        Common)
end

function klu_z_refactor(Ap, Ai, Ax, Symbolic, Numeric, Common)
    ccall((:klu_z_refactor, libklu),
        Cint,
        (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble},
            Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}),
        Ap,
        Ai,
        Ax,
        Symbolic,
        Numeric,
        Common)
end

function klu_l_refactor(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:klu_l_refactor, libklu),
        Cint,
        (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{klu_l_symbolic},
            Ptr{klu_l_numeric}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6)
end

function klu_zl_refactor(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:klu_zl_refactor, libklu),
        Cint,
        (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{klu_l_symbolic},
            Ptr{klu_l_numeric}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6)
end

function klu_free_symbolic(Symbolic, Common)
    ccall((:klu_free_symbolic, libklu), Cint,
        (Ptr{Ptr{klu_symbolic}}, Ptr{klu_common}), Symbolic, Common)
end

function klu_l_free_symbolic(arg1, arg2)
    ccall((:klu_l_free_symbolic, libklu), Cint,
        (Ptr{Ptr{klu_l_symbolic}}, Ptr{klu_l_common}), arg1, arg2)
end

function klu_free_numeric(Numeric, Common)
    ccall((:klu_free_numeric, libklu), Cint,
        (Ptr{Ptr{klu_numeric}}, Ptr{klu_common}), Numeric, Common)
end

function klu_z_free_numeric(Numeric, Common)
    ccall((:klu_z_free_numeric, libklu), Cint,
        (Ptr{Ptr{klu_numeric}}, Ptr{klu_common}), Numeric, Common)
end

function klu_l_free_numeric(arg1, arg2)
    ccall((:klu_l_free_numeric, libklu), Cint,
        (Ptr{Ptr{klu_l_numeric}}, Ptr{klu_l_common}), arg1, arg2)
end

function klu_zl_free_numeric(arg1, arg2)
    ccall((:klu_zl_free_numeric, libklu), Cint,
        (Ptr{Ptr{klu_l_numeric}}, Ptr{klu_l_common}), arg1, arg2)
end

function klu_sort(Symbolic, Numeric, Common)
    ccall(
        (:klu_sort, libklu), Cint, (Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}),
        Symbolic, Numeric, Common)
end

function klu_z_sort(Symbolic, Numeric, Common)
    ccall((:klu_z_sort, libklu), Cint,
        (Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}), Symbolic, Numeric, Common)
end

function klu_l_sort(arg1, arg2, arg3)
    ccall((:klu_l_sort, libklu), Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Ptr{klu_l_common}), arg1, arg2, arg3)
end

function klu_zl_sort(arg1, arg2, arg3)
    ccall((:klu_zl_sort, libklu), Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Ptr{klu_l_common}), arg1, arg2, arg3)
end

function klu_flops(Symbolic, Numeric, Common)
    ccall((:klu_flops, libklu), Cint,
        (Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}), Symbolic, Numeric, Common)
end

function klu_z_flops(Symbolic, Numeric, Common)
    ccall((:klu_z_flops, libklu), Cint,
        (Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}), Symbolic, Numeric, Common)
end

function klu_l_flops(arg1, arg2, arg3)
    ccall((:klu_l_flops, libklu), Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Ptr{klu_l_common}), arg1, arg2, arg3)
end

function klu_zl_flops(arg1, arg2, arg3)
    ccall((:klu_zl_flops, libklu), Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Ptr{klu_l_common}), arg1, arg2, arg3)
end

function klu_rgrowth(Ap, Ai, Ax, Symbolic, Numeric, Common)
    ccall((:klu_rgrowth, libklu),
        Cint,
        (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble},
            Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}),
        Ap,
        Ai,
        Ax,
        Symbolic,
        Numeric,
        Common)
end

function klu_z_rgrowth(Ap, Ai, Ax, Symbolic, Numeric, Common)
    ccall((:klu_z_rgrowth, libklu),
        Cint,
        (Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble},
            Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}),
        Ap,
        Ai,
        Ax,
        Symbolic,
        Numeric,
        Common)
end

function klu_l_rgrowth(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:klu_l_rgrowth, libklu),
        Cint,
        (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{klu_l_symbolic},
            Ptr{klu_l_numeric}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6)
end

function klu_zl_rgrowth(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:klu_zl_rgrowth, libklu),
        Cint,
        (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{klu_l_symbolic},
            Ptr{klu_l_numeric}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6)
end

function klu_condest(Ap, Ax, Symbolic, Numeric, Common)
    ccall((:klu_condest, libklu), Cint,
        (Ptr{Int32}, Ptr{Cdouble}, Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}),
        Ap, Ax, Symbolic, Numeric, Common)
end

function klu_z_condest(Ap, Ax, Symbolic, Numeric, Common)
    ccall((:klu_z_condest, libklu), Cint,
        (Ptr{Int32}, Ptr{Cdouble}, Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}),
        Ap, Ax, Symbolic, Numeric, Common)
end

function klu_l_condest(arg1, arg2, arg3, arg4, arg5)
    ccall((:klu_l_condest, libklu),
        Cint,
        (Ptr{Int64}, Ptr{Cdouble}, Ptr{klu_l_symbolic},
            Ptr{klu_l_numeric}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5)
end

function klu_zl_condest(arg1, arg2, arg3, arg4, arg5)
    ccall((:klu_zl_condest, libklu),
        Cint,
        (Ptr{Int64}, Ptr{Cdouble}, Ptr{klu_l_symbolic},
            Ptr{klu_l_numeric}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5)
end

function klu_rcond(Symbolic, Numeric, Common)
    ccall((:klu_rcond, libklu), Cint,
        (Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}), Symbolic, Numeric, Common)
end

function klu_z_rcond(Symbolic, Numeric, Common)
    ccall((:klu_z_rcond, libklu), Cint,
        (Ptr{klu_symbolic}, Ptr{klu_numeric}, Ptr{klu_common}), Symbolic, Numeric, Common)
end

function klu_l_rcond(arg1, arg2, arg3)
    ccall((:klu_l_rcond, libklu), Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Ptr{klu_l_common}), arg1, arg2, arg3)
end

function klu_zl_rcond(arg1, arg2, arg3)
    ccall((:klu_zl_rcond, libklu), Cint,
        (Ptr{klu_l_symbolic}, Ptr{klu_l_numeric}, Ptr{klu_l_common}), arg1, arg2, arg3)
end

function klu_scale(scale, n, Ap, Ai, Ax, Rs, W, Common)
    ccall((:klu_scale, libklu),
        Cint,
        (Cint, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble},
            Ptr{Cdouble}, Ptr{Int32}, Ptr{klu_common}),
        scale,
        n,
        Ap,
        Ai,
        Ax,
        Rs,
        W,
        Common)
end

function klu_z_scale(scale, n, Ap, Ai, Ax, Rs, W, Common)
    ccall((:klu_z_scale, libklu),
        Cint,
        (Cint, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble},
            Ptr{Cdouble}, Ptr{Int32}, Ptr{klu_common}),
        scale,
        n,
        Ap,
        Ai,
        Ax,
        Rs,
        W,
        Common)
end

function klu_l_scale(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:klu_l_scale, libklu),
        Cint,
        (Cint, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble},
            Ptr{Cdouble}, Ptr{Int64}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8)
end

function klu_zl_scale(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:klu_zl_scale, libklu),
        Cint,
        (Cint, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble},
            Ptr{Cdouble}, Ptr{Int64}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8)
end

function klu_extract(
        Numeric, Symbolic, Lp, Li, Lx, Up, Ui, Ux, Fp, Fi, Fx, P, Q, Rs, R, Common)
    ccall((:klu_extract, libklu),
        Cint,
        (Ptr{klu_numeric}, Ptr{klu_symbolic}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble},
            Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble},
            Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{klu_common}),
        Numeric,
        Symbolic,
        Lp,
        Li,
        Lx,
        Up,
        Ui,
        Ux,
        Fp,
        Fi,
        Fx,
        P,
        Q,
        Rs,
        R,
        Common)
end

function klu_z_extract(Numeric, Symbolic, Lp, Li, Lx, Lz, Up, Ui,
        Ux, Uz, Fp, Fi, Fx, Fz, P, Q, Rs, R, Common)
    ccall((:klu_z_extract, libklu),
        Cint,
        (Ptr{klu_numeric}, Ptr{klu_symbolic}, Ptr{Int32}, Ptr{Int32},
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble},
            Ptr{Cdouble}, Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble},
            Ptr{Int32}, Ptr{Int32}, Ptr{Cdouble}, Ptr{Int32}, Ptr{klu_common}),
        Numeric,
        Symbolic,
        Lp,
        Li,
        Lx,
        Lz,
        Up,
        Ui,
        Ux,
        Uz,
        Fp,
        Fi,
        Fx,
        Fz,
        P,
        Q,
        Rs,
        R,
        Common)
end

function klu_l_extract(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
        arg10, arg11, arg12, arg13, arg14, arg15, arg16)
    ccall((:klu_l_extract, libklu),
        Cint,
        (Ptr{klu_l_numeric}, Ptr{klu_l_symbolic}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble},
            Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble},
            Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg16)
end

function klu_zl_extract(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10,
        arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19)
    ccall((:klu_zl_extract, libklu),
        Cint,
        (Ptr{klu_l_numeric}, Ptr{klu_l_symbolic}, Ptr{Int64}, Ptr{Int64},
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble},
            Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble},
            Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}, Ptr{klu_l_common}),
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg16,
        arg17,
        arg18,
        arg19)
end

function klu_malloc(n, size, Common)
    ccall((:klu_malloc, libklu), Ptr{Cvoid},
        (Csize_t, Csize_t, Ptr{klu_common}), n, size, Common)
end

function klu_free(p, n, size, Common)
    ccall((:klu_free, libklu), Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{klu_common}), p, n, size, Common)
end

function klu_realloc(nnew, nold, size, p, Common)
    ccall((:klu_realloc, libklu), Ptr{Cvoid},
        (Csize_t, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{klu_common}),
        nnew, nold, size, p, Common)
end

function klu_l_malloc(arg1, arg2, arg3)
    ccall((:klu_l_malloc, libklu), Ptr{Cvoid},
        (Csize_t, Csize_t, Ptr{klu_l_common}), arg1, arg2, arg3)
end

function klu_l_free(arg1, arg2, arg3, arg4)
    ccall((:klu_l_free, libklu), Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t, Csize_t, Ptr{klu_l_common}), arg1, arg2, arg3, arg4)
end

function klu_l_realloc(arg1, arg2, arg3, arg4, arg5)
    ccall((:klu_l_realloc, libklu), Ptr{Cvoid},
        (Csize_t, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{klu_l_common}),
        arg1, arg2, arg3, arg4, arg5)
end

const SUITESPARSE_OPENMP_MAX_THREADS = 1

const SUITESPARSE_OPENMP_GET_NUM_THREADS = 1

const SUITESPARSE_OPENMP_GET_WTIME = 0

const SUITESPARSE_OPENMP_GET_THREAD_ID = 0

const SUITESPARSE_COMPILER_NVCC = 0

const SUITESPARSE_COMPILER_ICX = 0

const SUITESPARSE_COMPILER_ICC = 0

const SUITESPARSE_COMPILER_CLANG = 0

const SUITESPARSE_COMPILER_GCC = 0

const SUITESPARSE_COMPILER_MSC = 0

const SUITESPARSE_COMPILER_XLC = 0

const SUITESPARSE_DATE = "Oct 7, 2023"

const SUITESPARSE_MAIN_VERSION = 7

const SUITESPARSE_SUB_VERSION = 2

const SUITESPARSE_SUBSUB_VERSION = 1

SUITESPARSE_VER_CODE(main, sub) = main * 1000 + sub

const SUITESPARSE_VERSION = SUITESPARSE_VER_CODE(
    SUITESPARSE_MAIN_VERSION, SUITESPARSE_SUB_VERSION)

const AMD_CONTROL = 5

const AMD_INFO = 20

const AMD_DENSE = 0

const AMD_AGGRESSIVE = 1

const AMD_DEFAULT_DENSE = 10.0

const AMD_DEFAULT_AGGRESSIVE = 1

const AMD_STATUS = 0

const AMD_N = 1

const AMD_NZ = 2

const AMD_SYMMETRY = 3

const AMD_NZDIAG = 4

const AMD_NZ_A_PLUS_AT = 5

const AMD_NDENSE = 6

const AMD_MEMORY = 7

const AMD_NCMPA = 8

const AMD_LNZ = 9

const AMD_NDIV = 10

const AMD_NMULTSUBS_LDL = 11

const AMD_NMULTSUBS_LU = 12

const AMD_DMAX = 13

const AMD_OK = 0

const AMD_OUT_OF_MEMORY = -1

const AMD_INVALID = -2

const AMD_OK_BUT_JUMBLED = 1

const AMD_DATE = "Sept 18, 2023"

const AMD_MAIN_VERSION = 3

const AMD_SUB_VERSION = 2

const AMD_SUBSUB_VERSION = 1

AMD_VERSION_CODE(main, sub) = main * 1000 + sub

const AMD_VERSION = AMD_VERSION_CODE(AMD_MAIN_VERSION, AMD_SUB_VERSION)

const COLAMD_DATE = "Sept 18, 2023"

const COLAMD_MAIN_VERSION = 3

const COLAMD_SUB_VERSION = 2

const COLAMD_SUBSUB_VERSION = 1

COLAMD_VERSION_CODE(main, sub) = main * 1000 + sub

const COLAMD_VERSION = COLAMD_VERSION_CODE(COLAMD_MAIN_VERSION, COLAMD_SUB_VERSION)

const COLAMD_KNOBS = 20

const COLAMD_STATS = 20

const COLAMD_DENSE_ROW = 0

const COLAMD_DENSE_COL = 1

const COLAMD_AGGRESSIVE = 2

const COLAMD_DEFRAG_COUNT = 2

const COLAMD_STATUS = 3

const COLAMD_INFO1 = 4

const COLAMD_INFO2 = 5

const COLAMD_INFO3 = 6

const COLAMD_OK = 0

const COLAMD_OK_BUT_JUMBLED = 1

const COLAMD_ERROR_A_not_present = -1

const COLAMD_ERROR_p_not_present = -2

const COLAMD_ERROR_nrow_negative = -3

const COLAMD_ERROR_ncol_negative = -4

const COLAMD_ERROR_nnz_negative = -5

const COLAMD_ERROR_p0_nonzero = -6

const COLAMD_ERROR_A_too_small = -7

const COLAMD_ERROR_col_length_negative = -8

const COLAMD_ERROR_row_index_out_of_bounds = -9

const COLAMD_ERROR_out_of_memory = -10

const COLAMD_ERROR_internal_error = -999

const BTF_DATE = "Sept 18, 2023"

const BTF_MAIN_VERSION = 2

const BTF_SUB_VERSION = 2

const BTF_SUBSUB_VERSION = 1

BTF_VERSION_CODE(main, sub) = main * 1000 + sub

const BTF_VERSION = BTF_VERSION_CODE(BTF_MAIN_VERSION, BTF_SUB_VERSION)

const KLU_OK = 0

const KLU_SINGULAR = 1

const KLU_OUT_OF_MEMORY = -2

const KLU_INVALID = -3

const KLU_TOO_LARGE = -4

const KLU_DATE = "Sept 18, 2023"

const KLU_MAIN_VERSION = 2

const KLU_SUB_VERSION = 2

const KLU_SUBSUB_VERSION = 1

KLU_VERSION_CODE(main, sub) = main * 1000 + sub

const KLU_VERSION = KLU_VERSION_CODE(KLU_MAIN_VERSION, KLU_SUB_VERSION)
