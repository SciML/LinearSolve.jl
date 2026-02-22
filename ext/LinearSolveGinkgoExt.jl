module LinearSolveGinkgoExt

using LinearSolve, LinearAlgebra, SparseArrays
using LinearSolve: LinearCache, LinearVerbosity, OperatorAssumptions
import SciMLBase
import Ginkgo

function LinearSolve.GinkgoJL(
        args...;
        KrylovAlg = :gmres,
        executor = :omp,
        kwargs...
    )
    return GinkgoJL(KrylovAlg, executor, args, kwargs)
end

function LinearSolve.GinkgoJL_CG(args...; executor = :omp, kwargs...)
    return GinkgoJL(args...; KrylovAlg = :cg, executor = executor, kwargs...)
end

function LinearSolve.GinkgoJL_GMRES(args...; executor = :omp, kwargs...)
    return GinkgoJL(args...; KrylovAlg = :gmres, executor = executor, kwargs...)
end

LinearSolve.default_alias_A(::GinkgoJL, ::Any, ::Any) = true
LinearSolve.default_alias_b(::GinkgoJL, ::Any, ::Any) = true
LinearSolve.needs_concrete_A(::GinkgoJL) = true

# Julia CSC (1-indexed, col-major) → Ginkgo CSR (0-indexed, row-major).
# The get_const_* accessors return raw writable pointers; unsafe_copyto! fills
# Ginkgo's internal buffers directly — same pattern as Ginkgo.nonzeros() etc.

"""
    _csc_to_csr_arrays(A)

Convert a `SparseMatrixCSC{Float32,Int32}` to the three CSR arrays required
by Ginkgo (all 0-indexed): `rowptr` (length m+1), `col_idxs` (length nnz),
`vals` (length nnz).
"""
function _csc_to_csr_arrays(A::SparseMatrixCSC{Float32, Int32})
    m, n = size(A)
    nz = nnz(A)
    rvals = rowvals(A)
    nzv = nonzeros(A)

    row_counts = zeros(Int32, m)
    for r in rvals
        row_counts[r] += Int32(1)
    end

    rowptr = Vector{Int32}(undef, m + 1)
    rowptr[1] = Int32(0)
    for i in 1:m
        rowptr[i + 1] = rowptr[i] + row_counts[i]
    end

    col_idxs = Vector{Int32}(undef, nz)
    vals = Vector{Float32}(undef, nz)
    fill!(row_counts, Int32(0))
    for j in 1:n
        for k in nzrange(A, j)
            r = rvals[k]
            pos = rowptr[r] + row_counts[r] + 1
            col_idxs[pos] = Int32(j - 1)  # 0-indexed
            vals[pos] = nzv[k]
            row_counts[r] += Int32(1)
        end
    end

    return rowptr, col_idxs, vals
end

"""
    _to_gko_csr_inmem(A, exec) -> gko_matrix_csr_f32_i32

Build a Ginkgo CSR matrix from `A` entirely in memory (no files, no MTX
serialisation).  Returns a raw `gko_matrix_csr_f32_i32` pointer; the caller
**must** call `Ginkgo.API.ginkgo_matrix_csr_f32_i32_delete` when done.
"""
function _to_gko_csr_inmem(A, exec::Ptr)
    A_f32 = SparseMatrixCSC{Float32, Int32}(A isa SparseMatrixCSC ? A : sparse(A))
    m, n = size(A_f32)
    nz = nnz(A_f32)

    rowptr, col_idxs, vals = _csc_to_csr_arrays(A_f32)

    dim = Ginkgo.API.ginkgo_dim2_create(m, n)
    mat_ptr = Ginkgo.API.ginkgo_matrix_csr_f32_i32_create(exec, dim, nz)

    val_raw = Ginkgo.API.ginkgo_matrix_csr_f32_i32_get_const_values(mat_ptr)
    col_raw = Ginkgo.API.ginkgo_matrix_csr_f32_i32_get_const_col_idxs(mat_ptr)
    row_raw = Ginkgo.API.ginkgo_matrix_csr_f32_i32_get_const_row_ptrs(mat_ptr)

    GC.@preserve vals col_idxs rowptr begin
        unsafe_copyto!(val_raw, pointer(vals), nz)
        unsafe_copyto!(col_raw, pointer(col_idxs), nz)
        unsafe_copyto!(row_raw, pointer(rowptr), m + 1)
    end

    return mat_ptr
end

# ginkgo_matrix_dense_f32_read opens a file (wraps std::ifstream) — no in-memory
# path exists: the Dense API exposes no get_values pointer and setindex! is
# unimplemented (TODO in Dense.jl).  MTX content is built in an IOBuffer and
# flushed in a single write() to minimise disk overhead.

"""
    _to_gko_dense(v, exec) -> gko_matrix_dense_f32

Write `v` to a temp file in MTX array format and read it into a Ginkgo Dense
n×1 matrix.  Returns a raw `gko_matrix_dense_f32` pointer; the caller
**must** call `Ginkgo.API.ginkgo_matrix_dense_f32_delete` when done.
"""
function _to_gko_dense(v::AbstractVector, exec::Ptr)
    n = length(v)
    buf = IOBuffer()
    println(buf, "%%MatrixMarket matrix array real general")
    println(buf, "$n 1")
    for vi in v
        println(buf, Float32(vi))
    end
    tmpfile = tempname() * ".mtx"
    try
        write(tmpfile, take!(buf))
        return Ginkgo.API.ginkgo_matrix_dense_f32_read(tmpfile, exec)
    finally
        rm(tmpfile, force = true)
    end
end

# Ginkgo vectors are n×1 Dense matrices; ginkgo_matrix_dense_f32_at uses 0-based indices.
function _copy_gko_dense_ptr_to!(dest::AbstractVector, x_ptr, n::Int)
    for i in 0:(n - 1)
        dest[i + 1] = Ginkgo.API.ginkgo_matrix_dense_f32_at(
            x_ptr, Csize_t(i), Csize_t(0)
        )
    end
    return dest
end

function LinearSolve.init_cacheval(
        alg::GinkgoJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    exec = Ginkgo.create(alg.executor)
    return (; exec)
end

function SciMLBase.solve!(cache::LinearCache, alg::GinkgoJL; kwargs...)
    exec = cache.cacheval.exec
    n = length(cache.b)

    A_ptr = _to_gko_csr_inmem(cache.A, exec)
    b_ptr = _to_gko_dense(cache.b, exec)
    x_ptr = _to_gko_dense(cache.u, exec)

    try
        if alg.KrylovAlg === :cg
            Ginkgo.API.ginkgo_solver_cg_solve(
                exec, A_ptr, b_ptr, x_ptr,
                Cint(cache.maxiters), Cdouble(cache.reltol)
            )
        elseif alg.KrylovAlg === :gmres
            # Ginkgo.jl v1 does not yet expose a GMRES solver.
            # Track https://github.com/youwuyou/Ginkgo.jl for updates.
            error(
                "GinkgoJL: Ginkgo.jl v1 does not yet expose a GMRES solver. ",
                "Use GinkgoJL_CG() for symmetric positive definite systems."
            )
        else
            error(
                "GinkgoJL: unsupported KrylovAlg = $(alg.KrylovAlg). ",
                "Supported value: :cg"
            )
        end

        _copy_gko_dense_ptr_to!(cache.u, x_ptr, n)
    finally
        Ginkgo.API.ginkgo_matrix_csr_f32_i32_delete(A_ptr)
        Ginkgo.API.ginkgo_matrix_dense_f32_delete(b_ptr)
        Ginkgo.API.ginkgo_matrix_dense_f32_delete(x_ptr)
    end

    resid = norm(cache.A * cache.u - cache.b)
    return SciMLBase.build_linear_solution(alg, cache.u, resid, cache)
end

LinearSolve.update_tolerances_internal!(cache, alg::GinkgoJL, atol, rtol) = nothing

end # module LinearSolveGinkgoExt
