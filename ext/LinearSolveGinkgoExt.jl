module LinearSolveGinkgoExt

using LinearSolve, LinearAlgebra, SparseArrays
using LinearSolve: LinearCache, LinearVerbosity, OperatorAssumptions
import SciMLBase
import Ginkgo

# -----------------------------------------------------------------------
# Constructors (defined here so they can use Ginkgo.jl types directly)
# -----------------------------------------------------------------------

function LinearSolve.GinkgoJL(
        args...;
        KrylovAlg = :cg,
        executor = :omp,
        kwargs...
    )
    return GinkgoJL(KrylovAlg, executor, args, kwargs)
end

function LinearSolve.GinkgoJL_CG(args...; executor = :omp, kwargs...)
    return GinkgoJL(args...; KrylovAlg = :cg, executor = executor, kwargs...)
end

# -----------------------------------------------------------------------
# Trait overrides
# -----------------------------------------------------------------------

LinearSolve.default_alias_A(::GinkgoJL, ::Any, ::Any) = true
LinearSolve.default_alias_b(::GinkgoJL, ::Any, ::Any) = true
LinearSolve.needs_concrete_A(::GinkgoJL) = true

# -----------------------------------------------------------------------
# MTX format helpers
# -----------------------------------------------------------------------

"""
Write a sparse Julia matrix to a MatrixMarket coordinate file.
Ginkgo reads CSR matrices from MTX coordinate format.
"""
function _write_sparse_mtx(A::SparseMatrixCSC, filename::AbstractString)
    m, n = size(A)
    nnzA = nnz(A)
    rows = rowvals(A)
    vals = nonzeros(A)
    return open(filename, "w") do f
        println(f, "%%MatrixMarket matrix coordinate real general")
        println(f, "$m $n $nnzA")
        for j in 1:n
            for k in nzrange(A, j)
                i = rows[k]
                v = Float32(vals[k])
                println(f, "$i $j $v")
            end
        end
    end
end

"""
Write a Julia vector (or column of a matrix) to a MatrixMarket array file.
Ginkgo reads Dense matrices (vectors) from MTX array format.
"""
function _write_dense_vec_mtx(v::AbstractVector, filename::AbstractString)
    n = length(v)
    return open(filename, "w") do f
        println(f, "%%MatrixMarket matrix array real general")
        println(f, "$n 1")
        for vi in v
            println(f, Float32(vi))
        end
    end
end

# -----------------------------------------------------------------------
# Convert a Julia AbstractMatrix to SparseMatrixCsr{Float32, Int32}
# -----------------------------------------------------------------------

function _to_gko_csr(A, exec)
    A_sparse = A isa SparseMatrixCSC ? A : sparse(A)
    tmpfile = tempname() * ".mtx"
    try
        _write_sparse_mtx(SparseMatrixCSC{Float32, Int32}(A_sparse), tmpfile)
        return Ginkgo.SparseMatrixCsr{Float32, Int32}(tmpfile, exec)
    finally
        rm(tmpfile, force = true)
    end
end

# -----------------------------------------------------------------------
# Convert a Julia AbstractVector to Dense{Float32}
# -----------------------------------------------------------------------

function _to_gko_dense(v::AbstractVector, exec)
    tmpfile = tempname() * ".mtx"
    try
        _write_dense_vec_mtx(v, tmpfile)
        return Ginkgo.Dense{Float32}(tmpfile, exec)
    finally
        rm(tmpfile, force = true)
    end
end

# -----------------------------------------------------------------------
# Copy a Dense{Float32} result back into a Julia vector
# -----------------------------------------------------------------------

function _copy_gko_dense_to!(dest::AbstractVector, src::Ginkgo.Dense{Float32})
    n = length(dest)
    for i in 1:n
        dest[i] = src[i, 1]
    end
    return dest
end

# -----------------------------------------------------------------------
# Cache initialisation
# -----------------------------------------------------------------------

function LinearSolve.init_cacheval(
        alg::GinkgoJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    exec = Ginkgo.create(alg.executor)
    return (; exec)
end

# -----------------------------------------------------------------------
# solve!
# -----------------------------------------------------------------------

function SciMLBase.solve!(cache::LinearCache, alg::GinkgoJL; kwargs...)
    exec = cache.cacheval.exec

    # ---- Convert A -------------------------------------------------
    gko_A = _to_gko_csr(cache.A, exec)

    # ---- Convert b and initial guess x ----------------------------
    gko_b = _to_gko_dense(cache.b, exec)
    gko_x = _to_gko_dense(cache.u, exec)

    # ---- Dispatch to solver ----------------------------------------
    if alg.KrylovAlg === :cg
        Ginkgo.cg!(
            exec, gko_x, gko_A, gko_b;
            maxiter = cache.maxiters,
            reduction = Float64(cache.reltol),
            alg.kwargs...
        )
    else
        error("GinkgoJL: unsupported KrylovAlg = $(alg.KrylovAlg). Currently only :cg is supported.")
    end

    # ---- Copy result back -----------------------------------------
    _copy_gko_dense_to!(cache.u, gko_x)

    # ---- Compute residual (using Julia-side data) ------------------
    resid = norm(cache.A * cache.u - cache.b)

    return SciMLBase.build_linear_solution(alg, cache.u, resid, cache)
end

LinearSolve.update_tolerances_internal!(cache, alg::GinkgoJL, atol, rtol) = nothing

end # module LinearSolveGinkgoExt
