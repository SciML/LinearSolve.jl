module LinearSolveAMGXExt

using LinearSolve: LinearSolve, AMGXPreconditioner
using LinearAlgebra: LinearAlgebra, ldiv!, mul!
using AMGX: AMGX
using CUDA: CUDA, CuVector
using CUDA.CUSPARSE: CuSparseMatrixCSR

# AMGX needs one process-wide `initialize`, and `finalize` must not run while any
# object is still alive, so it is left to process exit.
const AMGX_INITIALIZED = Ref(false)

function ensure_amgx_initialized!()
    if !AMGX_INITIALIZED[]
        AMGX.initialize()
        AMGX_INITIALIZED[] = true
    end
    return nothing
end

# A single AMG cycle, which is what makes sense when this preconditions a Krylov
# method rather than being the solver. Convergence is the outer method's job, so the
# cycle runs once and reports nothing.
# AMGX registers no `config_version` variable when a Dict is handed over as a
# parameter string, so it is left out and AMGX applies its current version.
default_amgx_config() = Dict(
    "solver" => "AMG",
    "algorithm" => "CLASSICAL",
    "max_iters" => "1",
    "cycle" => "V",
    "presweeps" => "1",
    "postsweeps" => "1",
    "monitor_residual" => "0",
    "determinism_flag" => "1",
)

function free_amgx_preconditioner!(P::AMGXPreconditioner)
    for f in (:solver, :xvec, :bvec, :matrix, :resources, :config)
        obj = getfield(P, f)
        obj === nothing && continue
        try
            AMGX.close(obj)
        catch
            # A finalizer must not throw, and there is nothing to do if AMGX has
            # already torn the object down.
        end
        setfield!(P, f, nothing)
    end
    return nothing
end

function build_amgx_preconditioner(A::CuSparseMatrixCSR, config)
    ensure_amgx_initialized!()
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("AMGX needs a square matrix"))

    cfg = AMGX.Config(config === nothing ? default_amgx_config() : config)
    resources = AMGX.Resources(cfg)
    mat = AMGX.AMGXMatrix(resources, AMGX.dDDI)
    xvec = AMGX.AMGXVector(resources, AMGX.dDDI)
    bvec = AMGX.AMGXVector(resources, AMGX.dDDI)
    solver = AMGX.Solver(resources, AMGX.dDDI, cfg)

    # CUDA.jl's CSR is 1-based and AMGX is a C library expecting 0-based indices.
    # `AMGX.upload!` for a `CuSparseMatrixCSR` does not rebase them, and AMGX only
    # notices the structure is wrong when it builds the hierarchy, reporting
    # "Internal error" from `setup!`. Rebasing here is what makes the setup succeed.
    AMGX.upload!(
        mat, A.rowPtr .- Int32(1), A.colVal .- Int32(1), A.nzVal
    )
    AMGX.setup!(solver, mat)

    P = LinearSolve._new_amgx_preconditioner(cfg, resources, mat, solver, xvec, bvec, n)
    finalizer(free_amgx_preconditioner!, P)
    return P
end

function build_amgx_preconditioner(A, config)
    return throw(
        ArgumentError(
            "AMGXPreconditioner needs a `CuSparseMatrixCSR`, got a $(typeof(A)). " *
                "Convert with `CuSparseMatrixCSR(A)`."
        )
    )
end

function LinearAlgebra.ldiv!(y::CuVector, P::AMGXPreconditioner, x::CuVector)
    length(x) == P.n ||
        throw(DimensionMismatch("preconditioner is $(P.n) wide, right-hand side is $(length(x))"))
    AMGX.upload!(P.bvec, x)
    AMGX.set_zero!(P.xvec, P.n)
    AMGX.solve!(P.xvec, P.solver, P.bvec)
    AMGX.download!(y, P.xvec)
    return y
end

LinearAlgebra.ldiv!(P::AMGXPreconditioner, x::CuVector) = ldiv!(x, P, copy(x))

Base.:\(P::AMGXPreconditioner, x::CuVector) = ldiv!(similar(x), P, x)

Base.size(P::AMGXPreconditioner) = (P.n, P.n)
Base.size(P::AMGXPreconditioner, i::Integer) = i <= 2 ? P.n : 1
Base.eltype(::AMGXPreconditioner) = Float64

end
