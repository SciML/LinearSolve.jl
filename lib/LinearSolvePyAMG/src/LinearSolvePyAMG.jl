"""
    LinearSolvePyAMG

A wrapper around [PyAMG](https://pyamg.readthedocs.io) (Algebraic Multigrid Solvers in
Python) for use with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl).
The Python library is accessed via [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl)
and installed automatically via [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl).

## Usage

```julia
using LinearSolvePyAMG, LinearSolve, SparseArrays

A = spdiagm(-1 => -ones(99), 0 => 2ones(100), 1 => -ones(99))
b = rand(100)
prob = LinearProblem(A, b)

# Ruge–Stüben AMG (default)
sol = solve(prob, PyAMGJL())

# Smoothed-aggregation AMG
sol = solve(prob, PyAMGJL_SmoothedAggregation())

# With GMRES acceleration
sol = solve(prob, PyAMGJL(accel = "gmres"))
```
"""
module LinearSolvePyAMG

using LinearSolve
using LinearAlgebra
using SparseArrays
using PythonCall
using CondaPkg
using SciMLBase: SciMLBase, ReturnCode

import LinearSolve: LinearCache, LinearVerbosity, OperatorAssumptions

# ---------------------------------------------------------------------------
# Algorithm types
# ---------------------------------------------------------------------------

"""
    PyAMGJL(; method = :RugeStuben, accel = nothing, kwargs...)

Algebraic Multigrid solver backed by [PyAMG](https://pyamg.readthedocs.io)
(Python) via PythonCall.jl.

PyAMG is automatically installed into the Julia-managed Python environment via
CondaPkg.jl; no manual Python setup is required.

## Keyword Arguments

  - `method`: AMG hierarchy construction method. Choices:
      + `:RugeStuben` (default): classical Ruge–Stüben AMG. Robust for many
        problem types, especially those arising from FEM/FVM discretisations
        of scalar elliptic PDEs.
      + `:SmoothedAggregation`: smoothed-aggregation AMG. Often more efficient
        for vector problems or when the near-null space is known.
  - `accel`: optional Krylov acceleration applied on top of the AMG cycle.
    Common choices: `"cg"`, `"gmres"`, `"bicgstab"`. Default `nothing` means
    plain V-cycle stationary iteration.
  - Additional keyword arguments are forwarded verbatim to the corresponding
    PyAMG solver constructor (e.g. `max_levels`, `max_coarse`, `strength`, …).

## Performance Notes

The AMG hierarchy is built during `init` (or `reinit!` when `A` changes) and
reused across multiple `solve!` calls with the same coefficient matrix. This
makes re-solving with a different right-hand side very cheap.

## Examples

```julia
using LinearSolvePyAMG, LinearSolve, SparseArrays

n = 200
A = spdiagm(-1 => -ones(n-1), 0 => 2ones(n), 1 => -ones(n-1))
b = rand(n)

prob = LinearProblem(A, b)

# Plain AMG V-cycle (Ruge–Stüben)
sol = solve(prob, PyAMGJL())

# AMG preconditioned CG
sol = solve(prob, PyAMGJL(accel = "cg"))

# Smoothed-aggregation with GMRES acceleration
sol = solve(prob, PyAMGJL_SmoothedAggregation(accel = "gmres"))
```
"""
struct PyAMGJL{K} <: LinearSolve.SciMLLinearSolveAlgorithm
    method::Symbol
    accel::Union{String, Nothing}
    kwargs::K
end

function PyAMGJL(; method::Symbol = :RugeStuben, accel = nothing, kwargs...)
    if method ∉ (:RugeStuben, :SmoothedAggregation)
        throw(
            ArgumentError(
                "PyAMGJL: unsupported `method` = $method. " *
                    "Choose :RugeStuben or :SmoothedAggregation."
            )
        )
    end
    return PyAMGJL(method, accel isa String ? accel : nothing, kwargs)
end

"""
    PyAMGJL_RugeStuben(; kwargs...)

Ruge–Stüben AMG solver via PyAMG. Equivalent to
`PyAMGJL(method = :RugeStuben; kwargs...)`.
"""
PyAMGJL_RugeStuben(; kwargs...) = PyAMGJL(; method = :RugeStuben, kwargs...)

"""
    PyAMGJL_SmoothedAggregation(; kwargs...)

Smoothed-aggregation AMG solver via PyAMG. Equivalent to
`PyAMGJL(method = :SmoothedAggregation; kwargs...)`.
"""
PyAMGJL_SmoothedAggregation(; kwargs...) = PyAMGJL(; method = :SmoothedAggregation, kwargs...)

LinearSolve.needs_concrete_A(::PyAMGJL) = true
LinearSolve.default_alias_A(::PyAMGJL, ::Any, ::Any) = true
LinearSolve.default_alias_b(::PyAMGJL, ::Any, ::Any) = true

# ---------------------------------------------------------------------------
# Lazy Python imports (imported once, cached in module-level Refs)
# ---------------------------------------------------------------------------

const _np = Ref{Py}()
const _scipy_sparse = Ref{Py}()
const _pyamg = Ref{Py}()

function _get_np()
    isassigned(_np) || (_np[] = pyimport("numpy"))
    return _np[]
end

function _get_scipy_sparse()
    isassigned(_scipy_sparse) || (_scipy_sparse[] = pyimport("scipy.sparse"))
    return _scipy_sparse[]
end

function _get_pyamg()
    isassigned(_pyamg) || (_pyamg[] = pyimport("pyamg"))
    return _pyamg[]
end

# ---------------------------------------------------------------------------
# Julia sparse matrix → scipy CSR conversion
# ---------------------------------------------------------------------------

"""
    _to_scipy_csr(A) -> Py

Convert a Julia sparse (or dense) matrix `A` to a **scipy CSR matrix** (Python).

The matrix is cast to `Float64` first; PyAMG requires real-valued input.
"""
function _to_scipy_csr(A::AbstractMatrix)
    Acsc = A isa SparseMatrixCSC ? A : sparse(A)
    Af = SparseMatrixCSC{Float64, Cint}(Acsc)   # ensure Float64 & Int32 indices
    m, n = size(Af)

    np = _get_np()
    ss = _get_scipy_sparse()

    # Julia SparseMatrixCSC stores data in CSC layout:
    #   colptr  → 1-based column pointers   (length n+1)
    #   rowval  → 1-based row indices       (length nnz)
    #   nzval   → non-zero values           (length nnz)
    # scipy.csc_matrix expects 0-based indices and (data, indices, indptr).
    data = np.asarray(Af.nzval)
    indices = np.asarray(Af.rowval .- Cint(1))   # 0-based row indices
    indptr = np.asarray(Af.colptr .- Cint(1))   # 0-based col pointers

    py_csc = ss.csc_matrix((data, indices, indptr), shape = (m, n))
    return py_csc.tocsr()
end

# ---------------------------------------------------------------------------
# Build the AMG hierarchy
# ---------------------------------------------------------------------------

function _build_hierarchy(A::AbstractMatrix, method::Symbol, extra_kwargs)
    pyamg = _get_pyamg()
    pyA = _to_scipy_csr(A)

    if method === :RugeStuben
        return pyamg.ruge_stuben_solver(pyA; extra_kwargs...)
    elseif method === :SmoothedAggregation
        return pyamg.smoothed_aggregation_solver(pyA; extra_kwargs...)
    else
        error("PyAMGJL: unreachable method=$method")
    end
end

# ---------------------------------------------------------------------------
# LinearSolve interface
# ---------------------------------------------------------------------------

function LinearSolve.init_cacheval(
        alg::PyAMGJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    ml = _build_hierarchy(A, alg.method, alg.kwargs)
    return (; ml)
end

function SciMLBase.solve!(cache::LinearCache, alg::PyAMGJL; kwargs...)
    # Rebuild the AMG hierarchy if A has changed since the last solve
    if cache.isfresh
        cache.cacheval = LinearSolve.init_cacheval(
            alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol, cache.verbose,
            cache.assumptions
        )
        cache.isfresh = false
    end

    ml = cache.cacheval.ml
    np = _get_np()

    b_jl = Vector{Float64}(cache.b)
    x0_jl = Vector{Float64}(cache.u)

    b_py = np.asarray(b_jl)
    x0_py = np.asarray(x0_jl)

    solve_kw = (
        tol = Float64(cache.reltol),
        maxiter = cache.maxiters,
        x0 = x0_py,
    )
    if alg.accel !== nothing
        solve_kw = merge(solve_kw, (accel = alg.accel,))
    end

    x_py = ml.solve(b_py; solve_kw...)
    x_jl = pyconvert(Vector{Float64}, x_py)
    copyto!(cache.u, x_jl)

    resid = norm(cache.A * cache.u .- cache.b)
    return SciMLBase.build_linear_solution(
        alg, cache.u, resid, cache;
        retcode = ReturnCode.Success
    )
end

LinearSolve.update_tolerances_internal!(cache, ::PyAMGJL, atol, rtol) = nothing

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export PyAMGJL, PyAMGJL_RugeStuben, PyAMGJL_SmoothedAggregation

end # module LinearSolvePyAMG
