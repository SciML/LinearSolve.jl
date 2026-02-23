# LinearSolvePyAMG.jl

A [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) sub-library that
wraps the Python [PyAMG](https://pyamg.readthedocs.io) Algebraic Multigrid (AMG)
library via [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl).
PyAMG is installed automatically into Julia's managed Python environment via
[CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/SciML/LinearSolve.jl", subdir = "lib/LinearSolvePyAMG")
```

## Usage

```julia
using LinearSolvePyAMG, LinearSolve, SparseArrays

# Build a 1-D Poisson-like SPD system
n = 200
A = spdiagm(-1 => -ones(n-1), 0 => 2ones(n), 1 => -ones(n-1))
b = rand(n)
prob = LinearProblem(A, b)

# Ruge–Stüben AMG (plain V-cycle)
sol = solve(prob, PyAMGJL())

# Smoothed-aggregation AMG with CG acceleration
sol = solve(prob, PyAMGJL_SmoothedAggregation(accel = "cg"))

# Ruge–Stüben AMG with GMRES acceleration
sol = solve(prob, PyAMGJL(accel = "gmres"))
```

## Solvers

| Constructor | Description |
|---|---|
| `PyAMGJL()` | Ruge–Stüben AMG (default); plain V-cycle |
| `PyAMGJL(method = :SmoothedAggregation)` | Smoothed-aggregation AMG |
| `PyAMGJL(accel = "cg")` | AMG + CG acceleration |
| `PyAMGJL(accel = "gmres")` | AMG + GMRES acceleration |
| `PyAMGJL(accel = "bicgstab")` | AMG + BiCGSTAB acceleration |
| `PyAMGJL_RugeStuben(; kwargs...)` | Shortcut for Ruge–Stüben |
| `PyAMGJL_SmoothedAggregation(; kwargs...)` | Shortcut for smoothed aggregation |

Additional keyword arguments are forwarded to the PyAMG solver constructor
(e.g. `max_levels`, `max_coarse`, `strength`).

## How it works

1. When `init` is called, the AMG hierarchy is built from `A` using PyAMG.
2. On each `solve!` call the hierarchy is reused (only the RHS `b` changes).
3. If `reinit!` is called with a new matrix `A`, the hierarchy is rebuilt.

This makes `LinearSolvePyAMG` efficient for problems where the same linear
system structure is solved repeatedly with different right-hand sides.
