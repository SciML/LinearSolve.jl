# Solving Linear Systems with HYPRE and MPI

This tutorial shows how to solve a linear system with `HYPREAlgorithm` on an MPI
communicator using ordinary Julia sparse matrices and vectors.

Use this workflow when:

- your matrix is a plain Julia sparse matrix such as `SparseMatrixCSC`
- you want LinearSolve to construct distributed `HYPREMatrix` / `HYPREVector` objects for you
- you want the solve itself to run across multiple MPI ranks

Unlike a standard serial Julia workflow, this path uses a non-standard execution model:
the script is launched under `mpiexecjl`, `mpiexec`, or `mpirun`, and each rank participates
in the solve.

## Package setup

Load the HYPRE and MPI stack first:

```julia
using LinearSolve, HYPRE, MPI, SparseArrays, LinearAlgebra
```

This brings in the LinearSolve interface, the Julia HYPRE wrapper, MPI bindings, and sparse
matrix support.

Initialize MPI and HYPRE before constructing the solver:

```julia
MPI.Init()
```

```julia
HYPRE.Init()
```

## Building a sparse linear problem

Start from an ordinary Julia sparse matrix and right-hand side:

```julia
n = 20
```

Here `n` is the global problem size. Every MPI rank starts from the same global Julia matrix
and vector, and LinearSolve splits them into contiguous row blocks when it constructs the
distributed HYPRE objects.

```julia
A = spdiagm(0 => collect(1.0:n))
b = 2.0 .* collect(1.0:n)
```

This example uses a diagonal system so it is easy to inspect the local solution on each rank.

## Solving on an MPI communicator

Pass the communicator through `HYPREAlgorithm`:

```julia
alg = HYPREAlgorithm(HYPRE.PCG; comm = MPI.COMM_WORLD)
```

The `comm = MPI.COMM_WORLD` keyword is what switches this into the communicator-based
distributed HYPRE path.

Now solve the problem:

```julia
sol = solve(LinearProblem(A, b), alg; abstol = 1.0e-10, reltol = 1.0e-10)
```

For this workflow, `sol.u` is a distributed `HYPREVector`, not a replicated Julia vector.
Each rank owns only its local row interval. You can copy out that local piece with:

```julia
local_x = Vector{Float64}(undef, sol.u.iupper - sol.u.ilower + 1)
copy!(local_x, sol.u)
```

To see what each rank owns, print the local bounds and values:

```julia
@show MPI.Comm_rank(MPI.COMM_WORLD), sol.u.ilower, sol.u.iupper, local_x
```

For this diagonal example, every entry of `local_x` should be close to `2.0`.

Save the script and run it with MPI:

```bash
mpiexecjl -n 2 julia --project hypre_mpi_example.jl
```

Here `-n 2` means "launch 2 MPI ranks". You can replace `2` with another positive integer
such as `1`, `4`, or `8` depending on how many ranks you want to use for the run.

This command assumes you have installed the Julia MPI launcher wrapper `mpiexecjl`. If not,
use `$(MPI.mpiexec()) -n 2 julia --project hypre_mpi_example.jl` or an MPI launcher on your
machine such as `mpiexec` or `mpirun`, as long as it comes from the same MPI installation as
the Julia MPI stack.

## Reusing the HYPRE cache

If the matrix structure is unchanged, it is more efficient to reuse the cached HYPRE objects
than to rebuild them on every solve.

First, initialize the cache:

```julia
import SciMLBase
```

The caching interface lives in `SciMLBase`, so import it explicitly before using `init` and
`solve!`.

```julia
prob = LinearProblem(A, b)
```

```julia
cache = SciMLBase.init(
    prob,
    HYPREAlgorithm(HYPRE.PCG; comm = MPI.COMM_WORLD);
    abstol = 1.0e-10,
    reltol = 1.0e-10
)
```

Run the first solve:

```julia
sol = solve!(cache)
```

This allocates and stores the distributed HYPRE matrix, vectors, and solver inside the cache.

Now change only the right-hand side:

```julia
b2 = 3.0 .* collect(1.0:n)
cache.b = b2
sol = solve!(cache)
```

Because the matrix structure is unchanged, updating `cache.b` refreshes the distributed
right-hand side without rebuilding the whole solver stack.

You can inspect the updated local solution again with:

```julia
copy!(local_x, sol.u)
@show MPI.Comm_rank(MPI.COMM_WORLD), local_x
```

For this second solve, every entry of `local_x` should be close to `3.0`.

## Choosing the right HYPRE workflow

This tutorial covers the plain Julia matrix/vector MPI workflow. If you already have
distributed HYPRE objects, you can continue to pass `HYPREMatrix` and `HYPREVector`
directly instead of using the auto-construction path.

For a short standalone script, it is usually fine to let process exit clean up MPI and HYPRE
state. If you prefer to be explicit, you can call `MPI.Finalize()` at the end of the script.
