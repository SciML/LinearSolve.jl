# Solving Linear Systems with PETSc and MPI

This tutorial shows how to solve a linear system with `PETScAlgorithm` on an MPI
communicator using ordinary Julia sparse matrices and vectors.

Use this workflow when:

- your matrix is a plain Julia `SparseMatrixCSC`
- every rank starts from the same Julia matrix and right-hand side
- you want PETSc to solve the system in parallel and return the full solution on every rank

## Package setup

Load the PETSc MPI stack and initialize MPI:

```julia
using LinearSolve, PETSc, MPI, SparseArrays, SparseMatricesCSR, LinearAlgebra
```

This brings in the LinearSolve interface, PETSc and MPI bindings, Julia sparse
matrix support, and the CSR utilities required by the PETSc extension.

```julia
MPI.Init()
```

`SparseMatricesCSR` must be loaded because the PETSc extension uses it internally
for sparse matrix assembly.

## Building a sparse linear problem

Start from an ordinary Julia sparse matrix and right-hand side:

```julia
n = 12
```

Here `n` is the global problem size. PETSc will distribute the rows internally,
but you still define the global Julia problem in the usual way.

```julia
A = spdiagm(-1 => -ones(n - 1), 0 => 4.0 .* ones(n), 1 => -ones(n - 1))
b = ones(n)
```

Every MPI rank constructs the same Julia `A` and `b`. LinearSolve then builds a
distributed PETSc matrix underneath, with each rank contributing only its owned rows.

## Solving on an MPI communicator

Choose a PETSc Krylov solver and pass the MPI communicator through
`PETScAlgorithm`:

```julia
alg = PETScAlgorithm(:gmres; comm = MPI.COMM_WORLD)
```

The `comm = MPI.COMM_WORLD` keyword is what switches this into the multi-rank
PETSc path.

Now solve the linear problem:

```julia
sol = solve(LinearProblem(A, b), alg; abstol = 1.0e-10, reltol = 1.0e-10)
```

The returned `sol.u` is a full Julia vector on every rank, not just the owned slice.
You can verify the solve by checking the residual:

```julia
norm(A * sol.u - b) / norm(b)
```

If you want to see this on each MPI rank, add:

```julia
@show MPI.Comm_rank(MPI.COMM_WORLD)
@show norm(A * sol.u - b) / norm(b)
```

That makes it clear that every rank receives the same full Julia solution vector.

Save the script and run it with MPI:

```bash
mpiexecjl -n 2 julia --project petsc_mpi_example.jl
```

Here `-n 2` means "launch 2 MPI ranks". You can replace `2` with another
positive integer such as `1`, `4`, or `8` depending on how many ranks you want
to use for the run.

This command assumes you have installed the Julia MPI launcher wrapper
`mpiexecjl`. If not, use `$(MPI.mpiexec()) -n 2 julia --project
petsc_mpi_example.jl` or the MPI launcher installed on your machine, such as `mpiexec` or `mpirun`.

## Reusing the PETSc cache

If the matrix structure is unchanged, it is more efficient to reuse the PETSc
cache than to rebuild everything on every solve.

First, initialize the cache:

```julia
import SciMLBase
```

The caching interface lives in `SciMLBase`, so import it explicitly before using
`init`, `solve!`, and `reinit!`.

```julia
b0 = ones(n)
```

Use `b0` as the original right-hand side so later updates can be expressed
relative to a fixed baseline.

```julia
cache = SciMLBase.init(
    LinearProblem(A, b0),
    PETScAlgorithm(:gmres; comm = MPI.COMM_WORLD);
    abstol = 1.0e-10,
    reltol = 1.0e-10
)
```

Run the first solve:

```julia
sol = solve!(cache)
```

This allocates and stores the PETSc-side KSP, matrix, and vectors inside the cache.

```julia
norm(A * sol.u - b0) / norm(b0)
```

Now change only the right-hand side and reuse the existing PETSc objects:

```julia
b = 2.0 .* b0
SciMLBase.reinit!(cache; b = b)
sol = solve!(cache)
```

Because the matrix structure is unchanged, `reinit!` updates the cached problem
without rebuilding the PETSc solver from scratch.

```julia
norm(A * sol.u - b) / norm(b)
```

The same pattern works for additional right-hand sides:

```julia
for rhs_scale in (3.0, 4.0)
    b = rhs_scale .* b0
    SciMLBase.reinit!(cache; b = b)
    sol = solve!(cache)
    @show rhs_scale, norm(A * sol.u - b) / norm(b)
end
```

## Cleaning up PETSc resources

PETSc objects are C-managed resources, so clean them up explicitly when you are
done:

```julia
PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)
PETScExt.cleanup_petsc_cache!(cache)
```

This releases the PETSc objects immediately instead of waiting for Julia GC and
the fallback finalizer.

## Choosing the right MPI workflow

This tutorial covers the ordinary Julia `SparseMatrixCSC` MPI workflow. If your
data is already partitioned across ranks, use the existing `PSparseMatrix` /
`PVector` workflow from `PartitionedArrays.jl` instead.

For a short standalone script, it is usually fine to let process exit clean up
MPI state. If you prefer to be explicit, you can call `MPI.Finalize()` at the
end of the script.
