# Solving Linear Systems with PartitionedSolvers.jl

This tutorial shows how to solve a distributed linear system with
`PartitionedSolversAlgorithm` using `PSparseMatrix` / `PVector` inputs from
`PartitionedArrays.jl`.

Use this workflow when:

- your matrix and vectors are already partitioned across ranks
- you want the solve itself to stay inside the `PartitionedArrays` /
  `PartitionedSolvers` ecosystem
- you want repeated `solve!` reuse on a distributed problem

Unlike a standard serial Julia workflow, this path uses a non-standard execution
model: the script is launched under `mpiexecjl`, `mpiexec`, or `mpirun`, and
every MPI rank participates in the solve.

## Package setup

Load the distributed array and solver stack first:

```julia
using LinearSolve, MPI, PartitionedArrays, PartitionedSolvers
using PartitionedArrays: PVector, own_to_local, partition, tuple_of_arrays,
    uniform_partition, with_mpi
import SciMLBase
```

Initialize MPI before constructing the partitioned problem:

```julia
MPI.Init()
```

## Building a distributed linear problem

Start by defining a helper that turns the MPI rank layout into a contiguous row
partition:

```julia
function mpi_row_partition(distribute, n)
    parts = distribute(LinearIndices((MPI.Comm_size(MPI.COMM_WORLD),)))
    return uniform_partition(parts, n)
end
```

This tutorial uses a diagonal system because it is easy to inspect locally on
each rank. The helper below builds:

- a distributed `PSparseMatrix`
- a distributed right-hand side `PVector`
- a distributed zero initial guess `PVector`

```julia
function build_splitmat_diag(row_partition, scale = 1.0)
    I_v, J_v, V_v = map(row_partition) do rng
        collect(Int, rng), collect(Int, rng), scale .* Float64.(rng)
    end |> tuple_of_arrays
    A = psparse(I_v, J_v, V_v, row_partition, row_partition) |> fetch
    b = PVector(map(rng -> scale .* Float64.(rng), row_partition), row_partition)
    u = PVector(map(rng -> zeros(length(rng)), row_partition), row_partition)
    return A, b, u
end
```

Now build and solve the 16-by-16 distributed problem inside `with_mpi()` so the
partitioned objects stay in scope for the whole workflow:

```julia
with_mpi() do distribute
    rp = mpi_row_partition(distribute, 16)
    A, b, u0 = build_splitmat_diag(rp)
    prob = LinearProblem(A, b; u0 = u0)

    # Explicit CG solve
    alg = PartitionedSolversAlgorithm(PartitionedSolvers.cg)
    sol = solve(prob, alg; abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 40)

    # Inspect the owned local entries on this rank
    map(partition(sol.u), partition(axes(sol.u, 1))) do local_u, row_idx
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        for j in own_to_local(row_idx)
            @show rank, row_idx[j], local_u[j]
        end
    end

    # Inspect the distributed residual
    r = A * sol.u - b
    map(partition(r), partition(axes(r, 1))) do local_r, row_idx
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        @show rank, maximum(abs, local_r)
    end

    # Cache reuse on an updated right-hand side
    cache = SciMLBase.init(
        prob,
        PartitionedSolversAlgorithm(PartitionedSolvers.cg);
        abstol = 1.0e-12,
        reltol = 1.0e-12,
        maxiters = 40
    )
    sol = solve!(cache)

    b2 = PVector(map(rng -> 2.0 .* Float64.(rng), rp), rp)
    cache.b = b2
    sol = solve!(cache)

    # Typed default dispatch
    default_alg = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
    sol = solve(prob, default_alg; abstol = 1.0e-12, reltol = 1.0e-12)

    # Non-CG distributed solver
    jacobi_alg = PartitionedSolversAlgorithm(PartitionedSolvers.jacobi)
    sol = solve(prob, jacobi_alg; maxiters = 20)
end
```

Each rank owns only its local chunk of `A`, `b`, and `u0`.

## Solving with an explicit PartitionedSolvers solver

The explicit CG solve inside that same `with_mpi()` block is:

```julia
alg = PartitionedSolversAlgorithm(PartitionedSolvers.cg)
sol = solve(prob, alg; abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 40)
```

The returned `sol.u` is a distributed `PVector`, not a replicated Julia vector.
For this diagonal example, every owned value should be close to `1.0`.

You can inspect what each rank owns with:

```julia
map(partition(sol.u), partition(axes(sol.u, 1))) do local_u, row_idx
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    for j in own_to_local(row_idx)
        @show rank, row_idx[j], local_u[j]
    end
end
```

You can also check the distributed residual:

```julia
r = A * sol.u - b
map(partition(r), partition(axes(r, 1))) do local_r, row_idx
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    @show rank, maximum(abs, local_r)
end
```

Each rank should report a very small local residual.

Save the script and run it with MPI:

```bash
mpiexecjl -n 2 julia --project partitionedsolvers_example.jl
```

Here `-n 2` means "launch 2 MPI ranks". You can replace `2` with another
positive integer such as `1`, `4`, or `8` depending on how many ranks you want
to use for the run.

This command assumes you have installed the Julia MPI launcher wrapper
`mpiexecjl`. If not, use `$(MPI.mpiexec()) -n 2 julia --project
partitionedsolvers_example.jl` or an MPI launcher on your machine such as
`mpiexec` or `mpirun`, as long as it comes from the same MPI installation as the
Julia MPI stack.

## Reusing the distributed cache

If the partitioning and matrix structure are unchanged, it is more efficient to
reuse the cached distributed solver object than to rebuild it on every solve.

Inside that same `with_mpi()` block, initialize the cache:

```julia
cache = SciMLBase.init(
    prob,
    PartitionedSolversAlgorithm(PartitionedSolvers.cg);
    abstol = 1.0e-12,
    reltol = 1.0e-12,
    maxiters = 40
)
```

Run the first solve:

```julia
sol = solve!(cache)
```

Now change only the right-hand side and reuse the existing distributed solver
state:

```julia
b2 = PVector(map(rng -> 2.0 .* Float64.(rng), rp), rp)
cache.b = b2
sol = solve!(cache)
```

For this second solve, every owned entry of `sol.u` should be close to `2.0`.

## Using the typed default dispatch

For square `PSparseMatrix` / `PVector` problems, `LinearSolve.defaultalg`
chooses the CG-backed `PartitionedSolvers` path automatically inside the same
distributed workflow:

```julia
alg = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
sol = solve(prob, alg; abstol = 1.0e-12, reltol = 1.0e-12)
```

That gives you a distributed solve without manually naming the backend solver.

## Using a non-CG distributed solver

The integration is solver-agnostic. For example, you can switch to the
distributed Jacobi iteration inside that same `with_mpi()` block:

```julia
alg = PartitionedSolversAlgorithm(PartitionedSolvers.jacobi)
sol = solve(prob, alg; maxiters = 20)
```

LinearSolve forwards only the convergence keywords that the selected
`PartitionedSolvers` solver constructor actually accepts.

## Choosing the right distributed workflow

This tutorial covers the `PartitionedArrays.jl` /
`PartitionedSolvers.jl` workflow where the matrix and vectors are already
distributed as `PSparseMatrix` / `PVector` values.

If every rank instead starts from the same ordinary Julia sparse matrix and you
want LinearSolve to distribute it for you, use the PETSc or HYPRE MPI workflows
instead.

For a short standalone script, it is usually fine to let process exit clean up
MPI state. If you prefer to be explicit, you can call `MPI.Finalize()` at the
end of the script.
