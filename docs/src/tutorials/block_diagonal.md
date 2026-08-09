# Solving Block Diagonal Systems

A block diagonal matrix is one whose nonzeros sit entirely in square blocks along
the diagonal, so the system it defines is really a batch of smaller, independent
systems stacked together:

```math
\begin{bmatrix} A_1 & & \\ & A_2 & \\ & & A_3 \end{bmatrix}
\begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix} =
\begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}
```

These show up whenever a problem decouples into independent subsystems: per-element
systems in a finite element assembly, per-sample systems in a batched inverse
problem, or the diagonal approximation of a larger operator.

[BlockDiagonals.jl](https://github.com/JuliaArrays/BlockDiagonals.jl) provides a
`BlockDiagonal` type that stores only the blocks, and LinearSolve.jl accepts it
directly as the `A` of a `LinearProblem`.

## Setting Up the Problem

```@example blockdiag
import LinearSolve as LS
import LinearAlgebra as LA
using BlockDiagonals

blocks = [rand(3, 3) + 3LA.I for _ in 1:4]
A = BlockDiagonal(blocks)
b = rand(size(A, 1))

prob = LS.LinearProblem(A, b)
sol = LS.solve(prob)
sol.u
```

The blocks do not have to be the same size:

```@example blockdiag
A_mixed = BlockDiagonal([rand(n, n) + n * LA.I for n in (2, 3, 4)])
b_mixed = rand(size(A_mixed, 1))

LS.solve(LS.LinearProblem(A_mixed, b_mixed)).u
```

## Which Solvers Work

Most of the standard algorithms accept a `BlockDiagonal` and treat it as a plain
matrix, which is always correct but does not exploit the structure:

```@example blockdiag
u_ref = Matrix(A) \ b

for alg in (LS.LUFactorization(), LS.GenericFactorization(),
        LS.QRFactorization(LA.NoPivot()), LS.SimpleGMRES(), LS.KrylovJL_GMRES())
    u = LS.solve(LS.LinearProblem(A, b), alg).u
    println(rpad(nameof(typeof(alg)), 22), " residual = ",
        LA.norm(Matrix(A) * u - b))
end
```

Two caveats are worth knowing:

  - `QRFactorization` must use `NoPivot()`. A QR of a block diagonal matrix is
    itself block diagonal, so the factorization is computed in place and keeps
    the structure. Column pivoting would have to move entries between blocks,
    which the `BlockDiagonal` storage type cannot represent, so
    `QRFactorization(ColumnNorm())` raises an error rather than densifying.
  - Sparse-only algorithms such as `UMFPACKFactorization` and `KLUFactorization`
    expect a `SparseMatrixCSC`. Convert with `sparse(A)` first if you want those.

## The `SimpleGMRES` Specialization

The one algorithm that actually uses the block structure is
[`SimpleGMRES`](@ref). When every block is the same size, it solves the batch of
subsystems together instead of running one Krylov iteration over the whole
stacked system. LinearSolve.jl detects this automatically when the extension is
loaded, so no keyword is needed:

```@example blockdiag
cache_uniform = LS.init(LS.LinearProblem(A, b), LS.SimpleGMRES())
LS.solve!(cache_uniform).u
```

You can see the specialization in the cache that `init` builds. The first type
parameter of the cache records whether the batched path was selected, and the
`blocksize` field records the detected block size:

```@example blockdiag
cache_mixed = LS.init(LS.LinearProblem(A_mixed, b_mixed), LS.SimpleGMRES())

(uniform_blocksize = cache_uniform.cacheval.blocksize,
    mixed_blocksize = cache_mixed.cacheval.blocksize)
```

The uniform case reports the block size it found, while the mixed case reports
`0`, meaning it fell back to the generic path. Both give the right answer; only
the uniform case gets the batched treatment.

If your matrix is stored as a plain dense `Matrix` but you know it is block
diagonal with uniform blocks, you can request the same specialization explicitly
with the `blocksize` keyword:

```@example blockdiag
LS.solve(LS.LinearProblem(Matrix(A), b), LS.SimpleGMRES(; blocksize = 3)).u
```

## Reusing the Factorization

The [caching interface](@ref Linear-Solve-with-Caching-Interface) works with
`BlockDiagonal` exactly as it does for any other matrix, which matters when the
same block structure is solved against many right hand sides:

```@example blockdiag
cache = LS.init(LS.LinearProblem(A, b), LS.QRFactorization(LA.NoPivot()))
sol1 = LS.solve!(cache)

cache.b = rand(size(A, 1))
sol2 = LS.solve!(cache)

sol2.u
```

The second solve reuses the stored factorization and only applies it to the new
right hand side.
