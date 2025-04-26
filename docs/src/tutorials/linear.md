# Getting Started with Solving Linear Systems in Julia

A linear system $$Au=b$$ is specified by defining an `AbstractMatrix` or `AbstractSciMLOperator`.
For the sake of simplicity, this tutorial will start by only showcasing concrete matrices.

The following defines a matrix and a `LinearProblem` which is subsequently solved
by the default linear solver.

```@example linsys1
using LinearSolve

A = rand(4, 4)
b = rand(4)
prob = LinearProblem(A, b)
sol = solve(prob)
sol.u
```

Note that `solve(prob)` is equivalent to `solve(prob,nothing)` where `nothing`
denotes the choice of the default linear solver. This is equivalent to the
Julia built-in `A\b`, where the solution is recovered via `sol.u`. The power
of this package comes into play when changing the algorithms. For example,
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)
has some nice methods like GMRES which can be faster in some cases. With
LinearSolve.jl, there is one interface and changing linear solvers is simply
the switch of the algorithm choice:

```@example linsys1
sol = solve(prob, KrylovJL_GMRES())
sol.u
```

Thus, a package which uses LinearSolve.jl simply needs to allow the user to
pass in an algorithm struct and all wrapped linear solvers are immediately
available as tweaks to the general algorithm. For more information on the
available solvers, see [the solvers page](@ref linearsystemsolvers)

## Sparse and Structured Matrices

There is no difference in the interface for using LinearSolve.jl on sparse
and structured matrices. For example, the following now uses Julia's
built-in [SparseArrays.jl](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
to define a sparse matrix (`SparseMatrixCSC`) and solve the system using LinearSolve.jl.
Note that `sprand` is a shorthand for quickly creating a sparse random matrix
(see SparseArrays.jl for more details on defining sparse matrices).

```@example linsys1
using LinearSolve, SparseArrays

A = sprand(4, 4, 0.75)
b = rand(4)
prob = LinearProblem(A, b)
sol = solve(prob)
sol.u

sol = solve(prob, KrylovJL_GMRES()) # Choosing algorithms is done the same way
sol.u
```

Similerly structure matrix types, like banded matrices, can be provided using special matrix
types. While any `AbstractMatrix` type should be compatible via the general Julia interfaces,
LinearSolve.jl specifically tests with the following cases:

* [BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl)
* [BlockDiagonals.jl](https://github.com/JuliaArrays/BlockDiagonals.jl)
* [CUDA.jl](https://cuda.juliagpu.org/stable/) (CUDA GPU-based dense and sparse matrices)
* [FastAlmostBandedMatrices.jl](https://github.com/SciML/FastAlmostBandedMatrices.jl)
* [Metal.jl](https://metal.juliagpu.org/stable/) (Apple M-series GPU-based dense matrices)

## Using Matrix-Free Operators

In many cases where a sparse matrix gets really large, even the sparse representation
cannot be stored in memory. However, in many such cases, such as with PDE discretizations,
you may be able to write down a function that directly computes `A*x`. These "matrix-free"
operators allow the user to define the `Ax=b` problem to be solved giving only the definition
of `A*x` and allowing specific solvers (Krylov methods) to act without ever constructing
the full matrix.

**This will be documented in more detail in the near future**
