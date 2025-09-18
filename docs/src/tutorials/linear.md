# Getting Started with Solving Linear Systems in Julia

A linear system $$Au=b$$ is specified by defining an `AbstractMatrix` or `AbstractSciMLOperator`.
For the sake of simplicity, this tutorial will start by only showcasing concrete matrices.
And specifically, we will start by using the basic Julia `Matrix` type.

The following defines a `Matrix` and a `LinearProblem` which is subsequently solved
by the default linear solver.

```@example linsys1
import LinearSolve as LS

A = rand(4, 4)
b = rand(4)
prob = LS.LinearProblem(A, b)
sol = LS.solve(prob)
sol.u
```

Note that `LS.solve(prob)` is equivalent to `LS.solve(prob,nothing)` where `nothing`
denotes the choice of the default linear solver. This is equivalent to the
Julia built-in `A\b`, where the solution is recovered via `sol.u`. The power
of this package comes into play when changing the algorithms. For example,
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl)
has some nice methods like GMRES which can be faster in some cases. With
LinearSolve.jl, there is one interface and changing linear solvers is simply
the switch of the algorithm choice:

```@example linsys1
sol = LS.solve(prob, LS.KrylovJL_GMRES())
sol.u
```

Thus, a package which uses LinearSolve.jl simply needs to allow the user to
pass in an algorithm struct and all wrapped linear solvers are immediately
available as tweaks to the general algorithm. For more information on the
available solvers, see [the solvers page](@ref linearsystemsolvers)

## Sparse and Structured Matrices

There is no difference in the interface for LinearSolve.jl on sparse
and structured matrices. For example, the following now uses Julia's
built-in [SparseArrays.jl](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
to define a sparse matrix (`SparseMatrixCSC`) and solve the system with LinearSolve.jl.
Note that `sprand` is a shorthand for quickly creating a sparse random matrix
(see SparseArrays.jl for more details on defining sparse matrices).

```@example linsys1
import LinearSolve as LS
import SparseArrays as SA

A = SA.sprand(4, 4, 0.75)
b = rand(4)
prob = LS.LinearProblem(A, b)
sol = LS.solve(prob)
sol.u

sol = LS.solve(prob, LS.KrylovJL_GMRES()) # Choosing algorithms is done the same way
sol.u
```

Similarly structure matrix types, like banded matrices, can be provided using special matrix
types. While any `AbstractMatrix` type should be compatible via the general Julia interfaces,
LinearSolve.jl specifically tests with the following cases:

  - [LinearAlgebra.jl](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
    
      + Symmetric
      + Hermitian
      + UpperTriangular
      + UnitUpperTriangular
      + LowerTriangular
      + UnitLowerTriangular
      + SymTridiagonal
      + Tridiagonal
      + Bidiagonal
      + Diagonal

  - [BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl) `BandedMatrix`
  - [BlockDiagonals.jl](https://github.com/JuliaArrays/BlockDiagonals.jl) `BlockDiagonal`
  - [CUDA.jl](https://cuda.juliagpu.org/stable/) (CUDA GPU-based dense and sparse matrices) `CuArray` (`GPUArray`)
  - [FastAlmostBandedMatrices.jl](https://github.com/SciML/FastAlmostBandedMatrices.jl) `FastAlmostBandedMatrix`
  - [Metal.jl](https://metal.juliagpu.org/stable/) (Apple M-series GPU-based dense matrices) `MetalArray`

!!! note
    
    Choosing the most specific matrix structure that matches your specific system will give you the most performance.
    Thus if your matrix is symmetric, specifically building with `Symmetric(A)` will be faster than simply using `A`,
    and will generally lead to better automatic linear solver choices. Note that you can also choose the type for `b`,
    but generally a dense vector will be the fastest here and many solvers will not support a sparse `b`.

## Using Matrix-Free Operators via SciMLOperators.jl

In many cases where a sparse matrix gets really large, even the sparse representation
cannot be stored in memory. However, in many such cases, such as with PDE discretizations,
you may be able to write down a function that directly computes `A*x`. These "matrix-free"
operators allow the user to define the `Ax=b` problem to be solved giving only the definition
of `A*x` and allowing specific solvers (Krylov methods) to act without ever constructing
the full matrix.

The Matrix-Free operators are provided by the [SciMLOperators.jl interface](https://docs.sciml.ai/SciMLOperators/stable/).
For example, for the matrix `A` defined via:

```@example linsys1
A = [-2.0 1 0 0 0
     1 -2 1 0 0
     0 1 -2 1 0
     0 0 1 -2 1
     0 0 0 1 -2]
```

We can define the `FunctionOperator` that does the `A*v` operations, without using the matrix `A`. This is done by defining
a function `func(w,v,u,p,t)` which calculates `w = A(u,p,t)*v` (for the purposes of this tutorial, `A` is just a constant
operator. See the [SciMLOperators.jl documentation](https://docs.sciml.ai/SciMLOperators/stable/) for more details on defining
non-constant operators, operator algebras, and many more features). This is done by:

```@example linsys1
function Afunc!(w, v, u, p, t)
    w[1] = -2v[1] + v[2]
    for i in 2:4
        w[i] = v[i - 1] - 2v[i] + v[i + 1]
    end
    w[5] = v[4] - 2v[5]
    nothing
end

function Afunc!(v, u, p, t)
    w = zeros(5)
    Afunc!(w, v, u, p, t)
    w
end

import SciMLOperators as SMO
mfopA = SMO.FunctionOperator(Afunc!, zeros(5), zeros(5))
```

Let's check these are the same:

```@example linsys1
v = rand(5)
mfopA*v - A*v
```

Notice `mfopA` does this without having to have `A` because it just uses the equivalent `Afunc!` instead. Now, even though
we don't have a matrix, we can still solve linear systems defined by this operator. For example:

```@example linsys1
b = rand(5)
prob = LS.LinearProblem(mfopA, b)
sol = LS.solve(prob)
sol.u
```

And we can check this is successful:

```@example linsys1
mfopA * sol.u - b
```

!!! note
    
    Note that not all methods can use a matrix-free operator. For example, `LS.LUFactorization()` requires a matrix. If you use an
    invalid method, you will get an error. The methods particularly from KrylovJL are the ones preferred for these cases
    (and are defaulted to).
