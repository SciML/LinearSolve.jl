# [Eigenvalue Problem Solvers](@id eigenvaluesolvers)

`LS.solve(prob::LS.EigenvalueProblem, alg; kwargs)`

Solves for the eigenpairs $$(\lambda, v)$$ of $$Av = \lambda v$$ (or the generalized
problem $$Av = \lambda Bv$$ when a second operator is supplied) defined by `prob` using
the algorithm `alg`. If no algorithm is given, `DenseEigen()` is used.

## Recommended Methods

The default `DenseEigen()` wraps `LinearAlgebra.eigen` and is the right choice for
small to moderately sized dense matrices, or whenever most/all of the spectrum is
needed. It always computes the full dense eigendecomposition before selecting the
requested eigenpairs, so its cost does not depend on `num_eigenpairs`.

For large sparse or structured matrices where only a handful of eigenpairs are needed,
an iterative Krylov-based backend is preferred, since these only ever factorize/apply
`A` (and `B`) and never form the dense decomposition:

  - `KrylovKitEigen()` is a solid default iterative choice: it supports both standard
    and generalized problems, extremal targets, and interior (`shift`) targets.
  - `ArpackJL()` is a mature, widely-used implicitly restarted Arnoldi/Lanczos solver,
    also supporting extremal and shifted targets on standard and generalized problems.
  - `LS.ArnoldiMethod()` is a pure-Julia implicitly restarted Arnoldi method. It has no
    direct `EigenvalueTarget.SmallestMagnitude` support; use `shift` (shift-and-invert)
    or another backend for interior/smallest eigenvalues instead.
  - `JacobiDavidsonJL()` is a target/interior method: it is most effective when a
    `shift` close to the desired eigenvalues is known. It does not support generalized
    eigenvalue problems.

## Full List of Methods

### LinearAlgebra (built-in)

```@docs
AbstractEigenvalueAlgorithm
DenseEigen
```

### Arpack.jl

!!! note

    Using this solver requires adding the package Arpack.jl, i.e. `using Arpack`

```@docs
ArpackJL
```

### ArnoldiMethod.jl

!!! note

    Using this solver requires adding the package ArnoldiMethod.jl, i.e. `using ArnoldiMethod`

```@docs
ArnoldiMethod
ArnoldiMethodJL
```

### KrylovKit.jl

!!! note

    Using this solver requires adding the package KrylovKit.jl, i.e. `using KrylovKit`

```@docs
KrylovKitEigen
```

### JacobiDavidson.jl

!!! note

    Using this solver requires adding the package JacobiDavidson.jl, i.e. `using JacobiDavidson`

```@docs
JacobiDavidsonJL
```
