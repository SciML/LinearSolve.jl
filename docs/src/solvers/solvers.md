# [Linear System Solvers](@id linearsystemsolvers)

`solve(prob::LinearProblem,alg;kwargs)`

Solves for ``Au=b`` in the problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

## Recommended Methods

The default algorithm `nothing` is good for picking an algorithm that will work,
but one may need to change this to receive more performance or precision. If
more precision is necessary, `QRFactorization()` and `SVDFactorization()` are
the best choices, with SVD being the slowest but most precise.

For efficiency, `RFLUFactorization` is the fastest for dense LU-factorizations.
`FastLUFactorization` will be faster than `LUFactorization` which is the Base.LinearAlgebra
(`\` default) implementation of LU factorization. `SimpleLUFactorization` will be fast
on very small matrices.

For sparse LU-factorizations, `KLUFactorization` if there is less structure
to the sparsity pattern and `UMFPACKFactorization` if there is more structure.
Pardiso.jl's methods are also known to be very efficient sparse linear solvers.

While these sparse factorizations are based on implementations in other languages,
and therefore constrained to standard number types (`Float64`,  `Float32` and
their complex counterparts),  `SparspakFactorization` is able to handle general
number types, e.g. defined by `ForwardDiff.jl`, `MultiFloats.jl`,
or `IntervalArithmetics.jl`.

As sparse matrices get larger, iterative solvers tend to get more efficient than
factorization methods if a lower tolerance of the solution is required.

Krylov.jl generally outperforms IterativeSolvers.jl and KrylovKit.jl, and is compatible
with CPUs and GPUs, and thus is the generally preferred form for Krylov methods.

Finally, a user can pass a custom function for handling the linear solve using
`LinearSolveFunction()` if existing solvers are not optimally suited for their application.
The interface is detailed [here](@ref custom).

## Full List of Methods

### RecursiveFactorization.jl

```@docs
RFLUFactorization
```

### Base.LinearAlgebra

These overloads tend to work for many array types, such as `CuArrays` for GPU-accelerated
solving, using the overloads provided by the respective packages. Given that this can be
customized per-package, details given below describe a subset of important arrays
(`Matrix`, `SparseMatrixCSC`, `CuMatrix`, etc.)

```@docs
LUFactorization
GenericLUFactorization
QRFactorization
SVDFactorization
CholeskyFactorization
BunchKaufmanFactorization
CHOLMODFactorization
NormalCholeskyFactorization
NormalBunchKaufmanFactorization
```

### LinearSolve.jl

LinearSolve.jl contains some linear solvers built in for specailized cases.

```@docs
SimpleLUFactorization
DiagonalFactorization
```

### FastLapackInterface.jl

FastLapackInterface.jl is a package that allows for a lower-level interface to the LAPACK
calls to allow for preallocating workspaces to decrease the overhead of the wrappers.
LinearSolve.jl provides a wrapper to these routines in a way where an initialized solver
has a non-allocating LU factorization. In theory, this post-initialized solve should always
be faster than the Base.LinearAlgebra version.

```@docs
FastLUFactorization
FastQRFactorization
```

### SuiteSparse.jl

```@docs
KLUFactorization
UMFPACKFactorization
```

### Sparspak.jl

```@docs
SparspakFactorization
```

### Krylov.jl

```@docs
KrylovJL_CG
KrylovJL_MINRES
KrylovJL_GMRES
KrylovJL_BICGSTAB
KrylovJL_LSMR
KrylovJL_CRAIGMR
KrylovJL
```

### Pardiso.jl

!!! note
    
    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

```@docs
MKLPardisoFactorize
MKLPardisoIterate
PardisoJL
```

### CUDA.jl

Note that `CuArrays` are supported by `GenericFactorization` in the “normal” way.
The following are non-standard GPU factorization routines.

!!! note
    
    Using this solver requires adding the package CUDA.jl, i.e. `using CUDA`

```@docs
CudaOffloadFactorization
```

### IterativeSolvers.jl

!!! note
    
    Using these solvers requires adding the package IterativeSolvers.jl, i.e. `using IterativeSolvers`

```@docs
IterativeSolversJL_CG
IterativeSolversJL_GMRES
IterativeSolversJL_BICGSTAB
IterativeSolversJL_MINRES
IterativeSolversJL
```

### KrylovKit.jl

!!! note
    
    Using these solvers requires adding the package KrylovKit.jl, i.e. `using KrylovKit`

```@docs
KrylovKitJL_CG
KrylovKitJL_GMRES
KrylovKitJL
```

### HYPRE.jl

!!! note
    
    Using HYPRE solvers requires Julia version 1.9 or higher, and that the package HYPRE.jl
    is installed.

```@docs
HYPREAlgorithm
```
