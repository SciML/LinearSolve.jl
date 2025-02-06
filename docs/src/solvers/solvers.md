# [Linear System Solvers](@id linearsystemsolvers)

`solve(prob::LinearProblem,alg;kwargs)`

Solves for ``Au=b`` in the problem defined by `prob` using the algorithm
`alg`. If no algorithm is given, a default algorithm will be chosen.

## Recommended Methods

### Dense Matrices

The default algorithm `nothing` is good for picking an algorithm that will work,
but one may need to change this to receive more performance or precision. If
more precision is necessary, `QRFactorization()` and `SVDFactorization()` are
the best choices, with SVD being the slowest but most precise.

For efficiency, `RFLUFactorization` is the fastest for dense LU-factorizations until around
150x150 matrices, though this can be dependent on the exact details of the hardware. After this
point, `MKLLUFactorization` is usually faster on most hardware. Note that on Mac computers
that `AppleAccelerateLUFactorization` is generally always the fastest. `LUFactorization` will
use your base system BLAS which can be fast or slow depending on the hardware configuration.
`SimpleLUFactorization` will be fast only on very small matrices but can cut down on compile times.

For very large dense factorizations, offloading to the GPU can be preferred. Metal.jl can be used
on Mac hardware to offload, and has a cutoff point of being faster at around size 20,000 x 20,000
matrices (and only supports Float32). `CudaOffloadFactorization` can be more efficient at a
much smaller cutoff, possibly around size 1,000 x 1,000 matrices, though this is highly dependent
on the chosen GPU hardware. `CudaOffloadFactorization` requires a CUDA-compatible NVIDIA GPU.
CUDA offload supports Float64 but most consumer GPU hardware will be much faster on Float32
(many are >32x faster for Float32 operations than Float64 operations) and thus for most hardware
this is only recommended for Float32 matrices.

!!! note
    
    Performance details for dense LU-factorizations can be highly dependent on the hardware configuration.
    For details see [this issue](https://github.com/SciML/LinearSolve.jl/issues/357).
    If one is looking to best optimize their system, we suggest running the performance
    tuning benchmark.

### Sparse Matrices

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
with CPUs and GPUs, and thus is the generally preferred form for Krylov methods. The
choice of Krylov method should be the one most constrained to the type of operator one
has, for example if positive definite then `Krylov_CG()`, but if no good properties then
use `Krylov_GMRES()`.

Finally, a user can pass a custom function for handling the linear solve using
`LinearSolveFunction()` if existing solvers are not optimally suited for their application.
The interface is detailed [here](@ref custom).

### Lazy SciMLOperators

If the linear operator is given as a lazy non-concrete operator, such as a `FunctionOperator`,
then using a Krylov method is preferred in order to not concretize the matrix.
Krylov.jl generally outperforms IterativeSolvers.jl and KrylovKit.jl, and is compatible
with CPUs and GPUs, and thus is the generally preferred form for Krylov methods. The
choice of Krylov method should be the one most constrained to the type of operator one
has, for example if positive definite then `Krylov_CG()`, but if no good properties then
use `Krylov_GMRES()`.

!!! tip
    
    If your materialized operator is a uniform block diagonal matrix, then you can use
    `SimpleGMRES(; blocksize = <known block size>)` to further improve performance.
    This often shows up in Neural Networks where the Jacobian wrt the Inputs (almost always)
    is a Uniform Block Diagonal matrix of Block Size = size of the input divided by the
    batch size.

## Full List of Methods

### RecursiveFactorization.jl

!!! note
    
    Using this solver requires adding the package RecursiveFactorization.jl, i.e. `using RecursiveFactorization`

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

LinearSolve.jl contains some linear solvers built in for specialized cases.

```@docs
SimpleLUFactorization
DiagonalFactorization
SimpleGMRES
```

### FastLapackInterface.jl

FastLapackInterface.jl is a package that allows for a lower-level interface to the LAPACK
calls to allow for preallocating workspaces to decrease the overhead of the wrappers.
LinearSolve.jl provides a wrapper to these routines in a way where an initialized solver
has a non-allocating LU factorization. In theory, this post-initialized solve should always
be faster than the Base.LinearAlgebra version. In practice, with the way we wrap the solvers,
we do not see a performance benefit and in fact benchmarks tend to show this inhibits
performance.

!!! note
    
    Using this solver requires adding the package FastLapackInterface.jl, i.e. `using FastLapackInterface`

```@docs
FastLUFactorization
FastQRFactorization
```

### SuiteSparse.jl

!!! note
    
    Using this solver requires adding the package SparseArrays.jl, i.e. `using SparseArrays`

```@docs
KLUFactorization
UMFPACKFactorization
```

### Sparspak.jl

!!! note
    
    Using this solver requires adding the package Sparspak.jl, i.e. `using Sparspak`

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

### MKL.jl

```@docs
MKLLUFactorization
```

### AppleAccelerate.jl

!!! note
    
    Using this solver requires a Mac with Apple Accelerate. This should come standard in most "modern" Mac computers.

```@docs
AppleAccelerateLUFactorization
```

### Metal.jl

!!! note
    
    Using this solver requires adding the package Metal.jl, i.e. `using Metal`. This package is only compatible with Mac M-Series computers with a Metal-compatible GPU.

```@docs
MetalLUFactorization
```

### Pardiso.jl

!!! note
    
    Using this solver requires adding the package Pardiso.jl, i.e. `using Pardiso`

```@docs
MKLPardisoFactorize
MKLPardisoIterate
LinearSolve.PardisoJL
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
IterativeSolversJL_IDRS
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
