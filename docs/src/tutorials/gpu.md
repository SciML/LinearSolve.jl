# GPU-Accelerated Linear Solving in Julia

LinearSolve.jl provides two ways to GPU accelerate linear solves:

* Offloading: offloading takes a CPU-based problem and automatically transforms it into a
  GPU-based problem in the background, and returns the solution on CPU. Thus using
  offloading requires no change on the part of the user other than to choose an offloading
  solver.
* Array type interface: the array type interface requires that the user defines the
  `LinearProblem` using an `AbstractGPUArray` type and chooses an appropriate solver
  (or uses the default solver). The solution will then be returned as a GPU array type.

The offloading approach has the advantage of being simpler and requiring no change to
existing CPU code, while having the disadvantage of having more overhead. In the following
sections we will demonstrate how to use each of the approaches.

!!! warn

    GPUs are not always faster! Your matrices need to be sufficiently large in order for
    GPU accelerations to actually be faster. For offloading it's around 1,000 x 1,000 matrices
    and for Array type interface it's around 100 x 100. For sparse matrices, it is highly
    dependent on the sparsity pattern and the amount of fill-in.

## GPU-Offloading

GPU offloading is simple as it's done simply by changing the solver algorithm. Take the
example from the start of the documentation:

```julia
using LinearSolve

A = rand(4, 4)
b = rand(4)
prob = LinearProblem(A, b)
sol = solve(prob)
sol.u
```

This computation can be moved to the GPU by the following:

```julia
using CUDA # Add the GPU library
sol = solve(prob, CudaOffloadFactorization())
sol.u
```

## GPUArray Interface

For more manual control over the factorization setup, you can use the
[GPUArray interface](https://juliagpu.github.io/GPUArrays.jl/dev/), the most common
instantiation being [CuArray for CUDA-based arrays on NVIDIA GPUs](https://cuda.juliagpu.org/stable/usage/array/).
To use this, we simply send the matrix `A` and the value `b` over to the GPU and solve:

```julia
using CUDA

A = rand(4, 4) |> cu
b = rand(4) |> cu
prob = LinearProblem(A, b)
sol = solve(prob)
sol.u
```

```
4-element CuArray{Float32, 1, CUDA.DeviceMemory}:
 -27.02665
  16.338171
 -77.650116
 106.335686
```

Notice that the solution is a `CuArray`, and thus one must use `Array(sol.u)` if you with
to return it to the CPU. This setup does no automated memory transfers and will thus only
move things to CPU on command.

!!! warn

    Many GPU functionalities, such as `CUDA.cu`, have a built-in preference for `Float32`.
    Generally it is much faster to use 32-bit floating point operations on GPU than 64-bit
    operations, and thus this is generally the right choice if going to such platforms.
    However, this change in numerical precision needs to be accounted for in your mathematics
    as it could lead to instabilities. To disable this, use a constructor that is more
    specific about the bitsize, such as `CuArray{Float64}(A)`. Additionally, preferring more
    stable factorization methods, such as `QRFactorization()`, can improve the numerics in
    such cases.

Similarly to other use cases, you can choose the solver, for example:

```julia
sol = solve(prob, QRFactorization())
```

## Sparse Matrices on GPUs

Currently, sparse matrix computations on GPUs are only supported for CUDA. This is done using
the `CUDA.CUSPARSE` sublibrary.

```julia
using LinearAlgebra, CUDA.CUSPARSE
T = Float32
n = 100
A_cpu = sprand(T, n, n, 0.05) + I
x_cpu = zeros(T, n)
b_cpu = rand(T, n)

A_gpu_csr = CuSparseMatrixCSR(A_cpu)
b_gpu = CuVector(b_cpu)
```

In order to solve such problems using a direct method, you must add
[CUDSS.jl](https://github.com/exanauts/CUDSS.jl). This looks like:

```julia
using CUDSS
sol = solve(prob, LUFactorization())
```

!!! note

    For now, CUDSS only supports CuSparseMatrixCSR type matrices.

Note that `KrylovJL` methods also work with sparse GPU arrays:

```julia
sol = solve(prob, KrylovJL_GMRES())
```

Note that CUSPARSE also has some GPU-based preconditioners, such as a built-in `ilu`. However:

```julia
sol = solve(prob, KrylovJL_GMRES(precs = (A, p) -> (CUDA.CUSPARSE.ilu02!(A, 'O'), I)))
```

However, right now CUSPARSE is missing the right `ldiv!` implementation for this to work
in general. See https://github.com/SciML/LinearSolve.jl/issues/341 for details.
