# Frequently Asked Questions

## How is LinearSolve.jl compared to just using normal \, i.e. A\b?

Check out [this video from JuliaCon 2022](https://www.youtube.com/watch?v=JWI34_w-yYw) which goes
into detail on how and why LinearSolve.jl can be a more general and efficient interface.

Note that if `\` is good enough for you, great! We still tend to use `\` in the REPL all the time!
However, if you're building a package, you may want to consider using LinearSolve.jl for the improved
efficiency and ability to choose solvers.

## I'm seeing some dynamic dispatches in the default algorithm choice, how do I reduce that?

Make sure you set the `OperatorAssumptions` to get the full performance, especially the `issquare` choice
as otherwise that will need to be determined at runtime.

## I found a faster algorithm that can be used than what LinearSolve.jl chose?

What assumptions are made as part of your method? If your method only works on well-conditioned operators, then
make sure you set the `WellConditioned` assumption in the `assumptions`. See the
[OperatorAssumptions page for more details](@ref assumptions). If using the right assumptions does not improve
the performance to the expected state, please open an issue and we will improve the default algorithm.

## Python's NumPy/SciPy just calls fast Fortran/C code, why would LinearSolve.jl be any better?

This is addressed in the [JuliaCon 2022 video](https://www.youtube.com/watch?v=JWI34_w-yYw&t=182s). This happens in
a few ways:

 1. The Fortran/C code that NumPy/SciPy uses is actually slow. It's [OpenBLAS](https://github.com/xianyi/OpenBLAS),
    a library developed in part by the Julia Lab back in 2012 as a fast open source BLAS implementation. Many
    open source environments now use this build, including many R distributions. However, the Julia Lab has greatly
    improved its ability to generate optimized SIMD in platform-specific ways. This, and improved multithreading support
    (OpenBLAS's multithreading is rather slow), has led to pure Julia-based BLAS implementations which the lab now
    works on. This includes [RecursiveFactorization.jl](https://github.com/JuliaLinearAlgebra/RecursiveFactorization.jl)
    which generally outperforms OpenBLAS by 2x-10x depending on the platform. It even outperforms MKL for small matrices
    (<100). LinearSolve.jl uses RecursiveFactorization.jl by default sometimes, but switches to BLAS when it would be
    faster (in a platform and matrix-specific way).
 2. Standard approaches to handling linear solves re-allocate the pivoting vector each time. This leads to GC pauses that
    can slow down calculations. LinearSolve.jl has proper caches for fully preallocated no-GC workflows.
 3. LinearSolve.jl makes many other optimizations, like factorization reuse and symbolic factorization reuse, automatic.
    Many of these optimizations are not even possible from the high-level APIs of things like Python's major libraries and MATLAB.
 4. LinearSolve.jl has a much more extensive set of sparse matrix solvers, which is why you see a major difference (2x-10x) for sparse
    matrices. Which sparse matrix solver between KLU, UMFPACK, Pardiso, etc. is optimal depends a lot on matrix sizes, sparsity patterns,
    and threading overheads. LinearSolve.jl's heuristics handle these kinds of issues.

## How do I use IterativeSolvers solvers with a weighted tolerance vector?

IterativeSolvers.jl computes the norm after the application of the left preconditioner.
Thus, in order to use a vector tolerance `weights`, one can mathematically
hack the system via the following formulation:

```@example FAQPrec
using LinearSolve, LinearAlgebra

n = 2
A = rand(n, n)
b = rand(n)

weights = [1e-1, 1]
precs = Returns((LinearSolve.InvPreconditioner(Diagonal(weights)), Diagonal(weights)))

prob = LinearProblem(A, b)
sol = solve(prob, KrylovJL_GMRES(precs))

sol.u
```

If you want to use a “real” preconditioner under the norm `weights`, then one
can use `ComposePreconditioner` to apply the preconditioner after the application
of the weights like as follows:

```@example FAQ2
using LinearSolve, LinearAlgebra

n = 4
A = rand(n, n)
b = rand(n)

weights = rand(n)
realprec = lu(rand(n, n)) # some random preconditioner
Pl = LinearSolve.ComposePreconditioner(LinearSolve.InvPreconditioner(Diagonal(weights)),
    realprec)
Pr = Diagonal(weights)

prob = LinearProblem(A, b)
sol = solve(prob, KrylovJL_GMRES(precs = Returns((Pl, Pr))))
```
