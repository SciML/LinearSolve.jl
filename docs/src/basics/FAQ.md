# Frequently Asked Questions

Ask more questions.

## How is LinearSolve.jl compared to just using normal \, i.e. A\b?

Check out [this video from JuliaCon 2022](https://www.youtube.com/watch?v=JWI34_w-yYw) which goes
into detail on how and why LinearSolve.jl is able to be a more general and efficient interface.

Note that if `\` is good enough for you, great! We still tend to use `\` in the REPL all of the time!
However, if you're building a package, you may want to consider using LinearSolve.jl for the improved
efficiency and ability to choose solvers.

## Python's NumPy/SciPy just calls fast Fortran/C code, why would LinearSolve.jl be any better?

This is addressed in the [JuliaCon 2022 video](https://youtu.be/JWI34_w-yYw?t=182). This happens in
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
3. LinearSolve.jl makes a lot of other optimizations, like factorization reuse and symbolic factorization reuse, automatic.
   Many of these optimizations are not even possible from the high-level APIs of things like Python's major libraries and MATLAB.
4. LinearSolve.jl has a much more extensive set of sparse matrix solvers, which is why you see a major difference (2x-10x) for sparse
   matrices. Which sparse matrix solver between KLU, UMFPACK, Pardiso, etc. is optimal depends a lot on matrix sizes, sparsity patterns,
   and threading overheads. LinearSolve.jl's heuristics handle these kinds of issues.

## How do I use IterativeSolvers solvers with a weighted tolerance vector?

IterativeSolvers.jl computes the norm after the application of the left precondtioner
`Pl`. Thus in order to use a vector tolerance `weights`, one can mathematically
hack the system via the following formulation:

```julia
using LinearSolve, LinearAlgebra
Pl = LinearSolve.InvPreconditioner(Diagonal(weights))
Pr = Diagonal(weights)

A = rand(n,n)
b = rand(n)

prob = LinearProblem(A,b)
sol = solve(prob,IterativeSolversJL_GMRES(),Pl=Pl,Pr=Pr)
```

If you want to use a "real" preconditioner under the norm `weights`, then one
can use `ComposePreconditioner` to apply the preconditioner after the application
of the weights like as follows:

```julia
using LinearSolve, LinearAlgebra
Pl = ComposePreconitioner(LinearSolve.InvPreconditioner(Diagonal(weights),realprec))
Pr = Diagonal(weights)

A = rand(n,n)
b = rand(n)

prob = LinearProblem(A,b)
sol = solve(prob,IterativeSolversJL_GMRES(),Pl=Pl,Pr=Pr)
```
