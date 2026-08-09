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

 1. The Fortran/C code that NumPy/SciPy uses is actually slow. It's [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS),
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
import LinearSolve as LS
import LinearAlgebra as LA

n = 2
A = rand(n, n)
b = rand(n)

weights = [1e-1, 1]
precs = Returns((LS.InvPreconditioner(LA.Diagonal(weights)), LA.Diagonal(weights)))

prob = LS.LinearProblem(A, b)
sol = LS.solve(prob, LS.KrylovJL_GMRES(precs))

sol.u
```

If you want to use a “real” preconditioner under the norm `weights`, then one
can use `ComposePreconditioner` to apply the preconditioner after the application
of the weights like as follows:

```@example FAQ2
import LinearSolve as LS
import LinearAlgebra as LA

n = 4
A = rand(n, n)
b = rand(n)

weights = rand(n)
realprec = LA.lu(rand(n, n)) # some random preconditioner
Pl = LS.ComposePreconditioner(LS.InvPreconditioner(LA.Diagonal(weights)),
    realprec)
Pr = LA.Diagonal(weights)

prob = LS.LinearProblem(A, b)
sol = LS.solve(prob, LS.KrylovJL_GMRES(precs = Returns((Pl, Pr))))
```

## Why does LinearSolve.jl depend on MKL_jll, and how do I stop it from loading?

MKL is the fastest BLAS on most x86 hardware, often by a wide margin, so LinearSolve.jl
ships it by default and lets the default algorithm choice pick `MKLLUFactorization`
where it wins. Without it, most installations end up substantially slower.

If you would rather not load it, for example because you only solve small static-array
systems where it brings nothing, set the `LoadMKL_JLL` preference to `false`:

```julia
using Preferences, UUIDs

Preferences.set_preferences!(
    UUID("7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"),  # LinearSolve
    "LoadMKL_JLL" => false; force = true
)
```

The preference is read when LinearSolve.jl loads, so restart Julia afterwards and let
the package recompile. From then on `LinearSolve.usemkl` is `false`, MKL_jll is never
`using`'d, and the default algorithm falls back to the next best choice for your
matrix, typically `LUFactorization` or `RFLUFactorization`. Nothing else about the
interface changes and every algorithm you select explicitly keeps working.

Two details worth knowing:

  - The default is already architecture aware. MKL is only considered on `x86_64` and
    `i686`, and it is off by default on AMD EPYC CPUs, where it does not win. On other
    architectures such as Apple Silicon it is never loaded regardless of this setting.
  - The preference controls whether LinearSolve.jl *loads and uses* MKL_jll, not
    whether it is installed. MKL_jll stays a declared dependency, so it remains in the
    dependency graph and Pkg still installs it.
