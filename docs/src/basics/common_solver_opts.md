# Common Solver Options (Keyword Arguments for Solve)

While many algorithms have specific arguments within their constructor,
the keyword arguments for `solve` are common across all of the algorithms
in order to give composability. These are also the options taken at `init` time.
The following are the options these algorithms take, along with their defaults.

## General Controls

  - `alias_A`: Whether to alias the matrix `A` or use a copy by default. When true,
    algorithms like LU-factorization can be faster by reusing the memory via `lu!`,
    but care must be taken as the original input will be modified. Default is `false`.
  - `alias_b`: Whether to alias the matrix `b` or use a copy by default. When true,
    algorithms can write and change `b` upon usage. Care must be taken as the
    original input will be modified. Default is `false`.
  - `verbose`: Whether to print extra information. Defaults to `false`.

## Iterative Solver Controls

Error controls are not used by all algorithms. Specifically, direct solves always
solve completely. Error controls only apply to iterative solvers.

  - `abstol`: The absolute tolerance. Defaults to `√(eps(eltype(A)))`
  - `reltol`: The relative tolerance. Defaults to `√(eps(eltype(A)))`
  - `maxiters`: The number of iterations allowed. Defaults to `length(prob.b)`
  - `Pl,Pr`: The left and right preconditioners respectively. For more information
    see [the Preconditioners page](@ref prec).
