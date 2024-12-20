# Common Solver Options (Keyword Arguments for Solve)

While many algorithms have specific arguments within their constructor,
the keyword arguments for `solve` are common across all the algorithms
in order to give composability. These are also the options taken at `init` time.
The following are the options these algorithms take, along with their defaults.

## General Controls
  - `alias::LinearAliasSpecifier`: Holds the fields `alias_A` and `alias_b` which specify
    whether to alias the matrices `A` and `b` respectively. When these fields are `true`,
    `A` and `b` can be written to and changed by the solver algorithm. When fields are `nothing`
    the default behavior is used, which is to default to `true` when the algorithm is known 
    not to modify the matrices, and false otherwise.
  - `verbose`: Whether to print extra information. Defaults to `false`.
  - `assumptions`: Sets the assumptions of the operator in order to effect the default
    choice algorithm. See the [Operator Assumptions page for more details](@ref assumptions).

## Iterative Solver Controls

Error controls are not used by all algorithms. Specifically, direct solves always
solve completely. Error controls only apply to iterative solvers.

  - `abstol`: The absolute tolerance. Defaults to `√(eps(eltype(A)))`
  - `reltol`: The relative tolerance. Defaults to `√(eps(eltype(A)))`
  - `maxiters`: The number of iterations allowed. Defaults to `length(prob.b)`
  - `Pl,Pr`: The left and right preconditioners, respectively. For more information,
    see [the Preconditioners page](@ref prec).
