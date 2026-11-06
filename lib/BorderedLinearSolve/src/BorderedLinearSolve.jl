"""
    BorderedLinearSolve

Solvers for bordered linear systems

```
┌      ┐┌   ┐   ┌   ┐
│ J  b ││ v │ = │ f │
│ c' d ││ σ │   │ g │
└      ┘└   ┘   └   ┘
```

where `J` is `n x n` and the border `b`, `c` is `n x m` with `m` small (usually 1).
Systems of this shape come up in continuation and bifurcation work, where `J` is the
Jacobian of a large system and the border carries a handful of extra constraints.

The point of solving them this way rather than assembling the whole `(n+m) x (n+m)`
matrix is that every inner solve uses the *same* `J`, so one `LinearSolve` cache serves
them all, and it carries over between outer steps where `J` changes only slightly. That
is what makes factorization or Krylov subspace reuse available here.

See [`BorderedLinearProblem`](@ref), [`BorderingBLS`](@ref) and [`DirectBLS`](@ref).
"""
module BorderedLinearSolve

using LinearAlgebra: LinearAlgebra, dot, norm
using LinearSolve: LinearSolve, LinearProblem, init, solve!
using SciMLBase: SciMLBase, ReturnCode, solve
using SciMLOperators: FunctionOperator

export BorderedLinearProblem, BorderingBLS, DirectBLS
# Re-exported so the package can be used on its own, as the rest of SciML does.
export solve, init, solve!

include("problem.jl")
include("flatten.jl")
include("bordering.jl")
include("direct.jl")
include("interface.jl")

end
