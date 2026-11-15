# Bordered System Solvers

!!! note

    `BorderedLinearSolve` is a sub-library of LinearSolve.jl. Using these solvers
    requires adding it: `using BorderedLinearSolve`.

Solves for ``v`` and ``σ`` in

```math
\begin{pmatrix} J & b \\ c^{T} & d \end{pmatrix}
\begin{pmatrix} v \\ σ \end{pmatrix} =
\begin{pmatrix} f \\ g \end{pmatrix}
```

where `J` is `n × n` and the border `b`, `c` is `n × m` with `m` small. Systems of this
shape come up in continuation and bifurcation methods, where `J` is the Jacobian of a
large system and the border carries a few extra constraints.

`BorderingBLS` solves by block elimination against `J` itself, so its inner solves share
one LinearSolve cache, which carries over between steps where `J` changes only slightly.
`DirectBLS` assembles the whole matrix and solves it, which is fastest for a small dense
`J`.

```@example bordered
using BorderedLinearSolve, LinearSolve, LinearAlgebra
n = 100
J = rand(n, n) + n * I
b, c, f = rand(n), rand(n), rand(n)
prob = BorderedLinearProblem(J, b, c, 1.0, f, 2.0)
sol = solve(prob, BorderingBLS(LUFactorization()))
[J b; c' 1.0] * [sol.v; sol.σ] ≈ [f; 2.0]
```

```@docs
BorderedLinearSolve.BorderedLinearSolve
BorderedLinearProblem
solve(::BorderedLinearProblem, ::Any)
BorderingBLS
DirectBLS
```
