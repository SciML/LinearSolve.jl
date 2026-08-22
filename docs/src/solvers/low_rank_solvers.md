# [Low-Rank Updated System Solvers](@id lowranksolvers)

`LS.solve(prob::LS.LinearProblem, alg; kwargs)` for a
[`LinearProblem`](@ref linear_problem) whose matrix is a
[`LowRankUpdatedMatrix`](@ref).

Solves for ``u`` in

```math
(A + U C V^{*})\, u = b
```

where the update has rank ``k \ll n``: `U` and `V` are `n × k`, and `C` is a `k × k`
middle factor that defaults to `I`, which gives the Sherman-Morrison form
``A + U V^{*}``. A vector `U` or `V` is read as a rank-1 update.

`LowRankUpdatedMatrix` is a matrix type, not an algorithm. You build it and hand it to a
`LinearProblem` in place of an assembled matrix, and the sum is never formed. The
algorithm argument therefore chooses how the *base* matrix `A` is factorized; the update
is absorbed on top of that factorization by the Woodbury identity

```math
(A + U C V^{*})^{-1} = A^{-1} - A^{-1} U (C^{-1} + V^{*} A^{-1} U)^{-1} V^{*} A^{-1}
```

If no algorithm is given, `LS.LUFactorization()` is used.

## Recommended Methods

`LS.defaultalg` returns `LS.LUFactorization()` for a `LowRankUpdatedMatrix`. That is a
safe choice rather than an informed one: the default is picked from the wrapper, so it
does not see whether `A` underneath is sparse, symmetric, or badly conditioned. Pass an
algorithm explicitly whenever `A` has structure worth using, and pick the same
factorization you would have picked for `A` on its own.

| base matrix `A`                | pass                                                        |
|:------------------------------ |:----------------------------------------------------------- |
| dense, reasonably conditioned  | `LS.LUFactorization()`, the default                         |
| sparse                         | `LS.LUFactorization()`, which dispatches `lu` to UMFPACK    |
| symmetric positive definite    | `LS.CholeskyFactorization()`                                |
| symmetric indefinite           | `LS.BunchKaufmanFactorization()`                            |
| ill conditioned                | `LS.QRFactorization()`, or `LS.SVDFactorization()` for the most precision |

The sparse row is the case the type exists for. ``U C V^{*}`` is dense even when `A` is
not, so an assembled `A + U C V^{*}` is a dense matrix and a dense factorization, whatever
`A` was. Keeping the update unassembled keeps the sparse factorization of `A`.

### Supported Factorizations

The Woodbury path is enumerated rather than dispatched generically, in
`LinearSolve._LOWRANK_ALGS`. Exactly these algorithms are supported:

  - [`LUFactorization`](@ref)
  - [`QRFactorization`](@ref)
  - [`CholeskyFactorization`](@ref)
  - [`SVDFactorization`](@ref)
  - [`BunchKaufmanFactorization`](@ref)

The list is exactly those algorithms that reach their factorization through
`LinearSolve.do_factorization`, which is how the correction gets at `A`. A sparse `A` is
covered by `LUFactorization`: it dispatches `lu` to UMFPACK, so it produces a sparse
factorization rather than a dense one. `UMFPACKFactorization` and `KLUFactorization`
themselves are not on the list, because they reach their factorizations through their own
`solve!` rather than through `do_factorization`.

!!! warning
    
    Nothing outside that list takes the Woodbury path. Other factorizations, Krylov
    methods, and the extension solvers do not know about the wrapper and will not exploit
    the update, so treat the list above as the whole supported surface.

## Cost

The point of the type is where the work lands. A `k × k` factorization replaces an
`n × n` one:

|                                                          | how often          | cost                                        |
|:-------------------------------------------------------- |:------------------ |:------------------------------------------- |
| factorize `A`                                            | once per matrix    | whatever the chosen factorization costs      |
| form ``A^{-1}U``, factorize ``C^{-1} + V^{*} A^{-1} U``  | once per matrix    | `k` solves against `A`, then a `k × k` LU    |
| solve                                                    | per right-hand side | one solve against `A`, then a `k × k` solve  |

Because the setup is charged once and the per-solve cost is dominated by the existing
factorization of `A`, the update is close to free across a sequence of right-hand sides.
The [caching interface](@ref "Linear Solve with Caching Interface") applies unchanged:
`LS.init` once, then assign `cache.b` and call `LS.solve!` for each new right-hand side.

## Limits

  - **The capacitance matrix must be nonsingular.** The identity needs both `A` and
    ``C^{-1} + V^{*} A^{-1} U`` invertible. An update that makes the whole matrix singular
    shows up as a singular capacitance matrix, and the solve returns a non-`Success`
    retcode rather than a plausible wrong answer.
  - **`k` must stay small.** The saving is `k ≪ n`. At large `k` the setup solves against
    `A` and the `k × k` factorization stop being cheap, and assembling the sum wins.
  - **Only the enumerated factorizations.** See the list above.
  - **It is a matrix, so it can be materialized.** `getindex` and `mul!` are defined, so
    anything that calls `Matrix(M)`, or that reads `M` elementwise, gets the assembled dense
    sum and loses every advantage the type was providing. `copy(M)` is not such a case: it
    returns another `LowRankUpdatedMatrix`, which is what keeps the wrapper alive through
    `LS.init`.
  - **Woodbury is not as stable as a direct factorization.** When the capacitance matrix is
    ill conditioned, the difference of two large terms in the identity loses digits that a
    factorization of the assembled matrix would have kept.

For a worked example, including the sparse base case and the caching loop, see the
[Low-Rank Updates of a Factorized Matrix](@ref lowrankupdates) tutorial. The factorizations
themselves are documented on the
[Linear System Solvers](@ref linearsystemsolvers) page.

## Reference

```@docs
LowRankUpdatedMatrix
```
