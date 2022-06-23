# Efficiently Solving Large Linear Systems

```julia
using LinearSolve, LinearSolvePardiso, SparseArrays

A = sparse([1.0 0 -2 3
    0 5 1 2
    -2 1 4 -7
    3 2 -7 5])
b = rand(4)

prob = LinearProblem(A, b)

for alg in (
    MKLPardisoFactorize(),
    MKLPardisoIterate(),
    UMFPACKFactorization(),
    KLUFactorization())

    @time u = solve(prob, alg).u
end

using AlgebraicMultigrid
ml = ruge_stuben(A) # Construct a Ruge-Stuben solver
pl = aspreconditioner(ml)
solve(prob, KrylovJL_GMRES(), Pl = pl).u

using IncompleteLU
pl = ilu(A, τ = 0.1) # τ needs to be tuned per problem
solve(prob, KrylovJL_GMRES(), Pl = pl).u
```