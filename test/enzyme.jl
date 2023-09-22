using Enzyme, FiniteDiff
using LinearSolve, LinearAlgebra, Test

n = 4
A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);
b2 = rand(n);
db2 = zeros(n);

function f(A, b1, b2; alg = LUFactorization())
    prob = LinearProblem(A, b1)

    sol1 = solve(prob, alg)

    s1 = sol1.u
    norm(s1)
end

f(A, b1, b2) # Uses BLAS

Enzyme.autodiff(Reverse, f, Duplicated(copy(A), dA), Duplicated(copy(b1), db1), Duplicated(copy(b2), db2))

dA2 = FiniteDiff.finite_difference_gradient(x->f(x,b1, b2), copy(A))
db12 = FiniteDiff.finite_difference_gradient(x->f(A,x, b2), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12
@test db2 == zeros(4)

A = rand(n, n);
dA = zeros(n, n);
dA2 = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);
db12 = zeros(n);

b2 = rand(n);
db2 = zeros(n);
db22 = zeros(n);

@test_broken Enzyme.autodiff(Reverse, f, Duplicated(copy(A), dA), BatchDuplicated(copy(b1), (db1, db12)), BatchDuplicated(copy(b2), (db2, db22)))
@test_broken Enzyme.autodiff(Reverse, f, BatchDuplicated(copy(A), (dA, dA2)), Duplicated(copy(b1), db1), Duplicated(copy(b2), db2))