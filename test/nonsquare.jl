using LinearSolve
using SparseArrays, LinearAlgebra

m, n = 13, 3
A = sprand(m, n, 0.5); b = rand(m);
prob = LinearProblem(A, b)
solve(prob)