using LinearSolve
using ForwardDiff
using Test
using SparseArrays

function h(p)
    (A = [p[1] p[2]+1 p[2]^3;
          3*p[1] p[1]+5 p[2] * p[1]-4;
          p[2]^2 9*p[1] p[2]],
        b = [p[1] + 1, p[2] * 2, p[1]^2])
end

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
overload_x_p = solve(prob, LUFactorization())
backslash_x_p = A \ b
krylov_overload_x_p = solve(prob, KrylovJL_GMRES())
@test ≈(overload_x_p, backslash_x_p, rtol = 1e-9)
@test ≈(krylov_overload_x_p, backslash_x_p, rtol = 1e-9)

krylov_prob = LinearProblem(A, b, u0 = rand(3))
krylov_u0_sol = solve(krylov_prob, KrylovJL_GMRES())

@test ≈(krylov_u0_sol, backslash_x_p, rtol = 1e-9)

A, _ = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
backslash_x_p = A \ [6.0, 10.0, 25.0]
prob = LinearProblem(A, [6.0, 10.0, 25.0])

@test ≈(solve(prob).u, backslash_x_p, rtol = 1e-9)
@test ≈(solve(prob, KrylovJL_GMRES()).u, backslash_x_p, rtol = 1e-9)

_, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
A = [5.0 6.0 125.0; 15.0 10.0 21.0; 25.0 45.0 5.0]
backslash_x_p = A \ b
prob = LinearProblem(A, b)

@test ≈(solve(prob).u, backslash_x_p, rtol = 1e-9)
@test ≈(solve(prob, KrylovJL_GMRES()).u, backslash_x_p, rtol = 1e-9)

A, b = h([ForwardDiff.Dual(10.0, 1.0, 0.0), ForwardDiff.Dual(10.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

new_A, new_b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
cache.A = new_A
cache.b = new_b

@test cache.A == new_A
@test cache.b == new_b

x_p = solve!(cache)
backslash_x_p = new_A \ new_b

@test ≈(x_p, backslash_x_p, rtol = 1e-9)

# Just update A
A, b = h([ForwardDiff.Dual(10.0, 1.0, 0.0), ForwardDiff.Dual(10.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

new_A, _ = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
cache.A = new_A
@test cache.A == new_A

x_p = solve!(cache)
backslash_x_p = new_A \ b

@test ≈(x_p, backslash_x_p, rtol = 1e-9)

# Just update b
A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

_, new_b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
cache.b = new_b
@test cache.b == new_b

x_p = solve!(cache)
backslash_x_p = A \ new_b

@test ≈(x_p, backslash_x_p, rtol = 1e-9)

# Nested Duals
A,
b = h([ForwardDiff.Dual(ForwardDiff.Dual(5.0, 1.0, 0.0), 1.0, 0.0),
    ForwardDiff.Dual(ForwardDiff.Dual(5.0, 1.0, 0.0), 0.0, 1.0)])

prob = LinearProblem(A, b)
overload_x_p = solve(prob)

original_x_p = A \ b

@test ≈(overload_x_p, original_x_p, rtol = 1e-9)

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

new_A,
new_b = h([ForwardDiff.Dual(ForwardDiff.Dual(10.0, 1.0, 0.0), 1.0, 0.0),
    ForwardDiff.Dual(ForwardDiff.Dual(10.0, 1.0, 0.0), 0.0, 1.0)])

cache.A = new_A
cache.b = new_b

@test cache.A == new_A
@test cache.b == new_b

function linprob_f(p)
    A, b = h(p)
    prob = LinearProblem(A, b)
    solve(prob)
end

function slash_f(p)
    A, b = h(p)
    A \ b
end

@test ≈(
    ForwardDiff.jacobian(slash_f, [5.0, 5.0]), ForwardDiff.jacobian(linprob_f, [5.0, 5.0]))

@test ≈(ForwardDiff.jacobian(p -> ForwardDiff.jacobian(slash_f, [5.0, p[1]]), [5.0]),
    ForwardDiff.jacobian(p -> ForwardDiff.jacobian(linprob_f, [5.0, p[1]]), [5.0]))

function g(p)
    (A = [p[1] p[1]+1 p[1]^3;
          3*p[1] p[1]+5 p[1] * p[1]-4;
          p[1]^2 9*p[1] p[1]],
        b = [p[1] + 1, p[1] * 2, p[1]^2])
end

function slash_f_hes(p)
    A, b = g(p)
    x = A \ b
    sum(x)
end

function linprob_f_hes(p)
    A, b = g(p)
    prob = LinearProblem(A, b)
    x = solve(prob)
    sum(x)
end

@test ≈(ForwardDiff.hessian(slash_f_hes, [5.0]),
    ForwardDiff.hessian(linprob_f_hes, [5.0]))

# Test aliasing
A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())

new_A, new_b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
cache.A = new_A
cache.b = new_b

linu = [ForwardDiff.Dual(0.0, 0.0, 0.0), ForwardDiff.Dual(0.0, 0.0, 0.0),
    ForwardDiff.Dual(0.0, 0.0, 0.0)]
cache.u = linu
x_p = solve!(cache)
backslash_x_p = new_A \ new_b

@test linu == cache.u

# Test Float Only solvers

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(sparse(A), sparse(b))
overload_x_p = solve(prob, KLUFactorization())
backslash_x_p = A \ b

@test ≈(overload_x_p, backslash_x_p, rtol = 1e-9)

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(sparse(A), sparse(b))
overload_x_p = solve(prob, UMFPACKFactorization())
backslash_x_p = A \ b

@test ≈(overload_x_p, backslash_x_p, rtol = 1e-9)


# Test that GenericLU doesn't create a DualLinearCache
A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
@test init(prob, GenericLUFactorization()) isa LinearSolve.LinearCache

@test init(prob) isa LinearSolve.LinearCache