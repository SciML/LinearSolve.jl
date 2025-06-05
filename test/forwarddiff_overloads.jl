using LinearSolve
using ForwardDiff
using Test

function h(p)
    (A = [p[1] p[2]+1 p[2]^3;
          3*p[1] p[1]+5 p[2] * p[1]-4;
          p[2]^2 9*p[1] p[2]],
        b = [p[1] + 1, p[2] * 2, p[1]^2])
end

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
overload_x_p = solve(prob)
original_x_p = A \ b

@test ≈(overload_x_p, original_x_p, rtol = 1e-9)

A, _ = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
prob = LinearProblem(A, [6.0, 10.0, 25.0])
@test ≈(solve(prob).u, A \ [6.0, 10.0, 25.0], rtol = 1e-9) 

_, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
A = [5.0 6.0 125.0; 15.0 10.0 21.0; 25.0 45.0 5.0]
prob = LinearProblem(A, b)
@test ≈(solve(prob).u, A \ b, rtol = 1e-9)

A, b = h([ForwardDiff.Dual(10.0, 1.0, 0.0), ForwardDiff.Dual(10.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
cache = init(prob)

new_A, new_b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])
cache.A = new_A
cache.b = new_b

x_p = solve!(cache)
other_x_p = new_A \ new_b

@test ≈(x_p, other_x_p, rtol = 1e-9)