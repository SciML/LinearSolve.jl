using LinearSolve
using ForwardDiff


function h(p)
    (A = [p[1] p[2]+1 p[2]^3;
          3*p[1] p[1]+5 p[2] * p[1]-4;
          p[2]^2 9*p[1] p[2]],
        b = [p[1] + 1, p[2] * 2, p[1]^2])
end

A, b = h([ForwardDiff.Dual(5.0, 1.0, 0.0), ForwardDiff.Dual(5.0, 0.0, 1.0)])

prob = LinearProblem(A, b)
solve(prob)


