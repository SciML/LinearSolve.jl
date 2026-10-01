using LinearSolve, LinearAlgebra, Test, ForwardDiff, ReverseDiff

# ODE-free ReverseDiff reproducer: `lu_instance` drops TrackedReal origin tags via
# `zero`, so the default/LU cache slot must be typed from `eltype(A)`.
function loss_lu(p, alg)
    A = reshape([p[1], p[2], p[3], p[4]], 2, 2)
    return sum(solve(LinearProblem(A, [p[1], p[4]]), alg).u)
end

function loss_default_n(p, n)
    A = reshape([p[k] for k in 1:(n * n)], n, n)
    return sum(solve(LinearProblem(A, [p[k] for k in 1:n])).u)
end

@testset "ReverseDiff LU cache matches ForwardDiff" begin
    p0 = [3.0, 0.5, 0.2, 4.0]
    algs = (
        LUFactorization(),
        nothing,
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization),
    )
    for alg in algs
        g_fd = ForwardDiff.gradient(p -> loss_lu(p, alg), p0)
        g_rd = ReverseDiff.gradient(p -> loss_lu(p, alg), p0)
        @test isapprox(g_rd, g_fd; rtol = 1.0e-8, atol = 1.0e-10)
    end

    # Larger dense problem so the default algorithm selects LU (not GenericLU).
    n = 12
    pb = vec(Matrix(20.0 * I, n, n) .+ 0.1 .* reshape(1:(n * n), n, n) ./ (n * n))
    g_fd = ForwardDiff.gradient(p -> loss_default_n(p, n), pb)
    g_rd = ReverseDiff.gradient(p -> loss_default_n(p, n), pb)
    @test isapprox(g_rd, g_fd; rtol = 1.0e-8, atol = 1.0e-10)
end
