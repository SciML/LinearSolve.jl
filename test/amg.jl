using LinearSolve, AlgebraicMultigrid, LinearAlgebra, SparseArrays, Test

@testset "AlgebraicMultigridJL" begin
    n = 100
    A = spdiagm(-1 => -ones(n - 1), 0 => 2 * ones(n), 1 => -ones(n - 1))
    b = rand(n)

    prob = LinearProblem(A, b)
    
    # Test Ruge-Stuben (default)
    sol = solve(prob, AlgebraicMultigridJL())
    @test norm(A * sol.u - b) < 1e-6
    
    # Test Smoothed Aggregation
    sol = solve(prob, AlgebraicMultigridJL(AlgebraicMultigrid.smoothed_aggregation))
    @test norm(A * sol.u - b) < 1e-6
    
    # Test with tolerances
    prob_tol = LinearProblem(A, b)
    sol = solve(prob_tol, AlgebraicMultigridJL(), reltol=1e-8)
    @test norm(A * sol.u - b) < 1e-8


    # Negative Test 1: Singular Matrix
    # Construct a singular matrix (rows are linearly dependent)
    # AMG might run but should fail to converge or produce a large residual
    A_singular = sparse([1.0 1.0; 1.0 1.0])
    b_singular = [1.0, 2.0] # No solution
    prob_singular = LinearProblem(A_singular, b_singular)
    # Just check it runs without crashing, we expect potentially poor results
    sol_singular = solve(prob_singular, AlgebraicMultigridJL())
    @test sol_singular.retcode == ReturnCode.Success || sol_singular.retcode == ReturnCode.MaxIters
    
    # Negative Test 2: Non-square Matrix
    # AlgebraicMultigrid requires square matrices.
    A_rect = sparse([1.0 1.0 0.0; 0.0 1.0 1.0])
    b_rect = [1.0, 1.0]
    prob_rect = LinearProblem(A_rect, b_rect)
    @test_throws AssertionError solve(prob_rect, AlgebraicMultigridJL())
end
