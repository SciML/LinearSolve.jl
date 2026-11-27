using LinearSolve, LinearAlgebra, Test

@testset "CertifiedLUQR" begin
    A = [3.0 1 0; 1 2 1; 0 1 3]
    x = [1.0, 2, 3]
    u0 = [3.0, 0, 1]
    cache = init(LinearProblem(A, A * x; u0 = copy(u0)), CertifiedLUQR(reference = :u0))
    sol = solve!(cache)
    @test sol.retcode == ReturnCode.Success
    @test sol.u ≈ x
    @test sol.stats.numerical_rank == 3
    @test sol.stats.relative_residual <= cache.reltol

    A = [1.0 1 0; 2 2 0; 0 0 1]
    b = [2.0, 4, 3]
    workspace = cache.cacheval.qr.factors
    reinit!(cache; A, b)
    sol = solve!(cache)
    @test sol.retcode == ReturnCode.Success
    @test sol.u ≈ [2.5, -0.5, 3]
    @test A * sol.u ≈ b
    @test sol.stats.numerical_rank == 2
    @test cache.cacheval.relative_residual == sol.stats.relative_residual
    @test cache.cacheval.qr.factors === workspace
    @test solve(LinearProblem(A, b; u0 = copy(u0)), CertifiedLUQR()).u ≈ [1, 1, 3]

    cache.b = [2.0, 5, 3]
    sol = solve!(cache)
    @test sol.retcode == ReturnCode.Infeasible
    @test sol.stats.numerical_rank == 2
    @test sol.stats.relative_residual > cache.reltol
end
