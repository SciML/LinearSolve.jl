using LinearSolve, LinearAlgebra, Random, Test

@testset "ShiftedJacobian is I - γJ" begin
    for n in (1, 2, 3, 17)
        J = randn(MersenneTwister(n), n, n)
        x = randn(MersenneTwister(n + 1), n)
        γ = 0.31
        W = ShiftedJacobian(J, γ)
        Wd = I - γ * J
        @test size(W) == size(J)
        @test Matrix(W) ≈ Wd
        @test W ≈ Wd
        y = similar(x)
        @test mul!(y, W, x) ≈ Wd * x
        y .= 1
        @test mul!(y, W, x, 2.0, 3.0) ≈ 2 * (Wd * x) .+ 3
        @test Matrix(ShiftedJacobian(J, 2.0, -0.5)) ≈ 2 * J - 0.5I
        @test Matrix(copy(W)) ≈ Wd
    end
end

@testset "LHL solves I - γJ" begin
    for n in (1, 2, 3, 5, 40, 129), balance in (true, false), refine in (0, 1)
        J = randn(MersenneTwister(n), n, n)
        b = randn(MersenneTwister(n + 7), n)
        alg = LHLFactorization(; balance, refine)
        for γ in (0.0, 1.0e-8, 0.05, 1.7, -2.3)
            W = ShiftedJacobian(J, γ)
            u = solve(LinearProblem(W, b), alg).u
            @test u ≈ Matrix(I - γ * J) \ b rtol = 1.0e-9
        end
    end
end

@testset "update_gamma! matches a fresh factorization" begin
    n = 80
    J = randn(MersenneTwister(3), n, n)
    b = randn(MersenneTwister(4), n)
    cache = init(LinearProblem(ShiftedJacobian(J, 0.01), b), LHLFactorization())
    solve!(cache)
    for γ in (0.02, 1.0e-6, 5.0, 0.02)
        update_gamma!(cache, γ)
        u = copy(solve!(cache).u)
        @test u ≈ Matrix(I - γ * J) \ b rtol = 1.0e-9
        # identical to reducing from scratch at that γ
        fresh = solve(LinearProblem(ShiftedJacobian(J, γ), b), LHLFactorization()).u
        @test u ≈ fresh rtol = 1.0e-9
    end
end

@testset "update_shift! and the W-transform form" begin
    n = 60
    J = randn(MersenneTwister(5), n, n)
    b = randn(MersenneTwister(6), n)
    cache = init(LinearProblem(ShiftedJacobian(J, 1.0, -3.0), b), LHLFactorization())
    @test solve!(cache).u ≈ (J - 3I) \ b rtol = 1.0e-9
    for dtγ in (0.1, 1.0e-4, 20.0)
        update_shift!(cache, 1.0, -inv(dtγ))
        @test copy(solve!(cache).u) ≈ (J - inv(dtγ) * I) \ b rtol = 1.0e-9
    end
end

@testset "mark_jacobian_updated! forces a new reduction" begin
    n = 50
    J = randn(MersenneTwister(7), n, n)
    b = randn(MersenneTwister(8), n)
    W = ShiftedJacobian(J, 0.1)
    cache = init(LinearProblem(W, b), LHLFactorization(; refine = 0))
    solve!(cache)
    J .= randn(MersenneTwister(9), n, n)
    mark_jacobian_updated!(W)
    @test copy(solve!(cache).u) ≈ Matrix(I - 0.1 * J) \ b rtol = 1.0e-9
    @test mark_jacobian_updated!(rand(2, 2)) isa Matrix   # no-op off the type
end

@testset "plain matrix" begin
    n = 70
    A = randn(MersenneTwister(11), n, n)
    b = randn(MersenneTwister(12), n)
    @test solve(LinearProblem(A, b), LHLFactorization()).u ≈ A \ b rtol = 1.0e-9
end

@testset "complex" begin
    n = 30
    J = randn(MersenneTwister(13), ComplexF64, n, n)
    b = randn(MersenneTwister(14), ComplexF64, n)
    γ = 0.2 + 0.3im
    @test solve(LinearProblem(ShiftedJacobian(J, γ), b), LHLFactorization()).u ≈
        Matrix(I - γ * J) \ b rtol = 1.0e-9
end

@testset "singular W reports failure" begin
    J = [1.0 0.0; 0.0 2.0]
    b = [1.0, 1.0]
    sol = solve(LinearProblem(ShiftedJacobian(J, 1.0), b), LHLFactorization())
    @test !SciMLBase.successful_retcode(sol.retcode)
end

@testset "update_gamma! falls back for other algorithms" begin
    n = 40
    J = randn(MersenneTwister(15), n, n)
    b = randn(MersenneTwister(16), n)
    cache = init(LinearProblem(ShiftedJacobian(J, 0.1), b), LUFactorization())
    solve!(cache)
    update_gamma!(cache, 0.4)
    @test copy(solve!(cache).u) ≈ Matrix(I - 0.4 * J) \ b rtol = 1.0e-9
    @test_throws ArgumentError update_gamma!(
        init(LinearProblem(rand(3, 3), rand(3)), LHLFactorization()), 0.1
    )
end

@testset "refinement recovers backward error on a hostile Jacobian" begin
    # Strictly upper triangular with a tiny corner: every pivot of the reduction is
    # near zero, so κ(Z) is enormous and the unrefined solve loses ~8 digits.
    n = 60
    J = triu(randn(MersenneTwister(17), n, n), 1)
    J[n, 1] = 1.0e-8
    b = randn(MersenneTwister(18), n)
    γ = 0.5
    W = Matrix(I - γ * J)
    ref = W \ b
    raw = solve(LinearProblem(ShiftedJacobian(J, γ), b), LHLFactorization(; refine = 0)).u
    ref1 = solve(LinearProblem(ShiftedJacobian(J, γ), b), LHLFactorization(; refine = 1)).u
    bwd(x) = norm(b - W * x, Inf) / (opnorm(W, Inf) * norm(x, Inf) + norm(b, Inf))
    @test bwd(ref1) <= bwd(raw)
    @test bwd(ref1) < 1.0e-13
end

@testset "adjoint solve" begin
    n = 30
    J = randn(MersenneTwister(19), n, n)
    b = randn(MersenneTwister(20), n)
    γ = 0.4
    cache = init(LinearProblem(ShiftedJacobian(J, γ), b), LHLFactorization())
    @test LinearSolve._adjoint_factorization_reuse(LHLFactorization) isa
        LinearSolve._NoAdjointFactorizationReuse
    @test adjoint(Matrix(I - γ * J)) \ b ≈
        solve(LinearProblem(adjoint(Matrix(I - γ * J)), b), LHLFactorization()).u rtol = 1.0e-9
end
