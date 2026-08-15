using LinearSolve, LinearAlgebra, Random, SciMLOperators, Test
using SciMLOperators: WOperator, jacobian_version

# `LHLFactorization` takes its system matrix unassembled, as the split `J - M/γ` a
# `WOperator` holds, so that a new γ never touches the reduction of J.
wop(J, γ; u = zeros(size(J, 1))) = WOperator{true}(I, γ, J, u)
dense(J, γ) = Matrix(J - I / γ)

@testset "solves J - M/γ" begin
    for n in (1, 2, 3, 5, 40, 129), balance in (true, false), refine in (0, 1)
        J = randn(MersenneTwister(n), n, n)
        b = randn(MersenneTwister(n + 7), n)
        alg = LHLFactorization(; balance, refine)
        for γ in (1.0e-8, 0.05, 1.7, -2.3)
            W = wop(J, γ)
            @test solve(LinearProblem(W, b), alg).u ≈ dense(J, γ) \ b rtol = 1.0e-9
        end
    end
end

@testset "update_gamma! matches a fresh factorization" begin
    n = 80
    J = randn(MersenneTwister(3), n, n)
    b = randn(MersenneTwister(4), n)
    cache = init(LinearProblem(wop(J, 0.01), b), LHLFactorization())
    solve!(cache)
    for γ in (0.02, 1.0e-6, 5.0, 0.02)
        update_gamma!(cache, γ)
        u = copy(solve!(cache).u)
        @test u ≈ dense(J, γ) \ b rtol = 1.0e-9
        @test u ≈ solve(LinearProblem(wop(J, γ), b), LHLFactorization()).u rtol = 1.0e-9
    end
end

@testset "mark_jacobian_updated! forces a new reduction" begin
    n = 50
    J = randn(MersenneTwister(7), n, n)
    b = randn(MersenneTwister(8), n)
    W = wop(J, 0.1)
    cache = init(LinearProblem(W, b), LHLFactorization(; refine = 0))
    solve!(cache)
    ws = cache.cacheval
    @test ws.jac_version == jacobian_version(W)

    # Without the announcement an in-place write to J is invisible and the stale
    # reduction is reused; with it, the next solve reduces again.
    J .= randn(MersenneTwister(9), n, n)
    stale = copy(solve!(cache).u)
    @test !isapprox(stale, dense(J, 0.1) \ b, rtol = 1.0e-6)
    mark_jacobian_updated!(W)
    @test copy(solve!(cache).u) ≈ dense(J, 0.1) \ b rtol = 1.0e-9
    @test ws.jac_version == jacobian_version(W)
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
    @test solve(LinearProblem(wop(J, γ; u = zeros(ComplexF64, n)), b), LHLFactorization()).u ≈
        dense(J, γ) \ b rtol = 1.0e-9
end

@testset "singular shift reports failure" begin
    J = [1.0 0.0; 0.0 2.0]
    b = [1.0, 1.0]
    sol = solve(LinearProblem(wop(J, 1.0), b), LHLFactorization())
    @test !SciMLBase.successful_retcode(sol.retcode)
end

@testset "unsupported inputs are rejected" begin
    n = 20
    J = randn(MersenneTwister(15), n, n)
    b = randn(MersenneTwister(16), n)
    # A general mass matrix would need a Hessenberg–triangular reduction of the pencil.
    Wm = WOperator{true}(Diagonal(collect(1.0:n)), 0.1, J, zeros(n))
    @test_throws ArgumentError solve(LinearProblem(Wm, b), LHLFactorization())
    @test_throws ArgumentError update_gamma!(
        init(LinearProblem(rand(3, 3), rand(3)), LHLFactorization()), 0.1
    )
end

@testset "defaultalg picks it from the split form" begin
    b200 = randn(MersenneTwister(2), 200)
    J200 = randn(MersenneTwister(1), 200, 200)
    @test LinearSolve.defaultalg(
        wop(J200, 0.1), b200, LinearSolve.OperatorAssumptions(true)
    ) isa LHLFactorization
    s = solve(LinearProblem(wop(J200, 0.1), b200))
    @test s.u ≈ dense(J200, 0.1) \ b200 rtol = 1.0e-9

    # Too small to pay for the reduction, or a general mass matrix: both keep whatever
    # the pre-existing operator path chose, rather than being intercepted.
    b20 = randn(MersenneTwister(4), 20)
    J20 = randn(MersenneTwister(3), 20, 20)
    small = LinearSolve.defaultalg(wop(J20, 0.1), b20, LinearSolve.OperatorAssumptions(true))
    @test !(small isa LHLFactorization)
    @test small == @invoke LinearSolve.defaultalg(
        wop(J20, 0.1)::SciMLOperators.AbstractSciMLOperator, b20,
        LinearSolve.OperatorAssumptions(true)
    )
    Wmm = WOperator{true}(Diagonal(collect(1.0:200)), 0.1, J200, zeros(200))
    @test !(
        LinearSolve.defaultalg(Wmm, b200, LinearSolve.OperatorAssumptions(true)) isa
            LHLFactorization
    )
end

@testset "update_gamma! works for algorithms that are not LHL" begin
    # Below the size cutoff the default is the operator (Krylov) path, so this exercises
    # the generic branch: set gamma, invalidate, let the next solve do whatever it does.
    n = 20
    J = randn(MersenneTwister(15), n, n)
    b = randn(MersenneTwister(16), n)
    cache = init(LinearProblem(wop(J, 0.1), b), reltol = 1.0e-12, abstol = 1.0e-12)
    @test !(cache.alg isa LHLFactorization)
    solve!(cache)
    update_gamma!(cache, 0.4)
    @test copy(solve!(cache).u) ≈ dense(J, 0.4) \ b rtol = 1.0e-6
end

@testset "refinement recovers backward error on a hostile Jacobian" begin
    # Strictly upper triangular with a tiny corner: every pivot of the reduction is
    # near zero, so κ(Z) is enormous and the unrefined solve loses ~8 digits.
    n = 60
    J = triu(randn(MersenneTwister(17), n, n), 1)
    J[n, 1] = 1.0e-8
    b = randn(MersenneTwister(18), n)
    γ = 2.0
    W = dense(J, γ)
    raw = solve(LinearProblem(wop(J, γ), b), LHLFactorization(; refine = 0)).u
    ref1 = solve(LinearProblem(wop(J, γ), b), LHLFactorization(; refine = 1)).u
    bwd(x) = norm(b - W * x, Inf) / (opnorm(W, Inf) * norm(x, Inf) + norm(b, Inf))
    @test bwd(ref1) <= bwd(raw)
    @test bwd(ref1) < 1.0e-13
end

@testset "adjoint solve" begin
    n = 30
    J = randn(MersenneTwister(19), n, n)
    b = randn(MersenneTwister(20), n)
    @test LinearSolve._adjoint_factorization_reuse(LHLFactorization) isa
        LinearSolve._NoAdjointFactorizationReuse
    @test solve(LinearProblem(adjoint(dense(J, 0.4)), b), LHLFactorization()).u ≈
        adjoint(dense(J, 0.4)) \ b rtol = 1.0e-9
end
