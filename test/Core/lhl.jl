using LinearSolve, LinearAlgebra, Random, SciMLOperators, Test
using SciMLOperators: WOperator, jacobian_stale

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
    ws = cache.cacheval.ws
    @test ws.reduced
    # The solve claimed the operator by clearing the flag it was constructed with.
    @test !jacobian_stale(W)

    # Without the announcement an in-place write to J is invisible and the stale
    # reduction is reused; with it, the next solve reduces again and re-clears.
    J .= randn(MersenneTwister(9), n, n)
    stale = copy(solve!(cache).u)
    @test !isapprox(stale, dense(J, 0.1) \ b, rtol = 1.0e-6)
    mark_jacobian_updated!(W)
    @test jacobian_stale(W)
    @test copy(solve!(cache).u) ≈ dense(J, 0.1) \ b rtol = 1.0e-9
    @test !jacobian_stale(W)
end

@testset "thread option" begin
    # Threading the reduction is deterministic: bit-identical for any thread count, so the
    # option can never change an answer.
    n = 200
    J = randn(MersenneTwister(41), n, n)
    b = randn(MersenneTwister(42), n)
    ref = dense(J, 0.1) \ b
    us = [
        solve(LinearProblem(wop(J, 0.1), b), LHLFactorization(; thread)).u
            for thread in (Val(true), Val(false), true, false)
    ]
    for u in us
        @test u ≈ ref rtol = 1.0e-9
        @test u == us[1]
    end
    @test typeof(LHLFactorization()) === LHLFactorization{true}
    @test typeof(LHLFactorization(thread = false)) === LHLFactorization{false}
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

@testset "complex shift on a real Jacobian" begin
    # RadauIIA's shape: `J` stays real and only the shift goes complex, so the expensive
    # reduction is real and one of them can serve a complex γ.
    n = 40
    J = randn(MersenneTwister(31), n, n)
    b = randn(MersenneTwister(32), ComplexF64, n)
    γc = 0.03 + 0.02im
    W = WOperator{true}(I, γc, J, zeros(ComplexF64, n))
    for refine in (0, 1)
        u = solve(LinearProblem(W, b), LHLFactorization(; refine)).u
        @test u ≈ dense(J, γc) \ b rtol = 1.0e-9
    end

    cache = init(LinearProblem(W, b), LHLFactorization())
    solve!(cache)
    ws = cache.cacheval.ws
    @test eltype(ws.factors) === Float64        # the reduction stays real
    @test typeof(ws.σ) === ComplexF64           # only the shift is complex
    for γ in (0.05 - 0.01im, 0.2 + 0.4im)
        update_gamma!(cache, γ)
        @test copy(solve!(cache).u) ≈ dense(J, γ) \ b rtol = 1.0e-9
    end
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
    # Selected through the default polyalgorithm, so `defaultalg`'s return type does not
    # depend on the runtime size/mass-matrix checks.
    @test LinearSolve.defaultalg(wop(J200, 0.1), b200, LinearSolve.OperatorAssumptions(true)) ==
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LHLFactorization)
    s = solve(LinearProblem(wop(J200, 0.1), b200))
    @test s.u ≈ dense(J200, 0.1) \ b200 rtol = 1.0e-9

    # Too small to pay for the reduction, or a general mass matrix: both keep whatever
    # the pre-existing operator path chose, rather than being intercepted.
    nsmall = LinearSolve.LHL_DEFAULT_MIN_SIZE - 1
    b20 = randn(MersenneTwister(4), nsmall)
    J20 = randn(MersenneTwister(3), nsmall, nsmall)
    small = LinearSolve.defaultalg(wop(J20, 0.1), b20, LinearSolve.OperatorAssumptions(true))
    @test small != LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LHLFactorization)
    @test small == @invoke LinearSolve.defaultalg(
        wop(J20, 0.1)::SciMLOperators.AbstractSciMLOperator, b20,
        LinearSolve.OperatorAssumptions(true)
    )
    # and one element above the cutoff it is selected
    nbig = LinearSolve.LHL_DEFAULT_MIN_SIZE
    @test LinearSolve.defaultalg(
        wop(randn(MersenneTwister(7), nbig, nbig), 0.1), randn(MersenneTwister(8), nbig),
        LinearSolve.OperatorAssumptions(true)
    ) == LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LHLFactorization)
    Wmm = WOperator{true}(Diagonal(collect(1.0:200)), 0.1, J200, zeros(200))
    @test LinearSolve.defaultalg(Wmm, b200, LinearSolve.OperatorAssumptions(true)) !=
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LHLFactorization)

    # A plain matrix must not pay for the LHL workspace it will never use.
    cm = init(LinearProblem(randn(MersenneTwister(9), 200, 200), b200))
    @test cm.cacheval.LHLFactorization === nothing
end

@testset "a different Jacobian is never served a stale reduction" begin
    # Regression: keyed on the staleness flag alone, swapping in a *different* WOperator
    # whose flag some other consumer had already cleared reused the old reduction and
    # returned a confidently wrong answer (rel. error 0.89).
    n = 60
    J1 = randn(MersenneTwister(1), n, n)
    J2 = randn(MersenneTwister(2), n, n)
    b = randn(MersenneTwister(3), n)
    W1, W2 = wop(J1, 0.1), wop(J2, 0.1)
    cache = init(LinearProblem(W1, b), LHLFactorization(; refine = 0))
    @test copy(solve!(cache).u) ≈ dense(J1, 0.1) \ b rtol = 1.0e-9

    SciMLOperators.mark_jacobian_current!(W2)      # someone else already claimed it
    @test !jacobian_stale(W2)
    LinearSolve.reinit!(cache; A = W2)
    @test copy(solve!(cache).u) ≈ dense(J2, 0.1) \ b rtol = 1.0e-9

    # Same object, contents rewritten, flag raised: still reduces.
    J2 .= randn(MersenneTwister(4), n, n)
    mark_jacobian_updated!(W2)
    @test copy(solve!(cache).u) ≈ dense(J2, 0.1) \ b rtol = 1.0e-9

    # A bare matrix has no flag, so `isfresh` is the only signal and must be honoured.
    A1 = randn(MersenneTwister(5), n, n)
    A2 = randn(MersenneTwister(6), n, n)
    c2 = init(LinearProblem(A1, b), LHLFactorization(; refine = 0))
    @test copy(solve!(c2).u) ≈ A1 \ b rtol = 1.0e-9
    LinearSolve.reinit!(c2; A = A2)
    @test copy(solve!(c2).u) ≈ A2 \ b rtol = 1.0e-9
end

@testset "concrete factorizations refuse an externally-maintained WOperator" begin
    # `convert(AbstractMatrix, ·)` hands back the owner-maintained `_concrete_form` for an
    # in-place WOperator over a plain matrix, so it is stale after any gamma change. Better
    # a loud error than a factorization of the wrong matrix.
    n = 40
    J = randn(MersenneTwister(15), n, n)
    b = randn(MersenneTwister(16), n)
    @test_throws ArgumentError solve(LinearProblem(wop(J, 0.1), b), LUFactorization())
    # The guard sits on the factorization, not on cache construction: the default
    # algorithm builds a cacheval for every slot, so throwing at init would break
    # problems that never reach an LU.
    @test init(LinearProblem(wop(J, 0.1), b), LUFactorization()) isa LinearSolve.LinearCache
    # An operator-backed Jacobian does rebuild on conversion, so it is fine.
    Wop = WOperator{true}(I, 0.1, SciMLOperators.MatrixOperator(J), zeros(n))
    @test solve(LinearProblem(Wop, b), LUFactorization()).u ≈ dense(J, 0.1) \ b rtol = 1.0e-9
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
