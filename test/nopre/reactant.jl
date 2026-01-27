using Reactant, Enzyme
using LinearSolve, LinearAlgebra, Test
using FiniteDiff, RecursiveFactorization

# Test Reactant AD for linear solves
# This mirrors the enzyme.jl tests but uses Reactant's @jit compilation

n = 4
A = rand(n, n)
b1 = rand(n)

function f(A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg)
    s1 = sol1.u
    return norm(s1)
end

# First, verify the function works outside of Reactant
@test f(A, b1) > 0

# Test with Reactant compiled function using Enzyme.gradient via @jit
@testset "Reactant AD with LUFactorization" begin
    # Convert arrays to Reactant arrays
    A_r = Reactant.to_rarray(A)
    b_r = Reactant.to_rarray(b1)

    # Test forward mode
    function f_A(A)
        prob = LinearProblem(A, b1)
        sol = solve(prob, LUFactorization())
        return sum(sol.u)
    end

    function f_b(b)
        prob = LinearProblem(A, b)
        sol = solve(prob, LUFactorization())
        return sum(sol.u)
    end

    # Compute gradients using Reactant + Enzyme
    grad_A_reactant = @jit Enzyme.gradient(Reverse, f_A, A_r)
    grad_b_reactant = @jit Enzyme.gradient(Reverse, f_b, b_r)

    # Compare with FiniteDiff
    grad_A_fd = FiniteDiff.finite_difference_gradient(f_A, A)
    grad_b_fd = FiniteDiff.finite_difference_gradient(f_b, b1)

    @test grad_A_reactant ≈ grad_A_fd rtol = 1e-4
    @test grad_b_reactant ≈ grad_b_fd rtol = 1e-4
end

@testset "Reactant AD with DefaultLinearSolver LUFactorization" begin
    A_r = Reactant.to_rarray(A)
    b_r = Reactant.to_rarray(b1)

    function f_default(A, b)
        alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization)
        prob = LinearProblem(A, b)
        sol = solve(prob, alg)
        return sum(sol.u)
    end

    # Test that it works with default solver
    grad_A = @jit Enzyme.gradient(Reverse, x -> f_default(x, b1), A_r)
    grad_b = @jit Enzyme.gradient(Reverse, x -> f_default(A, x), b_r)

    # Compare with FiniteDiff
    grad_A_fd = FiniteDiff.finite_difference_gradient(x -> f_default(x, b1), A)
    grad_b_fd = FiniteDiff.finite_difference_gradient(x -> f_default(A, x), b1)

    @test grad_A ≈ grad_A_fd rtol = 1e-4
    @test grad_b ≈ grad_b_fd rtol = 1e-4
end

@testset "Reactant AD with RFLUFactorization" begin
    A_r = Reactant.to_rarray(A)
    b_r = Reactant.to_rarray(b1)

    function f_rf(A, b)
        prob = LinearProblem(A, b)
        sol = solve(prob, RFLUFactorization())
        return sum(sol.u)
    end

    grad_A = @jit Enzyme.gradient(Reverse, x -> f_rf(x, b1), A_r)
    grad_b = @jit Enzyme.gradient(Reverse, x -> f_rf(A, x), b_r)

    grad_A_fd = FiniteDiff.finite_difference_gradient(x -> f_rf(x, b1), A)
    grad_b_fd = FiniteDiff.finite_difference_gradient(x -> f_rf(A, x), b1)

    @test grad_A ≈ grad_A_fd rtol = 1e-4
    @test grad_b ≈ grad_b_fd rtol = 1e-4
end

@testset "Reactant AD with cache reuse" begin
    b2 = rand(n)
    A_r = Reactant.to_rarray(A)
    b1_r = Reactant.to_rarray(b1)
    b2_r = Reactant.to_rarray(b2)

    function f_cache(A, b1, b2; alg = LUFactorization())
        prob = LinearProblem(A, b1)
        cache = init(prob, alg)
        s1 = copy(solve!(cache).u)
        cache.b = b2
        s2 = solve!(cache).u
        return norm(s1 + s2)
    end

    # Verify function works
    @test f_cache(A, b1, b2) > 0

    # Test gradients with Reactant
    grad_A = @jit Enzyme.gradient(Reverse, x -> f_cache(x, b1, b2), A_r)
    grad_b1 = @jit Enzyme.gradient(Reverse, x -> f_cache(A, x, b2), b1_r)
    grad_b2 = @jit Enzyme.gradient(Reverse, x -> f_cache(A, b1, x), b2_r)

    # Compare with FiniteDiff
    grad_A_fd = FiniteDiff.finite_difference_gradient(x -> f_cache(x, b1, b2), A)
    grad_b1_fd = FiniteDiff.finite_difference_gradient(x -> f_cache(A, x, b2), b1)
    grad_b2_fd = FiniteDiff.finite_difference_gradient(x -> f_cache(A, b1, x), b2)

    @test grad_A ≈ grad_A_fd rtol = 1e-4
    @test grad_b1 ≈ grad_b1_fd rtol = 1e-4
    @test grad_b2 ≈ grad_b2_fd rtol = 1e-4
end

@testset "Reactant AD Forward mode" begin
    A_r = Reactant.to_rarray(A)
    b_r = Reactant.to_rarray(b1)

    function fnice(A, b, alg)
        prob = LinearProblem(A, b)
        sol = solve(prob, alg)
        return sum(sol.u)
    end

    for alg in (LUFactorization(), RFLUFactorization())
        fb_closure = b -> fnice(A, b, alg)

        # FiniteDiff Jacobian
        fd_jac = FiniteDiff.finite_difference_jacobian(fb_closure, b1) |> vec

        # Reactant forward mode
        function compute_jac_reactant(b_r)
            tangents = [fill!(similar(b_r), 0.0) for _ in 1:n]
            for i in 1:n
                tangents[i][i] = 1.0
            end
            jac = Float64[]
            for t in tangents
                result = Enzyme.autodiff(
                    Forward, fnice,
                    Const(A), Duplicated(b_r, t), Const(alg)
                )
                push!(jac, only(result))
            end
            return jac
        end

        en_jac = @jit compute_jac_reactant(b_r)

        @test en_jac ≈ fd_jac rtol = 1e-4
    end
end

@testset "Reactant AD with OperatorAssumptions" begin
    A_r = Reactant.to_rarray([1.0 2.0; 3.0 4.0])
    b_r = Reactant.to_rarray([1.0, 2.0])
    u_r = Reactant.to_rarray([0.0, 0.0])

    function testls(A, b, u)
        oa = OperatorAssumptions(
            true, condition = LinearSolve.OperatorCondition.WellConditioned
        )
        prob = LinearProblem(A, b)
        linsolve = init(prob, LUFactorization(), assumptions = oa)
        cache = solve!(linsolve)
        return sum(cache.u)
    end

    # Test that it works
    result = @jit testls(A_r, b_r, u_r)
    @test result ≈ testls(Array(A_r), Array(b_r), Array(u_r))

    # Test gradients
    grad_A = @jit Enzyme.gradient(Reverse, x -> testls(x, [1.0, 2.0], [0.0, 0.0]), A_r)
    grad_b = @jit Enzyme.gradient(
        Reverse, x -> testls([1.0 2.0; 3.0 4.0], x, [0.0, 0.0]), b_r
    )

    # Verify gradients are computed (non-zero)
    @test !all(iszero, grad_A)
    @test !all(iszero, grad_b)
end
