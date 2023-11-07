using Test
using ForwardDiff
using LinearSolve
using FiniteDiff

n = 4
A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);
for alg in (
        LUFactorization(), 
        # RFLUFactorization(),
        # KrylovJL_GMRES(),
    )
    alg_str = string(alg)
    @show alg_str
    function fb(b)
        prob = LinearProblem(A, b)

        sol1 = solve(prob, alg)

        sum(sol1.u)
    end
    fb(b1)

    fid_jac = FiniteDiff.finite_difference_jacobian(fb, b1) |> vec
    @show fid_jac

    fod_jac = ForwardDiff.gradient(fb, b1) |> vec
    @show fod_jac

    @test fod_jac ≈ fid_jac rtol=1e-6

    function fA(A)
        prob = LinearProblem(A, b1)

        sol1 = solve(prob, alg)

        sum(sol1.u)
    end
    fA(A)

    fid_jac = FiniteDiff.finite_difference_jacobian(fA, A) |> vec
    @show fid_jac

    # @test_throws MethodError fod_jac = ForwardDiff.gradient(fA, A) |> vec 
    fod_jac = ForwardDiff.gradient(fA, A) |> vec 
    # @show fod_jac

    # @test fod_jac ≈ fid_jac rtol=1e-6
end