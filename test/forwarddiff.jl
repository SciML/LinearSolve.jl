using Test
using ForwardDiff
using LinearSolve
using FiniteDiff
using Enzyme
using Random
Random.seed!(1234)

n = 4
A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);
for alg in (
        LUFactorization(), 
        RFLUFactorization(),
        # KrylovJL_GMRES(), dispatch fails
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

    fod_jac = ForwardDiff.gradient(fA, A) |> vec 
    @show fod_jac

    @test fod_jac ≈ fid_jac rtol=1e-6


    function fAb(Ab)
        A = Ab[:, 1:n]
        b1 = Ab[:, n+1]
        prob = LinearProblem(A, b1)

        sol1 = solve(prob, alg)

        sum(sol1.u)
    end
    fAb(hcat(A, b1))

    fid_jac = FiniteDiff.finite_difference_jacobian(fAb, hcat(A, b1)) |> vec
    @show fid_jac

    fod_jac = ForwardDiff.gradient(fAb, hcat(A, b1)) |> vec 
    @show fod_jac

    @test fod_jac ≈ fid_jac rtol=1e-6

end