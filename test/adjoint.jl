using Zygote, ForwardDiff
using LinearSolve, LinearAlgebra, Test
using FiniteDiff, RecursiveFactorization
using LazyArrays: BroadcastArray

n = 4
A = rand(n, n);
b1 = rand(n);

function f(A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)

    sol1 = solve(prob, alg)

    s1 = sol1.u
    norm(s1)
end

f(A, b1) # Uses BLAS

dA, db1 = Zygote.gradient(f, A, b1)
@test dA isa BroadcastArray

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12

A = rand(n, n);
b1 = rand(n);

_ff = (x, y) -> f(x,
    y;
    alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization))
_ff(copy(A), copy(b1))

dA, db1 = Zygote.gradient(_ff, copy(A), copy(b1))
@test dA isa BroadcastArray

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12

function f3(A, b1, b2; alg = KrylovJL_GMRES())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg)
    prob = LinearProblem(A, b2)
    sol2 = solve(prob, alg)
    norm(sol1.u .+ sol2.u)
end

dA, db1, db2 = Zygote.gradient(f3, A, b1, b1)
@test dA isa BroadcastArray

dA2 = FiniteDiff.finite_difference_gradient(
    x -> f3(x, eltype(x).(b1), eltype(x).(b1)), copy(A))
db12 = FiniteDiff.finite_difference_gradient(
    x -> f3(eltype(x).(A), x, eltype(x).(b1)), copy(b1))
db22 = FiniteDiff.finite_difference_gradient(
    x -> f3(eltype(x).(A), eltype(x).(b1), x), copy(b1))

@test dA≈dA2 rtol=1e-3
@test db1 ≈ db12
@test db2 ≈ db22

function f4(A, b1, b2; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg; sensealg = LinearSolveAdjoint(; linsolve = KrylovJL_LSMR()))
    prob = LinearProblem(A, b2)
    sol2 = solve(prob, alg; sensealg = LinearSolveAdjoint(; linsolve = KrylovJL_GMRES()))
    norm(sol1.u .+ sol2.u)
end

dA, db1, db2 = Zygote.gradient(f4, A, b1, b1)
@test dA isa BroadcastArray

dA2 = ForwardDiff.gradient(x -> f4(x, eltype(x).(b1), eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f4(eltype(x).(A), x, eltype(x).(b1)), copy(b1))
db22 = ForwardDiff.gradient(x -> f4(eltype(x).(A), eltype(x).(b1), x), copy(b1))

@test dA≈dA2 atol=5e-5
@test db1 ≈ db12
@test db2 ≈ db22

A = rand(n, n);
b1 = rand(n);
for alg in (
    LUFactorization(),
    RFLUFactorization(),
    KrylovJL_GMRES()
)
    @show alg
    function fb(b)
        prob = LinearProblem(A, b)

        sol1 = solve(prob, alg)

        sum(sol1.u)
    end
    fb(b1)

    fd_jac = FiniteDiff.finite_difference_jacobian(fb, b1) |> vec
    @show fd_jac

    zyg_jac = Zygote.jacobian(fb, b1) |> first |> vec
    @show zyg_jac

    @test zyg_jac≈fd_jac rtol=1e-4

    function fA(A)
        prob = LinearProblem(A, b1)

        sol1 = solve(prob, alg)

        sum(sol1.u)
    end
    fA(A)

    fd_jac = FiniteDiff.finite_difference_jacobian(fA, A) |> vec
    @show fd_jac

    zyg_jac = Zygote.jacobian(fA, A) |> first |> vec
    @show zyg_jac

    @test zyg_jac≈fd_jac rtol=1e-4
end
