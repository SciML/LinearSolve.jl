using Zygote, ForwardDiff
using LinearSolve, LinearAlgebra, Test
using FiniteDiff, RecursiveFactorization
using LazyArrays: BroadcastArray
using Mooncake

# first test
# zygote
n = 4
A = rand(n, n);
b1 = rand(n);

function f(A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)

    sol1 = solve(prob, alg)

    s1 = sol1.u
    norm(s1)
end

f_primal = f(A, b1) # Uses BLAS

dA, db1 = Zygote.gradient(f, A, b1)
@test dA isa BroadcastArray

cache = prepare_gradient_cache(f, (copy(A), copy(b1))...)
value, gradient = Mooncake.value_and_gradient!!(cache, f, (copy(A), copy(b1))...)

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

# Zygote
@test dA ≈ dA2
@test db1 ≈ db12

# Mooncake
@test value ≈ f_primal
@test gradient[2] ≈ dA2
@test gradient[3] ≈ db12

# Second test
# zygote
A = rand(n, n);
b1 = rand(n);

_ff = (x,
    y) -> f(x,
    y;
    alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization))
f_primal = _ff(copy(A), copy(b1))

dA, db1 = Zygote.gradient(_ff, copy(A), copy(b1))
@test dA isa BroadcastArray

cache = prepare_gradient_cache(_ff, (copy(A), copy(b1))...)
value, gradient = Mooncake.value_and_gradient!!(cache, _ff, (copy(A), copy(b1))...)

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

# Zygote
@test dA ≈ dA2
@test db1 ≈ db12

# Mooncake
@test value ≈ f_primal
@test gradient[2] ≈ dA2
@test gradient[3] ≈ db12

# third test
# Test complex numbers
A = rand(n, n) + 1im * rand(n, n);
b1 = rand(n) + 1im * rand(n);

function f3(A, b1, b2; alg = KrylovJL_GMRES())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg)
    prob = LinearProblem(A, b2)
    sol2 = solve(prob, alg)
    norm(sol1.u .+ sol2.u)
end

dA, db1, db2 = Zygote.gradient(f3, A, b1, b1)
@test dA isa BroadcastArray

# Mooncake needs atomic Complex Number tangents instead of NamedTuples.
# cache = Mooncake.prepare_gradient_cache(f3, (copy(A), copy(b1), copy(b1))...)
# results = Mooncake.value_and_gradient!!(cache, f3, (copy(A), copy(b1), copy(b1))...)

# @test f3(A, b1, b1) ≈ results[1]
# @test dA2 ≈ results[2][2]
# @test db12 ≈ results[2][3]
# @test db22 ≈ results[2][4]

dA2 = FiniteDiff.finite_difference_gradient(
    x -> f3(x, eltype(x).(b1), eltype(x).(b1)), copy(A))
db12 = FiniteDiff.finite_difference_gradient(
    x -> f3(eltype(x).(A), x, eltype(x).(b1)), copy(b1))
db22 = FiniteDiff.finite_difference_gradient(
    x -> f3(eltype(x).(A), eltype(x).(b1), x), copy(b1))

@test dA≈dA2 rtol=1e-3
@test db1 ≈ db12
@test db2 ≈ db22

# fourth test
A = rand(n, n);
b1 = rand(n);

function f4(A, b1, b2; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg; sensealg = LinearSolveAdjoint(; linsolve = KrylovJL_LSMR()))
    prob = LinearProblem(A, b2)
    sol2 = solve(prob, alg; sensealg = LinearSolveAdjoint(; linsolve = KrylovJL_GMRES()))
    norm(sol1.u .+ sol2.u)
end

dA, db1, db2 = Zygote.gradient(f4, A, b1, b1)
@test dA isa BroadcastArray

cache = Mooncake.prepare_gradient_cache(f4, (copy(A), copy(b1), copy(b1))...)
results = Mooncake.value_and_gradient!!(cache, f4, (copy(A), copy(b1), copy(b1))...)

dA2 = ForwardDiff.gradient(x -> f4(x, eltype(x).(b1), eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f4(eltype(x).(A), x, eltype(x).(b1)), copy(b1))
db22 = ForwardDiff.gradient(x -> f4(eltype(x).(A), eltype(x).(b1), x), copy(b1))

@test dA ≈ dA2 atol = 5e-5
@test db1 ≈ db12
@test db2 ≈ db22

@test f4(A, b1, b1) ≈ results[1]
@test dA2 ≈ results[2][2]
@test db12 ≈ results[2][3]
@test db22 ≈ results[2][4]

# fifth test
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

    cache = Mooncake.prepare_gradient_cache(fb, copy(b1))
    results = Mooncake.value_and_gradient!!(cache, fb, copy(b1))
    @show results

    @test zyg_jac ≈ fd_jac rtol = 1e-4
    @test results[1] ≈ fb(b1)
    @test results[2][2] ≈ fd_jac rtol = 1e-5

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

    cache = Mooncake.prepare_gradient_cache(fA, copy(A))
    results = Mooncake.value_and_gradient!!(cache, fA, copy(A))
    @show results
    mooncake_gradient = results[2][2] |> vec

    @test zyg_jac ≈ fd_jac rtol = 1e-4
    @test results[1] ≈ fA(A)
    @test mooncake_gradient ≈ fd_jac rtol = 1e-5
end
