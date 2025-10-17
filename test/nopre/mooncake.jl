using ForwardDiff
using LinearSolve, LinearAlgebra, Test
using FiniteDiff, RecursiveFactorization
using LazyArrays: BroadcastArray
using Mooncake

# first test
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

cache = prepare_gradient_cache(f, (copy(A), copy(b1))...)
value, gradient = Mooncake.value_and_gradient!!(cache, f, (copy(A), copy(b1))...)

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

# Mooncake
@test value ≈ f_primal
@test gradient[2] ≈ dA2
@test gradient[3] ≈ db12

# Second test
A = rand(n, n);
b1 = rand(n);

_ff = (x,
    y) -> f(x,
    y;
    alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization))
f_primal = _ff(copy(A), copy(b1))

cache = prepare_gradient_cache(_ff, (copy(A), copy(b1))...)
value, gradient = Mooncake.value_and_gradient!!(cache, _ff, (copy(A), copy(b1))...)

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

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

# Mooncake needs atomic Complex Number tangents instead of NamedTuples.
# cache = Mooncake.prepare_gradient_cache(f3, (copy(A), copy(b1), copy(b1))...)
# results = Mooncake.value_and_gradient!!(cache, f3, (copy(A), copy(b1), copy(b1))...)

# dA2 = FiniteDiff.finite_difference_gradient(
#     x -> f3(x, eltype(x).(b1), eltype(x).(b1)), copy(A))
# db12 = FiniteDiff.finite_difference_gradient(
#     x -> f3(eltype(x).(A), x, eltype(x).(b1)), copy(b1))
# db22 = FiniteDiff.finite_difference_gradient(
#     x -> f3(eltype(x).(A), eltype(x).(b1), x), copy(b1))

# @test f3(A, b1, b1) ≈ results[1]
# @test dA2 ≈ results[2][2]
# @test db12 ≈ results[2][3]
# @test db22 ≈ results[2][4]

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

cache = Mooncake.prepare_gradient_cache(f4, (copy(A), copy(b1), copy(b1))...)
results = Mooncake.value_and_gradient!!(cache, f4, (copy(A), copy(b1), copy(b1))...)

dA2 = ForwardDiff.gradient(x -> f4(x, eltype(x).(b1), eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f4(eltype(x).(A), x, eltype(x).(b1)), copy(b1))
db22 = ForwardDiff.gradient(x -> f4(eltype(x).(A), eltype(x).(b1), x), copy(b1))

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

    cache = Mooncake.prepare_gradient_cache(fb, copy(b1))
    results = Mooncake.value_and_gradient!!(cache, fb, copy(b1))
    @show results

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

    cache = Mooncake.prepare_gradient_cache(fA, copy(A))
    results = Mooncake.value_and_gradient!!(cache, fA, copy(A))
    @show results
    mooncake_gradient = results[2][2] |> vec

    @test results[1] ≈ fA(A)
    @test mooncake_gradient ≈ fd_jac rtol = 1e-5
end

# Tests for solve! and init rrules.

n = 4
A = rand(n, n);
b1 = rand(n);
b2 = rand(n);

function f(A, b1, b2; alg=LUFactorization())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    s1 = copy(solve!(cache).u)
    cache.b = b2
    s2 = solve!(cache).u
    norm(s1 + s2)
end

f_primal = f(copy(A), copy(b1), copy(b2))
value, gradient = Mooncake.value_and_gradient!!(
    prepare_gradient_cache(f, copy(A), copy(b1), copy(b2)),
    f, copy(A), copy(b1), copy(b2)
)

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1), eltype(x).(b2)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x, eltype(x).(b2)), copy(b1))
db22 = ForwardDiff.gradient(x -> f(eltype(x).(A), eltype(x).(b1), x), copy(b2))

@test value == f_primal
@test gradient[2] ≈ dA2
@test gradient[3] ≈ db12
@test gradient[4] ≈ db22

function f2(A, b1, b2; alg=RFLUFactorization())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    s1 = copy(solve!(cache).u)
    cache.b = b2
    s2 = solve!(cache).u
    norm(s1 + s2)
end

f_primal = f2(copy(A), copy(b1), copy(b2))
value, gradient = Mooncake.value_and_gradient!!(
    prepare_gradient_cache(f2, copy(A), copy(b1), copy(b2)),
    f2, copy(A), copy(b1), copy(b2)
)

@test value == f_primal
@test gradient[2] ≈ dA2
@test gradient[3] ≈ db12
@test gradient[4] ≈ db22

function f3(A, b1, b2; alg=LUFactorization())
    # alg = KrylovJL_GMRES())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    s1 = copy(solve!(cache).u)
    cache.b = b2
    s2 = solve!(cache).u
    norm(s1 + s2)
end

f_primal = f3(copy(A), copy(b1), copy(b2))
value, gradient = Mooncake.value_and_gradient!!(
    prepare_gradient_cache(f3, copy(A), copy(b1), copy(b2)),
    f3, copy(A), copy(b1), copy(b2)
)

@test value == f_primal
@test gradient[2] ≈ dA2 atol = 5e-5
@test gradient[3] ≈ db12
@test gradient[4] ≈ db22

A = rand(n, n);
b1 = rand(n);

function fnice(A, b, alg)
    prob = LinearProblem(A, b)
    sol1 = solve(prob, alg)
    return sum(sol1.u)
end

@testset for alg in (
    LUFactorization(),
    RFLUFactorization(),
    KrylovJL_GMRES()
)
    # for B
    fb_closure = b -> fnice(A, b, alg)
    fd_jac_b = FiniteDiff.finite_difference_jacobian(fb_closure, b1) |> vec

    val, en_jac = Mooncake.value_and_gradient!!(
        prepare_gradient_cache(fnice, copy(A), copy(b1), alg),
        fnice, copy(A), copy(b1), alg
    )
    @test en_jac[3] ≈ fd_jac_b rtol = 1e-5

    # For A
    fA_closure = A -> fnice(A, b1, alg)
    fd_jac_A = FiniteDiff.finite_difference_jacobian(fA_closure, A) |> vec
    A_grad = en_jac[2] |> vec
    @test A_grad ≈ fd_jac_A rtol = 1e-5
end

# The below test function cases fails !
# AVOID Adjoint case in code as : `solve!(cache); s1 = copy(cache.u)`.
# Instead stick to code like : `sol = solve!(cache); s1 = copy(sol.u)`.

function f4(A, b1, b2; alg=LUFactorization())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    solve!(cache)
    s1 = copy(cache.u)
    cache.b = b2
    solve!(cache)
    s2 = copy(cache.u)
    norm(s1 + s2)
end

# value, grad = Mooncake.value_and_gradient!!(
# prepare_gradient_cache(f4, copy(A), copy(b1), copy(b2)),
# f4, copy(A), copy(b1), copy(b2)
# )
# (0.0, (Mooncake.NoTangent(), [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]))

# dA2 = ForwardDiff.gradient(x -> f4(x, eltype(x).(b1), eltype(x).(b2)), copy(A))
# db12 = ForwardDiff.gradient(x -> f4(eltype(x).(A), x, eltype(x).(b2)), copy(b1))
# db22 = ForwardDiff.gradient(x -> f4(eltype(x).(A), eltype(x).(b1), x), copy(b2))

# @test value == f_primal
# @test grad[2] ≈ dA2
# @test grad[3] ≈ db12
# @test grad[4] ≈ db22

function testls(A, b, u)
    oa = OperatorAssumptions(
        true, condition=LinearSolve.OperatorCondition.WellConditioned)
    prob = LinearProblem(A, b)
    linsolve = init(prob, LUFactorization(), assumptions=oa)
    cache = solve!(linsolve)
    sum(cache.u)
end

# A = [1.0 2.0; 3.0 4.0]
# b = [1.0, 2.0]
# u = zero(b)
# value, gradient = Mooncake.value_and_gradient!!(
#     prepare_gradient_cache(testls, copy(A), copy(b), copy(u)),
#     testls, copy(A), copy(b), copy(u)
# )

# dA = gradient[2]
# db = gradient[3]
# du = gradient[4]

function testls(A, b, u)
    oa = OperatorAssumptions(
        true, condition=LinearSolve.OperatorCondition.WellConditioned)
    prob = LinearProblem(A, b)
    linsolve = init(prob, LUFactorization(), assumptions=oa)
    solve!(linsolve)
    sum(linsolve.u)
end

# value, gradient = Mooncake.value_and_gradient!!(
#     prepare_gradient_cache(testls, copy(A), copy(b), copy(u)),
#     testls, copy(A), copy(b), copy(u)
# )

# dA2 = gradient[2]
# db2 = gradient[3]
# du2 = gradient[4]

# @test dA == dA2
# @test db == db2
# @test du == du2
