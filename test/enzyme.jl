using Enzyme, ForwardDiff
using LinearSolve, LinearAlgebra, Test

n = 4
A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);

function f(A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)

    sol1 = solve(prob, alg)

    s1 = sol1.u
    norm(s1)
end

f(A, b1) # Uses BLAS

Enzyme.autodiff(Reverse, f, Duplicated(copy(A), dA), Duplicated(copy(b1), db1))

dA2 = ForwardDiff.gradient(x->f(x,eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x->f(eltype(x).(A),x), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12

A = rand(n, n);
dA = zeros(n, n);
dA2 = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);
db12 = zeros(n);

function fbatch(y, A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)

    sol1 = solve(prob, alg)

    s1 = sol1.u
    y[1] = norm(s1)
    nothing
end

y = [0.0]
dy1 = [1.0]
dy2 = [1.0]
Enzyme.autodiff(Reverse, fbatch, BatchDuplicated(y, (dy1, dy2)), BatchDuplicated(copy(A), (dA, dA2)), BatchDuplicated(copy(b1), (db1, db12)))

dA2 = ForwardDiff.gradient(x->f(x,eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x->f(eltype(x).(A),x), copy(b1))

@test_broken dA ≈ dA_2
@test_broken dA2 ≈ dA_2
@test_broken db1 ≈ db1_2
@test_broken db12 ≈ db1_2

function f(A, b1, b2; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    s1 = copy(solve!(cache).u)
    cache.b = b2
    s2 = solve!(cache).u
    norm(s1 + s2)
end

A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);
b2 = rand(n);
db2 = zeros(n);

f(A, b1, b2)
Enzyme.autodiff(Reverse, f, Duplicated(copy(A), dA), Duplicated(copy(b1), db1), Duplicated(copy(b2), db2))

dA2 = ForwardDiff.gradient(x->f(x,eltype(x).(b1),eltype(x).(b2)), copy(A))
db12 = ForwardDiff.gradient(x->f(eltype(x).(A),x,eltype(x).(b2)), copy(b1))
db22 = ForwardDiff.gradient(x->f(eltype(x).(A),eltype(x).(b1),x), copy(b2))

@test dA ≈ dA2
@test db1 ≈ db12
@test db2 ≈ db22

function f2(A, b1, b2; alg = RFLUFactorization())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    s1 = copy(solve!(cache).u)
    cache.b = b2
    s2 = solve!(cache).u
    norm(s1 + s2)
end

f2(A, b1, b2)
dA = zeros(n, n);
db1 = zeros(n);
db2 = zeros(n);
Enzyme.autodiff(Reverse, f2, Duplicated(copy(A), dA), Duplicated(copy(b1), db1), Duplicated(copy(b2), db2))

@test dA ≈ dA2
@test db1 ≈ db12
@test db2 ≈ db22

function f3(A, b1, b2; alg = KrylovJL_GMRES())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    s1 = copy(solve!(cache).u)
    cache.b = b2
    s2 = solve!(cache).u
    norm(s1 + s2)
end

Enzyme.autodiff(Reverse, f3, Duplicated(copy(A), dA), Duplicated(copy(b1), db1), Duplicated(copy(b2), db2))

@test dA ≈ dA2 atol=5e-5
@test db1 ≈ db12
@test db2 ≈ db22