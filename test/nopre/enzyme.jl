using Enzyme, ForwardDiff
using LinearSolve, LinearAlgebra, Test
using FiniteDiff, RecursiveFactorization

n = 4
A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);

function f(A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)

    sol1 = solve(prob, alg)

    s1 = sol1.u
    return norm(s1)
end

f(A, b1) # Uses BLAS

Enzyme.autodiff(Reverse, f, Duplicated(copy(A), dA), Duplicated(copy(b1), db1))

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12

A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);

_ff = (
    x,
    y,
) -> f(
    x,
    y;
    alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization)
)
_ff(copy(A), copy(b1))

Enzyme.autodiff(
    Reverse,
    (
        x,
        y,
    ) -> f(
        x,
        y;
        alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization)
    ),
    Duplicated(copy(A), dA),
    Duplicated(copy(b1), db1)
)

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12

A = rand(n, n);
dA = zeros(n, n);
dA2 = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);
db12 = zeros(n);

# Batch test
n = 4
A = rand(n, n);
dA = zeros(n, n);
dA2 = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);
db12 = zeros(n);

function f(A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg)
    s1 = sol1.u
    return norm(s1)
end

function fbatch(y, A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg)
    s1 = sol1.u
    y[1] = norm(s1)
    return nothing
end

y = [0.0]
dy1 = [1.0]
dy2 = [1.0]
Enzyme.autodiff(
    Reverse, fbatch, Duplicated(y, dy1), Duplicated(copy(A), dA), Duplicated(copy(b1), db1)
)

@test y[1] ≈ f(copy(A), b1)
dA_2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db1_2 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

@test dA ≈ dA_2
@test db1 ≈ db1_2

y .= 0
dy1 .= 1
dy2 .= 1
dA .= 0
dA2 .= 0
db1 .= 0
db12 .= 0
Enzyme.autodiff(
    Reverse, fbatch, BatchDuplicated(y, (dy1, dy2)),
    BatchDuplicated(copy(A), (dA, dA2)), BatchDuplicated(copy(b1), (db1, db12))
)

@test dA ≈ dA_2
@test db1 ≈ db1_2
@test dA2 ≈ dA_2
@test db12 ≈ db1_2

function f(A, b1, b2; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    s1 = copy(solve!(cache).u)
    cache.b = b2
    s2 = solve!(cache).u
    return norm(s1 + s2)
end

A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);
b2 = rand(n);
db2 = zeros(n);

f(A, b1, b2)
Enzyme.autodiff(
    Reverse, f, Duplicated(copy(A), dA),
    Duplicated(copy(b1), db1), Duplicated(copy(b2), db2)
)

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1), eltype(x).(b2)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x, eltype(x).(b2)), copy(b1))
db22 = ForwardDiff.gradient(x -> f(eltype(x).(A), eltype(x).(b1), x), copy(b2))

@test dA ≈ dA2
@test db1 ≈ db12
@test db2 ≈ db22

function f2(A, b1, b2; alg = RFLUFactorization())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    s1 = copy(solve!(cache).u)
    cache.b = b2
    s2 = solve!(cache).u
    return norm(s1 + s2)
end

f2(A, b1, b2)
dA = zeros(n, n);
db1 = zeros(n);
db2 = zeros(n);
Enzyme.autodiff(
    Reverse, f2, Duplicated(copy(A), dA),
    Duplicated(copy(b1), db1), Duplicated(copy(b2), db2)
)

@test dA ≈ dA2
@test db1 ≈ db12
@test db2 ≈ db22

function f3(A, b1, b2; alg = KrylovJL_GMRES())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    s1 = copy(solve!(cache).u)
    cache.b = b2
    s2 = solve!(cache).u
    return norm(s1 + s2)
end

dA = zeros(n, n);
db1 = zeros(n);
db2 = zeros(n);
Enzyme.autodiff(
    set_runtime_activity(Reverse), f3, Duplicated(copy(A), dA),
    Duplicated(copy(b1), db1), Duplicated(copy(b2), db2)
)

@test dA ≈ dA2 atol = 5.0e-5
@test db1 ≈ db12
@test db2 ≈ db22

function f4(A, b1, b2; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    cache = init(prob, alg)
    solve!(cache)
    s1 = copy(cache.u)
    cache.b = b2
    solve!(cache)
    s2 = copy(cache.u)
    return norm(s1 + s2)
end

A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);
db1 = zeros(n);
b2 = rand(n);
db2 = zeros(n);

f4(A, b1, b2)
@test_throws "Adjoint case currently not handled" Enzyme.autodiff(
    Reverse, f4, Duplicated(copy(A), dA),
    Duplicated(copy(b1), db1), Duplicated(copy(b2), db2)
)

#=
dA2 = ForwardDiff.gradient(x -> f4(x, eltype(x).(b1), eltype(x).(b2)), copy(A))
db12 = ForwardDiff.gradient(x -> f4(eltype(x).(A), x, eltype(x).(b2)), copy(b1))
db22 = ForwardDiff.gradient(x -> f4(eltype(x).(A), eltype(x).(b1), x), copy(b2))

@test dA ≈ dA2
@test db1 ≈ db12
@test db2 ≈ db22
=#

A = rand(n, n);
dA = zeros(n, n);
b1 = rand(n);

function fnice(A, b, alg)
    prob = LinearProblem(A, b)
    sol1 = solve(prob, alg)
    return sum(sol1.u)
end

@testset for alg in (
        LUFactorization(),
        RFLUFactorization(),    # KrylovJL_GMRES(), fails
    )
    fb_closure = b -> fnice(A, b, alg)

    fd_jac = FiniteDiff.finite_difference_jacobian(fb_closure, b1) |> vec
    @show fd_jac

    en_jac = map(onehot(b1)) do db1
        return only(
            Enzyme.autodiff(
                set_runtime_activity(Forward), fnice,
                Const(A), Duplicated(b1, db1), Const(alg)
            )
        )
    end |> collect
    @show en_jac

    @test en_jac ≈ fd_jac rtol = 1.0e-4

    fA_closure = A -> fnice(A, b1, alg)

    fd_jac = FiniteDiff.finite_difference_jacobian(fA_closure, A) |> vec
    @show fd_jac

    en_jac = map(onehot(A)) do dA
        return only(
            Enzyme.autodiff(
                set_runtime_activity(Forward), fnice,
                Duplicated(A, dA), Const(b1), Const(alg)
            )
        )
    end |> collect
    @show en_jac

    @test en_jac ≈ fd_jac rtol = 1.0e-4
end

# https://github.com/SciML/LinearSolve.jl/issues/479
function testls(A, b, u)
    oa = OperatorAssumptions(
        true, condition = LinearSolve.OperatorCondition.WellConditioned
    )
    prob = LinearProblem(A, b)
    linsolve = init(prob, LUFactorization(), assumptions = oa)
    cache = solve!(linsolve)
    return sum(cache.u)
end

A = [1.0 2.0; 3.0 4.0]
b = [1.0, 2.0]
u = zero(b)
dA = copy(A)
db = copy(b)
du = copy(u)
Enzyme.autodiff(Reverse, testls, Duplicated(A, dA), Duplicated(b, db), Duplicated(u, du))

function testls(A, b, u)
    oa = OperatorAssumptions(
        true, condition = LinearSolve.OperatorCondition.WellConditioned
    )
    prob = LinearProblem(A, b)
    linsolve = init(prob, LUFactorization(), assumptions = oa)
    solve!(linsolve)
    return sum(linsolve.u)
end
A = [1.0 2.0; 3.0 4.0]
b = [1.0, 2.0]
u = zero(b)
dA2 = copy(A)
db2 = copy(b)
du2 = copy(u)
Enzyme.autodiff(Reverse, testls, Duplicated(A, dA2), Duplicated(b, db2), Duplicated(u, du2))

@test dA == dA2
@test db == db2
@test du == du2

# https://github.com/SciML/LinearSolve.jl/issues/929
@testset "Symmetric BunchKaufman reverse" begin

    function bk_solve(b)
        A = Symmetric(Float64[4 1 0.5; 1 3 0.2; 0.5 0.2 2])
        prob = LinearProblem(A, copy(b))
        sol = solve(prob, BunchKaufmanFactorization())
        return sum(sol.u)
    end

    b = Float64[1.0, 2.0, 3.0]
    db = zero(b)

    Enzyme.autodiff(Reverse, Const(bk_solve), Active, Duplicated(copy(b), db))

    A = Symmetric(Float64[4 1 0.5; 1 3 0.2; 0.5 0.2 2])
    expected = A \ ones(3)
    @test db ≈ expected rtol = 1.0e-12 atol = 1.0e-12
end

# https://github.com/SciML/LinearSolve.jl/pull/935 — cover all of LinearAlgebra.jl
# Each test confirms: (1) Enzyme reverse-mode doesn't throw on the wrapper type, and
# (2) the gradient matches a ForwardDiff reference.

@testset "Hermitian reverse" begin
    n = 4
    _A = rand(n, n)
    _A = _A + _A'  # make symmetric
    A_herm = Hermitian(_A)
    b = rand(n)

    function f_herm(b)
        prob = LinearProblem(Hermitian(_A), copy(b))
        sol = solve(prob, CholeskyFactorization())
        return sum(sol.u)
    end

    db_en = zero(b)
    Enzyme.autodiff(Reverse, Const(f_herm), Active, Duplicated(copy(b), db_en))

    db_fd = ForwardDiff.gradient(f_herm, b)
    @test db_en ≈ db_fd rtol = 1.0e-8
end

@testset "UpperTriangular reverse" begin
    n = 4
    _A = triu(rand(n, n) + n * I)  # well-conditioned upper triangular
    A_ut = UpperTriangular(_A)
    b = rand(n)

    function f_ut(b)
        prob = LinearProblem(UpperTriangular(_A), copy(b))
        sol = solve(prob)
        return sum(sol.u)
    end

    db_en = zero(b)
    Enzyme.autodiff(Reverse, Const(f_ut), Active, Duplicated(copy(b), db_en))

    db_fd = ForwardDiff.gradient(f_ut, b)
    @test db_en ≈ db_fd rtol = 1.0e-8
end

@testset "LowerTriangular reverse" begin
    n = 4
    _A = tril(rand(n, n) + n * I)
    b = rand(n)

    function f_lt(b)
        prob = LinearProblem(LowerTriangular(_A), copy(b))
        sol = solve(prob)
        return sum(sol.u)
    end

    db_en = zero(b)
    Enzyme.autodiff(Reverse, Const(f_lt), Active, Duplicated(copy(b), db_en))

    db_fd = ForwardDiff.gradient(f_lt, b)
    @test db_en ≈ db_fd rtol = 1.0e-8
end

@testset "UnitUpperTriangular reverse" begin
    n = 4
    _A = triu(rand(n, n))
    for i in 1:n; _A[i, i] = 1.0; end
    b = rand(n)

    function f_uut(b)
        prob = LinearProblem(UnitUpperTriangular(_A), copy(b))
        sol = solve(prob)
        return sum(sol.u)
    end

    db_en = zero(b)
    Enzyme.autodiff(Reverse, Const(f_uut), Active, Duplicated(copy(b), db_en))

    db_fd = ForwardDiff.gradient(f_uut, b)
    @test db_en ≈ db_fd rtol = 1.0e-8
end

@testset "UnitLowerTriangular reverse" begin
    n = 4
    _A = tril(rand(n, n))
    for i in 1:n; _A[i, i] = 1.0; end
    b = rand(n)

    function f_ult(b)
        prob = LinearProblem(UnitLowerTriangular(_A), copy(b))
        sol = solve(prob)
        return sum(sol.u)
    end

    db_en = zero(b)
    Enzyme.autodiff(Reverse, Const(f_ult), Active, Duplicated(copy(b), db_en))

    db_fd = ForwardDiff.gradient(f_ult, b)
    @test db_en ≈ db_fd rtol = 1.0e-8
end

@testset "Diagonal reverse" begin
    n = 4
    d = rand(n) .+ 1.0  # avoid near-zero diagonal
    b = rand(n)

    function f_diag(b)
        prob = LinearProblem(Diagonal(d), copy(b))
        sol = solve(prob, DiagonalFactorization())
        return sum(sol.u)
    end

    db_en = zero(b)
    Enzyme.autodiff(Reverse, Const(f_diag), Active, Duplicated(copy(b), db_en))

    db_fd = ForwardDiff.gradient(f_diag, b)
    @test db_en ≈ db_fd rtol = 1.0e-8
end

@testset "Bidiagonal reverse" begin
    n = 4
    dv = rand(n) .+ 1.0
    ev = rand(n - 1) .* 0.1
    b = rand(n)

    function f_bidiag(b)
        prob = LinearProblem(Bidiagonal(dv, ev, :U), copy(b))
        sol = solve(prob)
        return sum(sol.u)
    end

    db_en = zero(b)
    Enzyme.autodiff(Reverse, Const(f_bidiag), Active, Duplicated(copy(b), db_en))

    db_fd = ForwardDiff.gradient(f_bidiag, b)
    @test db_en ≈ db_fd rtol = 1.0e-8
end

@testset "Tridiagonal reverse" begin
    n = 4
    dl = rand(n - 1) .* 0.1
    d  = rand(n) .+ 2.0
    du = rand(n - 1) .* 0.1
    b  = rand(n)

    function f_tridiag(b)
        prob = LinearProblem(Tridiagonal(dl, d, du), copy(b))
        sol = solve(prob)
        return sum(sol.u)
    end

    db_en = zero(b)
    Enzyme.autodiff(Reverse, Const(f_tridiag), Active, Duplicated(copy(b), db_en))

    db_fd = ForwardDiff.gradient(f_tridiag, b)
    @test db_en ≈ db_fd rtol = 1.0e-8
end

@testset "SymTridiagonal reverse" begin
    n = 4
    dv = rand(n) .+ 2.0
    ev = rand(n - 1) .* 0.1
    b  = rand(n)

    function f_symtridiag(b)
        prob = LinearProblem(SymTridiagonal(dv, ev), copy(b))
        sol = solve(prob, LDLtFactorization())
        return sum(sol.u)
    end

    db_en = zero(b)
    Enzyme.autodiff(Reverse, Const(f_symtridiag), Active, Duplicated(copy(b), db_en))

    db_fd = ForwardDiff.gradient(f_symtridiag, b)
    @test db_en ≈ db_fd rtol = 1.0e-8
end
