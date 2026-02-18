using Zygote, ForwardDiff
using LinearSolve, LinearAlgebra, Test
using FiniteDiff, RecursiveFactorization
using Random

Random.seed!(1234)
n = 4
A = rand(n, n);
b1 = rand(n);

function f(A, b1; alg = LUFactorization())
    prob = LinearProblem(A, b1)

    sol1 = solve(prob, alg)

    s1 = sol1.u
    return norm(s1)
end

f(A, b1) # Uses BLAS

dA, db1 = Zygote.gradient(f, A, b1)
@test dA isa AbstractMatrix

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12

A = rand(n, n);
b1 = rand(n);

_ff = (
    x,
    y,
) -> f(
    x,
    y;
    alg = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization)
)
_ff(copy(A), copy(b1))

dA, db1 = Zygote.gradient(_ff, copy(A), copy(b1))
@test dA isa AbstractMatrix

dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

@test dA ≈ dA2
@test db1 ≈ db12

# Test complex numbers
A = rand(n, n) + 1im * rand(n, n);
b1 = rand(n) + 1im * rand(n);

function f3(A, b1, b2; alg = KrylovJL_GMRES())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg)
    prob = LinearProblem(A, b2)
    sol2 = solve(prob, alg)
    return norm(sol1.u .+ sol2.u)
end

dA, db1, db2 = Zygote.gradient(f3, A, b1, b1)
@test dA isa AbstractMatrix

dA2 = FiniteDiff.finite_difference_gradient(
    x -> f3(x, eltype(x).(b1), eltype(x).(b1)), copy(A)
)
db12 = FiniteDiff.finite_difference_gradient(
    x -> f3(eltype(x).(A), x, eltype(x).(b1)), copy(b1)
)
db22 = FiniteDiff.finite_difference_gradient(
    x -> f3(eltype(x).(A), eltype(x).(b1), x), copy(b1)
)

@test dA ≈ dA2 rtol = 1.0e-3
@test db1 ≈ db12
@test db2 ≈ db22

A = rand(n, n);
b1 = rand(n);

function f4(A, b1, b2; alg = LUFactorization())
    prob = LinearProblem(A, b1)
    sol1 = solve(prob, alg; sensealg = LinearSolveAdjoint(; linsolve = KrylovJL_LSMR()))
    prob = LinearProblem(A, b2)
    sol2 = solve(prob, alg; sensealg = LinearSolveAdjoint(; linsolve = KrylovJL_GMRES()))
    return norm(sol1.u .+ sol2.u)
end

dA, db1, db2 = Zygote.gradient(f4, A, b1, b1)
@test dA isa AbstractMatrix

dA2 = ForwardDiff.gradient(x -> f4(x, eltype(x).(b1), eltype(x).(b1)), copy(A))
db12 = ForwardDiff.gradient(x -> f4(eltype(x).(A), x, eltype(x).(b1)), copy(b1))
db22 = ForwardDiff.gradient(x -> f4(eltype(x).(A), eltype(x).(b1), x), copy(b1))

@test dA ≈ dA2 atol = 5.0e-5
@test db1 ≈ db12
@test db2 ≈ db22

A = rand(n, n);
b1 = rand(n);
for alg in (
        LUFactorization(),
        RFLUFactorization(),
        KrylovJL_GMRES(),
    )
    @show alg
    function fb(b)
        prob = LinearProblem(A, b)

        sol1 = solve(prob, alg)

        return sum(sol1.u)
    end
    fb(b1)

    fd_jac = FiniteDiff.finite_difference_jacobian(fb, b1) |> vec
    @show fd_jac

    zyg_jac = Zygote.jacobian(fb, b1) |> first |> vec
    @show zyg_jac

    @test zyg_jac ≈ fd_jac rtol = 5.0e-4

    function fA(A)
        prob = LinearProblem(A, b1)

        sol1 = solve(prob, alg)

        return sum(sol1.u)
    end
    fA(A)

    fd_jac = FiniteDiff.finite_difference_jacobian(fA, A) |> vec
    @show fd_jac

    zyg_jac = Zygote.jacobian(fA, A) |> first |> vec
    @show zyg_jac

    @test zyg_jac ≈ fd_jac rtol = 5.0e-4
end

@testset "Direct solution indexing without .u" begin
    N = 2
    Random.seed!(1234)
    function test_func(x::AbstractVector{T}) where {T <: Real}
        A = reshape(x[1:(N * N)], (N, N))
        b = x[(N * N + 1):end]
        prob = LinearProblem(A, b)
        sol = solve(prob)
        return sum(sol)
    end

    x0 = rand(N * N + N)

    grad_zygote = Zygote.gradient(test_func, x0)
    grad_forwarddiff = ForwardDiff.gradient(test_func, x0)
    @test grad_zygote[1] ≈ grad_forwarddiff rtol = 1e-5
end

struct System end
function update_A!(A, p::Vector{T}) where {T}
    A[1, 1] = sum(p)
    A[1, 2] = one(T)
    A[2, 1] = 2one(T)
    return A[2, 2] = 3one(T)
end
function update_b!(b, p::Vector{T}) where {T}
    b[1] = p[1]
    return b[2] = 2one(T)
end

function SciMLBase.get_new_A_b(::System, f, p, A, b; kw...)
    if eltype(A) != eltype(p)
        A = similar(A, eltype(p))
        b = similar(b, eltype(p))
    end
    f.update_A!(A, p)
    f.update_b!(b, p)
    return A, b
end


@testset "Avoid stackoverflow with `SciMLBase.get_new_A_b` hook" begin
    # See https://github.com/SciML/LinearSolve.jl/pull/868#issuecomment-3723338914
    p = [1.0, 2.0, 3.0]
    A = [6.0 1.0; 2.0 3.0]
    b = [1.0, 2.0]
    linfun = SciMLBase.SymbolicLinearInterface(update_A!, update_b!, System(), nothing, nothing)

    prob = LinearProblem(A, b; f = linfun)
    sol = solve(prob)

    newp = [2.0, 3.0, 4.0]
    @test_nowarn ForwardDiff.gradient(newp) do p
        prob2 = remake(prob; p)
        sum(solve(prob2).u)
    end
end
