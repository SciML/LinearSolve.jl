using LinearSolve, Test
using SparseArrays, LinearAlgebra
using Krylov

m, n = 13, 3

A = rand(m, n)
b = rand(m)
prob = LinearProblem(A, b)
res = A \ b
@test solve(prob).u ≈ res
@test !LinearSolve.needs_square_A(QRFactorization())
@test solve(prob, QRFactorization()) ≈ res
@test !LinearSolve.needs_square_A(FastQRFactorization())
@test solve(prob, FastQRFactorization()) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL_LSMR())
@test solve(prob, KrylovJL_LSMR()) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.lsqr!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.lsqr!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.cgls!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.cgls!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.crls!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.crls!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.lslq!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.lslq!)) ≈ res

A = sprand(m, n, 0.5)
b = rand(m)
prob = LinearProblem(A, b)
res = A \ b
@test solve(prob).u ≈ res
@test solve(prob, QRFactorization()) ≈ res
@test solve(prob, KrylovJL_LSMR()) ≈ res

A = sprand(n, m, 0.5)
b = rand(n)
prob = LinearProblem(A, b)
res = Matrix(A) \ b
@test !LinearSolve.needs_square_A(KrylovJL_CRAIGMR())
@test solve(prob, KrylovJL_CRAIGMR()) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.cgne!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.cgne!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.craig!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.craig!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.crmr!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.crmr!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.lnlq!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.lnlq!)) ≈ res

A = sprandn(1000, 100, 0.1)
b = randn(1001)
prob = LinearProblem(A, view(b, 1:1000))
linsolve = init(prob, QRFactorization())
solve!(linsolve)

A = randn(1000, 100)
b = randn(1000)
@test isapprox(solve(LinearProblem(A, b)).u, Symmetric(A' * A) \ (A' * b))
solve(LinearProblem(A, b)).u;
@test !LinearSolve.needs_square_A(NormalCholeskyFactorization())
solve(LinearProblem(A, b), (LinearSolve.NormalCholeskyFactorization())).u;
@test !LinearSolve.needs_square_A(NormalBunchKaufmanFactorization())
solve(LinearProblem(A, b), (LinearSolve.NormalBunchKaufmanFactorization())).u;
solve(
    LinearProblem(A, b),
    assumptions = (
        OperatorAssumptions(
            false;
            condition = OperatorCondition.WellConditioned
        )
    )
).u;

A = sprandn(5000, 100, 0.1)
b = randn(5000)
@test isapprox(solve(LinearProblem(A, b)).u, ldlt(A' * A) \ (A' * b))
solve(LinearProblem(A, b)).u;
solve(LinearProblem(A, b), (LinearSolve.NormalCholeskyFactorization())).u;
solve(
    LinearProblem(A, b),
    assumptions = (
        OperatorAssumptions(
            false;
            condition = OperatorCondition.WellConditioned
        )
    )
).u;

# Underdetermined
m, n = 2, 3

A = rand(m, n)
b = rand(m)
prob = LinearProblem(A, b)
res = A \ b
@test solve(prob).u ≈ res

# Least-squares Krylov solvers with preconditioning: identity-equivalent counting
# preconditioner verifies Pl/Pr are actually forwarded to Krylov (not silently dropped).
mutable struct CountingDiagPrec
    d::Vector{Float64}
    calls::Int
end
CountingDiagPrec(d::AbstractVector) = CountingDiagPrec(collect(Float64, d), 0)
Base.size(P::CountingDiagPrec) = (length(P.d), length(P.d))
Base.size(P::CountingDiagPrec, ::Integer) = length(P.d)
Base.eltype(::CountingDiagPrec) = Float64
function LinearAlgebra.ldiv!(y::AbstractVector, P::CountingDiagPrec, x::AbstractVector)
    P.calls += 1
    @. y = x / P.d
    return y
end
function LinearAlgebra.ldiv!(P::CountingDiagPrec, x::AbstractVector)
    P.calls += 1
    @. x = x / P.d
    return x
end


@testset "LS family preconditioning" begin
    m, n = 30, 10
    A = randn(m, n)
    b = randn(m)
    res = A \ b

    ls_algs = [
        (KrylovJL_LSMR(), "LSMR", :both),
        (KrylovJL(KrylovAlg = Krylov.lsqr!), "LSQR", :both),
        (KrylovJL(KrylovAlg = Krylov.lslq!), "LSLQ", :both),
        (KrylovJL(KrylovAlg = Krylov.cgls!), "CGLS", :left_only),
        (KrylovJL(KrylovAlg = Krylov.crls!), "CRLS", :left_only),
    ]

    for (alg, name, support) in ls_algs
        @testset "$name" begin
            Pl = CountingDiagPrec(ones(m))
            Pr = CountingDiagPrec(ones(n))
            sol = solve(LinearProblem(A, b), alg; Pl = Pl, Pr = Pr)
            @test sol.u ≈ res
            @test Pl.calls > 0
            support === :both && @test Pr.calls > 0
        end
    end
end
