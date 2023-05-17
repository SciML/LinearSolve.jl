using LinearSolve, LinearAlgebra, SparseArrays, InteractiveUtils, Test

for alg in subtypes(LinearSolve.AbstractFactorization)
    @show alg
    if !(alg in [DiagonalFactorization])
        A = [1.0 2.0; 3.0 4.0]
        alg in [KLUFactorization, UMFPACKFactorization, SparspakFactorization] &&
            (A = sparse(A))
        A = A' * A
        b = [1.0, 2.0]
        prob = LinearProblem(A, b)
        linsolve = init(prob, alg(), alias_A = false, alias_b = false)
        @test solve!(linsolve).u ≈ [-2.0, 1.5]
        @test !linsolve.isfresh
        @test solve!(linsolve).u ≈ [-2.0, 1.5]
        A = [1.0 2.0; 3.0 4.0]
        alg in [KLUFactorization, UMFPACKFactorization, SparspakFactorization] &&
            (A = sparse(A))
        A = A' * A
        linsolve.A = A
        @test linsolve.isfresh
        @test solve!(linsolve).u ≈ [-2.0, 1.5]
    end
end

A = Diagonal([1.0, 4.0])
b = [1.0, 2.0]
prob = LinearProblem(A, b)
linsolve = init(prob, DiagonalFactorization(), alias_A = false, alias_b = false)
@test solve!(linsolve).u ≈ [1.0, 0.5]
@test solve!(linsolve).u ≈ [1.0, 0.5]
A = Diagonal([1.0, 4.0])
linsolve.A = A
@test solve!(linsolve).u ≈ [1.0, 0.5]
