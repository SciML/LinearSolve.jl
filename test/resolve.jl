using LinearSolve, LinearAlgebra, SparseArrays, InteractiveUtils, Test
using LinearSolve: AbstractDenseFactorization, AbstractSparseFactorization

for alg in vcat(InteractiveUtils.subtypes(AbstractDenseFactorization),
    InteractiveUtils.subtypes(AbstractSparseFactorization))
    if alg in [PardisoJL]
        ## Pardiso has extra tests in test/pardiso/pardiso.jl
        continue
    end
    @show alg
    if !(alg in [
           DiagonalFactorization,
           CudaOffloadFactorization,
           AppleAccelerateLUFactorization,
           MetalLUFactorization
       ]) &&
       (!(alg == AppleAccelerateLUFactorization) ||
        LinearSolve.appleaccelerate_isavailable()) &&
       (!(alg == MKLLUFactorization) || LinearSolve.usemkl)
        A = [1.0 2.0; 3.0 4.0]
        alg in [KLUFactorization, UMFPACKFactorization, SparspakFactorization] &&
            (A = sparse(A))
        A = A' * A
        @show A
        alg in [CHOLMODFactorization] && (A = sparse(Symmetric(A, :L)))
        alg in [BunchKaufmanFactorization] && (A = Symmetric(A, :L))
        alg in [LDLtFactorization] && (A = SymTridiagonal(A))
        b = [1.0, 2.0]
        prob = LinearProblem(A, b)
        linsolve = init(
            prob, alg(), alias = LinearAliasSpecifier(alias_A = false, alias_b = false))
        @test solve!(linsolve).u ≈ [-2.0, 1.5]
        @test !linsolve.isfresh
        @test solve!(linsolve).u ≈ [-2.0, 1.5]

        A = [1.0 2.0; 3.0 4.0]
        alg in [KLUFactorization, UMFPACKFactorization, SparspakFactorization] &&
            (A = sparse(A))
        A = A' * A
        alg in [CHOLMODFactorization] && (A = sparse(Symmetric(A, :L)))
        alg in [BunchKaufmanFactorization] && (A = Symmetric(A, :L))
        alg in [LDLtFactorization] && (A = SymTridiagonal(A))
        linsolve.A = A
        @test linsolve.isfresh
        @test solve!(linsolve).u ≈ [-2.0, 1.5]
    end
end

A = Diagonal([1.0, 4.0])
b = [1.0, 2.0]
prob = LinearProblem(A, b)
linsolve = init(prob, DiagonalFactorization(),
    alias = LinearAliasSpecifier(alias_A = false, alias_b = false))
@test solve!(linsolve).u ≈ [1.0, 0.5]
@test solve!(linsolve).u ≈ [1.0, 0.5]
A = Diagonal([1.0, 4.0])
linsolve.A = A
@test solve!(linsolve).u ≈ [1.0, 0.5]

A = Symmetric([1.0 2.0
               2.0 1.0])
b = [1.0, 2.0]
prob = LinearProblem(A, b)
linsolve = init(prob, BunchKaufmanFactorization(),
    alias = LinearAliasSpecifier(alias_A = false, alias_b = false))
@test solve!(linsolve).u ≈ [1.0, 0.0]
@test solve!(linsolve).u ≈ [1.0, 0.0]
A = Symmetric([1.0 2.0
               2.0 1.0])
linsolve.A = A
@test solve!(linsolve).u ≈ [1.0, 0.0]

A = [1.0 2.0
     2.0 1.0]
A = Symmetric(A * A')
b = [1.0, 2.0]
prob = LinearProblem(A, b)
linsolve = init(prob, CholeskyFactorization(), alias_A = false, alias_b = false)
@test solve!(linsolve).u ≈ [-1 / 3, 2 / 3]
@test solve!(linsolve).u ≈ [-1 / 3, 2 / 3]
A = [1.0 2.0
     2.0 1.0]
A = Symmetric(A * A')
b = [1.0, 2.0]
@test solve!(linsolve).u ≈ [-1 / 3, 2 / 3]
