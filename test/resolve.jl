using LinearSolve, LinearAlgebra, SparseArrays, InteractiveUtils, Test
using LinearSolve: AbstractDenseFactorization, AbstractSparseFactorization,
    BLISLUFactorization, CliqueTreesFactorization,
    AMDGPUOffloadLUFactorization, AMDGPUOffloadQRFactorization,
    SparspakFactorization

# Function to check if an algorithm is mixed precision
function is_mixed_precision_alg(alg)
    alg_name = string(alg)
    return contains(alg_name, "32Mixed") || contains(alg_name, "Mixed32")
end

for alg in vcat(
        InteractiveUtils.subtypes(AbstractDenseFactorization),
        InteractiveUtils.subtypes(AbstractSparseFactorization)
    )
    if alg in [PardisoJL]
        ## Pardiso has extra tests in test/pardiso/pardiso.jl
        continue
    end
    @show alg
    if !(
            alg in [
                DiagonalFactorization,
                CudaOffloadFactorization,
                CudaOffloadLUFactorization,
                CudaOffloadQRFactorization,
                CUSOLVERRFFactorization,
                AppleAccelerateLUFactorization,
                MetalLUFactorization,
                FastLUFactorization,
                FastQRFactorization,
                CliqueTreesFactorization,
                BLISLUFactorization,
                AMDGPUOffloadLUFactorization,
                AMDGPUOffloadQRFactorization,
            ]
        ) &&
            (
            !(alg == AppleAccelerateLUFactorization) ||
                LinearSolve.appleaccelerate_isavailable()
        ) &&
            (!(alg == MKLLUFactorization) || LinearSolve.usemkl) &&
            (!(alg == OpenBLASLUFactorization) || LinearSolve.useopenblas) &&
            (!(alg == RFLUFactorization) || LinearSolve.userecursivefactorization(nothing)) &&
            (!(alg == RF32MixedLUFactorization) || LinearSolve.userecursivefactorization(nothing)) &&
            (!(alg == MKL32MixedLUFactorization) || LinearSolve.usemkl) &&
            (!(alg == AppleAccelerate32MixedLUFactorization) || Sys.isapple()) &&
            (!(alg == OpenBLAS32MixedLUFactorization) || LinearSolve.useopenblas) &&
            (!(alg == SparspakFactorization) || false)
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
            prob, alg(), alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
        )

        # Use higher tolerance for mixed precision algorithms
        expected = [-2.0, 1.5]
        if is_mixed_precision_alg(alg)
            @test solve!(linsolve).u ≈ expected atol = 1.0e-4 rtol = 1.0e-4
            @test !linsolve.isfresh
            @test solve!(linsolve).u ≈ expected atol = 1.0e-4 rtol = 1.0e-4
        else
            @test solve!(linsolve).u ≈ expected
            @test !linsolve.isfresh
            @test solve!(linsolve).u ≈ expected
        end

        A = [1.0 2.0; 3.0 4.0]
        alg in [KLUFactorization, UMFPACKFactorization, SparspakFactorization] &&
            (A = sparse(A))
        A = A' * A
        alg in [CHOLMODFactorization] && (A = sparse(Symmetric(A, :L)))
        alg in [BunchKaufmanFactorization] && (A = Symmetric(A, :L))
        alg in [LDLtFactorization] && (A = SymTridiagonal(A))
        linsolve.A = A
        @test linsolve.isfresh

        # Use higher tolerance for mixed precision algorithms
        if is_mixed_precision_alg(alg)
            @test solve!(linsolve).u ≈ expected atol = 1.0e-4 rtol = 1.0e-4
        else
            @test solve!(linsolve).u ≈ expected
        end
    end
end

A = Diagonal([1.0, 4.0])
b = [1.0, 2.0]
prob = LinearProblem(A, b)
linsolve = init(
    prob, DiagonalFactorization(),
    alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
)
@test solve!(linsolve).u ≈ [1.0, 0.5]
@test solve!(linsolve).u ≈ [1.0, 0.5]
A = Diagonal([1.0, 4.0])
linsolve.A = A
@test solve!(linsolve).u ≈ [1.0, 0.5]

A = Symmetric(
    [
        1.0 2.0
        2.0 1.0
    ]
)
b = [1.0, 2.0]
prob = LinearProblem(A, b)
linsolve = init(
    prob, BunchKaufmanFactorization(),
    alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
)
@test solve!(linsolve).u ≈ [1.0, 0.0]
@test solve!(linsolve).u ≈ [1.0, 0.0]
A = Symmetric(
    [
        1.0 2.0
        2.0 1.0
    ]
)
linsolve.A = A
@test solve!(linsolve).u ≈ [1.0, 0.0]

A = [
    1.0 2.0
    2.0 1.0
]
A = Symmetric(A * A')
b = [1.0, 2.0]
prob = LinearProblem(A, b)
linsolve = init(prob, CholeskyFactorization(), alias = LinearAliasSpecifier(alias_A = false, alias_b = false))
@test solve!(linsolve).u ≈ [-1 / 3, 2 / 3]
@test solve!(linsolve).u ≈ [-1 / 3, 2 / 3]
A = [
    1.0 2.0
    2.0 1.0
]
A = Symmetric(A * A')
b = [1.0, 2.0]
@test solve!(linsolve).u ≈ [-1 / 3, 2 / 3]
