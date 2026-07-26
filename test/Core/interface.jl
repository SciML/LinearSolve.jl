using LinearSolve, LinearAlgebra, SparseArrays, Test
using SciMLBase: SciMLBase
# Loading the backends of the extension-provided algorithms brings their
# `solve!` methods into scope so they are covered by the sweep below.
using AlgebraicMultigrid, IterativeSolvers, KrylovKit, RecursiveFactorization

# `solve!` for these lives in a package extension whose backend is not a test
# dependency of the Core group. Their traits must still resolve without the
# backend -- that is what the trait sweep checks -- and their `solve!` is
# covered by their own test group.
const SOLVE_NEEDS_UNLOADED_BACKEND = Set(
    Any[GinkgoJL, HYPREAlgorithm, PETScAlgorithm, PartitionedSolversAlgorithm]
)

# A minimal compliant algorithm: implements the two required methods and takes
# the default for everything else.
struct OnlyRequiredAlg <: LinearSolve.SciMLLinearSolveAlgorithm end
LinearSolve.needs_concrete_A(::OnlyRequiredAlg) = true
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::OnlyRequiredAlg; kwargs...)
    ldiv!(cache.u, lu(cache.A), cache.b)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = SciMLBase.ReturnCode.Success
    )
end

struct NoTraitAlg <: LinearSolve.SciMLLinearSolveAlgorithm end

struct NoSolveAlg <: LinearSolve.SciMLLinearSolveAlgorithm end
LinearSolve.needs_concrete_A(::NoSolveAlg) = true

@testset "Every algorithm satisfies the SciMLLinearSolveAlgorithm interface" begin
    # The deliberately non-compliant fixtures above live in this module.
    algorithm_types = filter(
        T -> parentmodule(T) !== @__MODULE__, LinearSolve.concrete_algorithm_types()
    )
    # A sanity floor: if `subtypes` ever stops seeing the algorithms, the sweep
    # below would pass vacuously.
    @test length(algorithm_types) > 40

    for T in algorithm_types
        check_solve = !(T in SOLVE_NEEDS_UNLOADED_BACKEND)
        # Compared as a pair so a failure names the offending algorithm.
        @test (T, LinearSolve.algorithm_interface_issues(T; check_solve)) == (T, String[])
    end

    # An algorithm may only be excluded from the `solve!` check while it really
    # is missing one; an entry that has become loadable must not stay excluded.
    missing_solve = Set(
        Any[
            T for T in algorithm_types
                if !isempty(LinearSolve.algorithm_interface_issues(T))
        ]
    )
    @test issubset(missing_solve, SOLVE_NEEDS_UNLOADED_BACKEND)
end

@testset "Traits resolve for algorithms whose backend is not loaded" begin
    # #277/#1115: downstream solvers query the traits of a `linsolve` before the
    # backend package is loaded, so trait definitions must not live in an
    # extension. These types are constructible only with their backend, so the
    # trait methods are checked at the type level.
    for T in (
            HYPREAlgorithm, PETScAlgorithm, PartitionedSolversAlgorithm, GinkgoJL,
            AlgebraicMultigridJL,
        )
        @test which(LinearSolve.needs_concrete_A, Tuple{T}) !==
            which(LinearSolve.needs_concrete_A, Tuple{LinearSolve.SciMLLinearSolveAlgorithm})
        @test Base.return_types(LinearSolve.needs_concrete_A, Tuple{T}) == [Bool]
    end

    # Ginkgo copies `A` into its own sparse format, so it is the one Krylov
    # wrapper that overrides the `AbstractKrylovSubspaceMethod` default.
    @test LinearSolve.needs_concrete_A(GinkgoJL(nothing, :omp, (), (;)))
    @test !LinearSolve.needs_concrete_A(KrylovJL_GMRES())
end

@testset "Traits defined in an extension are reported" begin
    amg_ext = Base.get_extension(LinearSolve, :LinearSolveAlgebraicMultigridExt)
    @test amg_ext !== nothing
    @test LinearSolve._is_extension_of(amg_ext, LinearSolve)
    @test !LinearSolve._is_extension_of(LinearSolve, LinearSolve)
    @test !LinearSolve._is_extension_of(amg_ext, Base)

    # `_extension_defining_trait` is what turns "the trait lives in the
    # extension" into an interface violation; it returns the offending module.
    for (trait, sig) in (
            (LinearSolve.needs_concrete_A, Tuple{AlgebraicMultigridJL}),
            (LinearSolve.needs_square_A, Tuple{AlgebraicMultigridJL}),
            (LinearSolve.default_alias_A, Tuple{AlgebraicMultigridJL, Any, Any}),
            (LinearSolve.default_alias_b, Tuple{AlgebraicMultigridJL, Any, Any}),
        )
        @test LinearSolve._extension_defining_trait(trait, sig, AlgebraicMultigridJL) ===
            nothing
        @test which(trait, sig).module === LinearSolve
    end

    # `init_cacheval` and `solve!` legitimately live in the extension.
    @test which(
        LinearSolve.init_cacheval,
        Tuple{
            AlgebraicMultigridJL, Any, Any, Any, Any, Any, Int, Any, Any,
            LinearSolve.LinearVerbosity, LinearSolve.OperatorAssumptions,
        }
    ).module === amg_ext
    @test which(
        SciMLBase.solve!, Tuple{LinearSolve.LinearCache, AlgebraicMultigridJL}
    ).module === amg_ext

    # The detector fires on a method that really is extension-defined.
    # `update_tolerances_internal!` is allowed to live there (it pokes at the
    # backend's iterable), so it is not one of the traits checked above.
    is_ext = Base.get_extension(LinearSolve, :LinearSolveIterativeSolversExt)
    @test LinearSolve._extension_defining_trait(
        LinearSolve.update_tolerances_internal!,
        Tuple{Any, IterativeSolversJL, Any, Any}, IterativeSolversJL
    ) === is_ext
end

@testset "Trait values" begin
    @test LinearSolve.needs_concrete_A(LUFactorization())
    @test LinearSolve.needs_concrete_A(UMFPACKFactorization())
    @test !LinearSolve.needs_concrete_A(KrylovJL_CG())
    @test !LinearSolve.needs_concrete_A(LinearSolve.DirectLdiv!())
    @test LinearSolve.needs_concrete_A(
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization)
    )

    @test LinearSolve.needs_square_A(LUFactorization())
    @test !LinearSolve.needs_square_A(QRFactorization())
    @test !LinearSolve.needs_square_A(nothing)

    A = rand(4, 4)
    b = rand(4)
    @test !LinearSolve.default_alias_A(LUFactorization(), A, b)
    @test !LinearSolve.default_alias_b(LUFactorization(), A, b)
    @test LinearSolve.default_alias_A(KrylovJL_GMRES(), A, b)
    @test LinearSolve.default_alias_b(KrylovJL_GMRES(), A, b)
    @test LinearSolve.default_alias_A(UMFPACKFactorization(), sparse(A), b)
    @test LinearSolve.default_alias_b(UMFPACKFactorization(), sparse(A), b)
end

@testset "Optional methods have the documented defaults" begin
    alg = OnlyRequiredAlg()
    @test isempty(LinearSolve.algorithm_interface_issues(alg))
    @test isempty(LinearSolve.algorithm_interface_issues(OnlyRequiredAlg))

    A = rand(4, 4)
    b = rand(4)
    @test LinearSolve.needs_square_A(alg)
    @test !LinearSolve.default_alias_A(alg, A, b)
    @test !LinearSolve.default_alias_b(alg, A, b)

    prob = LinearProblem(A, b)
    cache = init(prob, alg)
    @test cache.cacheval === nothing            # init_cacheval default
    @test solve!(cache).u ≈ A \ b

    # An algorithm with no tolerances to update says so rather than silently
    # ignoring the request.
    @test_throws ArgumentError LinearSolve.update_tolerances!(cache; abstol = 1.0e-10)
end

@testset "update_tolerances! reaches every in-tree algorithm class" begin
    prob = LinearProblem(rand(4, 4), rand(4))
    # Krylov methods read the tolerances from the cache at solve time, and the
    # default algorithm dispatches only to algorithms that do that or ignore
    # tolerances entirely.
    for alg in (nothing, KrylovJL_GMRES(), LinearSolve.DirectLdiv!())
        cache = init(prob, alg)
        @test LinearSolve.update_tolerances!(cache; abstol = 1.0e-10, reltol = 1.0e-9) ===
            nothing
        @test cache.abstol == 1.0e-10
        @test cache.reltol == 1.0e-9
    end
    # Factorizations have no tolerance to update and say so.
    @test_throws ErrorException LinearSolve.update_tolerances!(
        init(prob, LUFactorization()); abstol = 1.0e-10
    )
end

@testset "Missing required methods give actionable errors" begin
    @test LinearSolve.algorithm_interface_issues(NoTraitAlg) ==
        LinearSolve.algorithm_interface_issues(NoTraitAlg())
    @test length(LinearSolve.algorithm_interface_issues(NoTraitAlg())) == 2
    @test only(LinearSolve.algorithm_interface_issues(NoSolveAlg())) ==
        "$(NoSolveAlg) does not implement `SciMLBase.solve!(::LinearCache, ::$(NoSolveAlg))`."
    @test isempty(LinearSolve.algorithm_interface_issues(NoSolveAlg; check_solve = false))

    err = try
        LinearSolve.needs_concrete_A(NoTraitAlg())
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("needs_concrete_A", err.msg)
    @test occursin("package extension", err.msg)

    # Without the `solve!` fallback this dispatches to
    # `solve!(cache::LinearCache, args...)`, which forwards to itself forever.
    prob = LinearProblem(rand(4, 4), rand(4))
    @test_throws ArgumentError solve(prob, NoSolveAlg())

    @test_throws ArgumentError LinearSolve.algorithm_interface_issues(Matrix{Float64})
end
