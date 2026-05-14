using LinearSolve
using LinearAlgebra
using SparseArrays
using JET
using Test

# Type-inference QA for `solve!(cache)`. The regression this guards against:
# `solve!(cache)` going through `DefaultLinearSolver` returned
# `LinearSolution{_A, _B, _C, _D, DefaultLinearSolver, _E, _F} where {...}`
# (a UnionAll over 6 free type parameters). The expected concrete return is
# `LinearSolution{Float64, 1, Vector{Float64}, Nothing, DefaultLinearSolver,
# <concrete LinearCache type>, Nothing}`.

# solve!(init(prob, alg)) is the full chain the user sees from solve(prob, alg).
# We re-init each time so the test exercises both `init` and `solve!`.
_solve_alg(A, b, alg) = solve!(init(LinearProblem(A, b), alg))
_solve_default(A, b) = solve!(init(LinearProblem(A, b)))

@testset "JET / type-inference" begin
    @testset "Default solver — solve!(cache) returns concrete LinearSolution" begin
        # Headline case: `solve!(cache)` after `init(LinearProblem(A, b))` must
        # not return a UnionAll-typed LinearSolution. Was broken by the
        # `_default_lu_solve_with_fallback`/`_do_qr_fallback` helpers reading
        # `sol.u`/`sol.resid`/`sol.cache`/`sol.stats` from an inner `sol` whose
        # rettype got capped to `Any` during precompile.
        rt = Core.Compiler.return_type(
            _solve_default, Tuple{Matrix{Float64}, Vector{Float64}}
        )
        @test isconcretetype(rt)
        @test rt <: LinearSolve.SciMLBase.LinearSolution{Float64, 1, Vector{Float64}}
    end

    @testset "solve!(cache) is concrete for each algorithm" begin
        # Each algorithm passed directly through `init(prob, alg)` must give a
        # concrete LinearSolution out of `solve!(cache)`.
        algs_concrete = (
            LUFactorization(),
            GenericLUFactorization(),
            QRFactorization(LinearAlgebra.ColumnNorm()),
            QRFactorization(LinearAlgebra.NoPivot()),
            DiagonalFactorization(),
            SVDFactorization(),
            CholeskyFactorization(),
            NormalCholeskyFactorization(),
        )
        for alg in algs_concrete
            @testset "$(nameof(typeof(alg)))" begin
                rt = Core.Compiler.return_type(
                    _solve_alg,
                    Tuple{Matrix{Float64}, Vector{Float64}, typeof(alg)}
                )
                @test isconcretetype(rt)
            end
        end

        # These have known unrelated inference issues (see test/nopre/jet.jl).
        # Tracked separately; not what this group is guarding against.
        algs_broken = (
            BunchKaufmanFactorization(),
            LDLtFactorization(),
        )
        for alg in algs_broken
            @testset "$(nameof(typeof(alg))) (broken)" begin
                rt = Core.Compiler.return_type(
                    _solve_alg,
                    Tuple{Matrix{Float64}, Vector{Float64}, typeof(alg)}
                )
                @test_broken isconcretetype(rt)
            end
        end
    end

    @testset "JET.@test_opt on the default solver" begin
        # Marked broken: the default-solver @generated function dispatches to
        # every algorithm branch at inference time. Several of those branches
        # (LDLt, Krylov, etc.) still have unrelated runtime-dispatch sites
        # inside LinearSolve and Krylov that JET reports. The concrete-rettype
        # tests above are the load-bearing check for this group; the @test_opt
        # is here to ratchet down the remaining dispatch issues over time.
        JET.@test_opt target_modules=(LinearSolve,) broken=true _solve_default(
            rand(4, 4), rand(4)
        )
    end
end
