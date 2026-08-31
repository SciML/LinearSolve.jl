using LinearSolve
using ForwardDiff
using JLArrays
using LinearAlgebra
using SciMLOperators: WOperator
using Test

# JLArray is a CPU-backed AbstractGPUArray, so it exercises the GPU contract
# without a GPU: scalar getindex/setindex! on it throws unless explicitly
# permitted, and nothing here permits it. That makes these testsets a proof
# rather than a smoke test -- the whole split-Dual path (partials extraction,
# rhs construction, back-solves, dual reassembly) has to stay in broadcasts and
# BLAS-level calls, or they error instead of merely running slowly.
#
# The device solves use KrylovJL_GMRES, not an LU factorization: JLArray ships
# no native `lu` (real GPU backends get theirs from CUSOLVER etc.), and since
# JLArrays 0.3.2 (JuliaGPU/GPUArrays.jl#549) `unsafe_convert(Ptr, ::JLArray)`
# throws, so the stdlib LAPACK fallback that used to silently factorize the
# underlying CPU buffer errors instead of running. A Krylov solve is the path
# that genuinely honors the GPU contract here, which is what this file proves.

const N_PARTIALS = 3

@testset "WOperator with a GPU right-hand side" begin
    W = WOperator{true}(I, 0.1, Matrix{Float64}(I, 2, 2), zeros(2))
    alg = LinearSolve.defaultalg(W, JLArray(ones(2)), LinearSolve.OperatorAssumptions(true))
    @test alg.alg === LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
end

function dual_problem(n, p)
    A = [
        ForwardDiff.Dual{Nothing}(
            float(i == j ? 10 + i : 0.3 * (i + j)), ntuple(k -> 0.1k + 0.01 * (i + j), p)
        ) for i in 1:n, j in 1:n
    ]
    b = [ForwardDiff.Dual{Nothing}(float(i), ntuple(k -> 0.05k + 0.1i, p)) for i in 1:n]
    return A, b
end

value_error(got, ref) = maximum(abs, ForwardDiff.value.(got) .- ForwardDiff.value.(ref))

function partials_error(got, ref)
    return maximum(
        maximum(
            abs,
            collect(ForwardDiff.partials(x)) .- collect(ForwardDiff.partials(y))
        ) for (x, y) in zip(got, ref)
    )
end

@testset "Dual solve! on a GPU array (JLArray)" begin
    # The premise the rest of the file rests on. If a future GPUArraysCore
    # defaulted to permitting scalar indexing these testsets would still pass
    # while proving nothing, so pin it down explicitly.
    @test_throws ErrorException JLArray([1.0, 2.0])[1]

    n = 6
    A, b = dual_problem(n, N_PARTIALS)
    A_primal = ForwardDiff.value.(A)
    b_primal = ForwardDiff.value.(b)

    # Each case reaches a different `xp_linsolve_rhs!` method: the fused dense-∂A
    # one, the ∂_A === nothing one, and the ∂_b === nothing one.
    cases = (
        ("Dual A and Dual b", A, b),
        ("Dual b only", A_primal, b),
        ("Dual A only", A, b_primal),
    )

    @testset "$name" for (name, A_case, b_case) in cases
        # Reference from the CPU path, so this checks agreement with the
        # supported implementation rather than mere self-consistency.
        reference = solve(LinearProblem(A_case, b_case), LUFactorization()).u

        cache = init(
            LinearProblem(JLArray(A_case), JLArray(b_case)), KrylovJL_GMRES();
            abstol = 1.0e-14, reltol = 1.0e-14
        )
        solution = Array(solve!(cache).u)
        @test value_error(solution, reference) < 1.0e-12
        @test partials_error(solution, reference) < 1.0e-12

        # Re-solving after a mutation is what drives `update_partials_list!`;
        # the first solve skips it, since the cache starts marked valid.
        cache.b = JLArray(b_case)
        resolved = Array(solve!(cache).u)
        @test value_error(resolved, reference) < 1.0e-12
        @test partials_error(resolved, reference) < 1.0e-12
    end

    @testset "A reassigned between solves" begin
        cache = init(
            LinearProblem(JLArray(A), JLArray(b)), KrylovJL_GMRES();
            abstol = 1.0e-14, reltol = 1.0e-14
        )
        solve!(cache)

        A2 = A .+ ForwardDiff.Dual{Nothing}(0.5, ntuple(k -> 0.02k, N_PARTIALS))
        cache.A = JLArray(A2)
        resolved = Array(solve!(cache).u)
        reference = solve(LinearProblem(A2, b), LUFactorization()).u
        @test value_error(resolved, reference) < 1.0e-12
        @test partials_error(resolved, reference) < 1.0e-12
    end
end
