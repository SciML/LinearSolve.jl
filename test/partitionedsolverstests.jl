using LinearSolve
using PartitionedArrays
using PartitionedArrays: PSparseMatrix, PVector, own_to_local, partition, tuple_of_arrays,
    uniform_partition, with_debug
using PartitionedSolvers
using SparseArrays
using Test
import SciMLBase

const PSExt = Base.get_extension(LinearSolve, :LinearSolvePartitionedSolversExt)

function debug_row_partition(distribute, n)
    parts = distribute(LinearIndices((1,)))
    return uniform_partition(parts, n)
end

function build_splitmat_diag(row_partition, scale = 1.0)
    I_v, J_v, V_v = map(row_partition) do rng
        collect(Int, rng), collect(Int, rng), scale .* Float64.(rng)
    end |> tuple_of_arrays
    A = psparse(I_v, J_v, V_v, row_partition, row_partition) |> fetch
    b = PVector(map(rng -> scale .* Float64.(rng), row_partition), row_partition)
    u = PVector(map(rng -> zeros(length(rng)), row_partition), row_partition)
    return A, b, u
end

function assert_owned_approx(u::PVector, expected_val; atol = 1.0e-8)
    map(partition(u), partition(axes(u, 1))) do local_u, row_idx
        for j in own_to_local(row_idx)
            @test local_u[j] ≈ expected_val atol = atol
        end
    end
    return nothing
end

@testset "PartitionedSolversAlgorithm: extension loaded" begin
    @test PSExt !== nothing
end

@testset "PartitionedSolversAlgorithm: debug solve and cache reuse" begin
    with_debug() do distribute
        rp = debug_row_partition(distribute, 8)
        A, b, u0 = build_splitmat_diag(rp)
        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u0),
            PartitionedSolversAlgorithm(PartitionedSolvers.cg);
            abstol = 1.0e-12,
            reltol = 1.0e-12,
            maxiters = 20
        )

        sol1 = solve!(cache)
        @test sol1.retcode == SciMLBase.ReturnCode.Success
        @test cache.cacheval.solver !== nothing
        assert_owned_approx(sol1.u, 1.0)

        b2 = PVector(map(rng -> 2.0 .* Float64.(rng), rp), rp)
        cache.b = b2
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol2.u, 2.0)

        A2, b3, _ = build_splitmat_diag(rp, 3.0)
        cache.A = A2
        cache.b = b3
        sol3 = solve!(cache)
        @test sol3.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol3.u, 1.0)
    end
end

@testset "PartitionedSolversAlgorithm: defaultalg dispatch" begin
    with_debug() do distribute
        rp = debug_row_partition(distribute, 8)
        A, b, u0 = build_splitmat_diag(rp)

        alg = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
        @test alg isa PartitionedSolversAlgorithm

        sol = solve(LinearProblem(A, b; u0 = u0), alg; abstol = 1.0e-12, reltol = 1.0e-12)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
    end
end

@testset "PartitionedSolversAlgorithm: local default solver path" begin
    with_debug() do distribute
        rp = debug_row_partition(distribute, 8)
        A, b, u0 = build_splitmat_diag(rp)

        sol = solve(LinearProblem(A, b; u0 = u0), PartitionedSolversAlgorithm())
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test sol.iters == 0
        assert_owned_approx(sol.u, 1.0)
    end
end

@testset "PartitionedSolversAlgorithm: solver-agnostic (jacobi)" begin
    # cg slurps keyword arguments, but jacobi has a fixed signature (iterations, omega).
    # This exercises the solver-aware keyword forwarding so the integration is not cg-only.
    with_debug() do distribute
        rp = debug_row_partition(distribute, 8)
        A, b, u0 = build_splitmat_diag(rp)
        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u0),
            PartitionedSolversAlgorithm(PartitionedSolvers.jacobi);
            maxiters = 20
        )

        sol1 = solve!(cache)
        @test sol1.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol1.u, 1.0)

        b2 = PVector(map(rng -> 2.0 .* Float64.(rng), rp), rp)
        cache.b = b2
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol2.u, 2.0)
    end
end

@testset "PartitionedSolversAlgorithm: maxiters maps to non-success retcode" begin
    with_debug() do distribute
        rp = debug_row_partition(distribute, 64)
        A, b, u0 = build_splitmat_diag(rp)
        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u0),
            PartitionedSolversAlgorithm(PartitionedSolvers.cg);
            abstol = 1.0e-16,
            reltol = 1.0e-16,
            maxiters = 1
        )

        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.MaxIters
        @test sol.iters == 1
        @test sol.resid > 0.0
    end
end

@testset "PartitionedSolversAlgorithm: init rejects non-partitioned inputs" begin
    A = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
    b = ones(2)
    @test_throws ArgumentError SciMLBase.init(
        LinearProblem(A, b), PartitionedSolversAlgorithm()
    )
end
