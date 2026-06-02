using LinearSolve
using MPI
using PartitionedArrays
using PartitionedArrays: PVector, own_to_local, partition, tuple_of_arrays,
    uniform_partition, with_mpi
using PartitionedSolvers
using Test
import SciMLBase

MPI.Init()

function mpi_row_partition(distribute, n)
    parts = distribute(LinearIndices((MPI.Comm_size(MPI.COMM_WORLD),)))
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

@testset "PartitionedSolversAlgorithm MPI: CG solve and reuse" begin
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, 16)
        A, b, u0 = build_splitmat_diag(rp)
        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u0),
            PartitionedSolversAlgorithm(PartitionedSolvers.cg);
            abstol = 1.0e-12,
            reltol = 1.0e-12,
            maxiters = 40
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

@testset "PartitionedSolversAlgorithm MPI: solver-agnostic (jacobi)" begin
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, 16)
        A, b, u0 = build_splitmat_diag(rp)
        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u0),
            PartitionedSolversAlgorithm(PartitionedSolvers.jacobi);
            maxiters = 20
        )

        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
    end
end

@testset "PartitionedSolversAlgorithm MPI: defaultalg dispatch" begin
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, 16)
        A, b, u0 = build_splitmat_diag(rp)

        alg = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
        @test alg isa PartitionedSolversAlgorithm

        sol = solve(LinearProblem(A, b; u0 = u0), alg; abstol = 1.0e-12, reltol = 1.0e-12)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
    end
end
