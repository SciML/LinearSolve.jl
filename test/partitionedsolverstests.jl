using LinearSolve
using PartitionedArrays
using PartitionedArrays: PSparseMatrix, PVector, with_debug, uniform_partition, tuple_of_arrays
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

@testset "PartitionedSolversAlgorithm: init validates PSparseMatrix/PVector" begin
    with_debug() do distribute
        rp = debug_row_partition(distribute, 8)
        A, b, u = build_splitmat_diag(rp)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), PartitionedSolversAlgorithm())
        @test cache.A isa PSparseMatrix
        @test cache.b isa PVector
        @test cache.u isa PVector
    end
end

@testset "PartitionedSolversAlgorithm: solve! stub is explicit" begin
    with_debug() do distribute
        rp = debug_row_partition(distribute, 8)
        A, b, u = build_splitmat_diag(rp)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), PartitionedSolversAlgorithm())
        err = try
            solve!(cache)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        @test occursin("not implemented yet", sprint(showerror, err))
    end
end

@testset "PartitionedSolversAlgorithm: init rejects non-partitioned inputs" begin
    A = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
    b = ones(2)
    @test_throws ArgumentError SciMLBase.init(
        LinearProblem(A, b), PartitionedSolversAlgorithm()
    )
end
