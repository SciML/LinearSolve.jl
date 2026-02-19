using LinearAlgebra
using LinearSolve
using MPI
using PETSc
using SparseArrays
using Random
using Test

Random.seed!(1234)

MPI.Init()

const PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)
const rank     = MPI.Comm_rank(MPI.COMM_WORLD)
const comm_size   = MPI.Comm_size(MPI.COMM_WORLD)

# ══════════════════════════════════════════════════════════════════════════════
#  MPI-PARALLEL TESTS  (comm = MPI.COMM_WORLD)
#
#  Run with: mpiexecjl -n 2 julia test/petsctests_mpi.jl
#  (or -n 4, etc.)
#
#  Phase 1 strategy: every rank holds the full Julia matrix; PETSc distributes
#  the rows across ranks for the solve.
# ══════════════════════════════════════════════════════════════════════════════


@testset "MPI (comm_size=$comm_size): Ownership Distribution" begin
    n = 100
    A = sprand(n, n, 0.1) + 10I
    alg = PETScAlgorithm(:gmres; comm = MPI.COMM_WORLD)
    cache = SciMLBase.init(LinearProblem(A, rand(n)), alg)
    solve!(cache)

    pcache = cache.cacheval
    @test pcache.rend - pcache.rstart <= (n ÷ comm_size) + 1

    total_rows = MPI.Allreduce(pcache.rend - pcache.rstart, MPI.SUM, MPI.COMM_WORLD)
    @test total_rows == n

    PETScExt.cleanup_petsc_cache!(cache)
end


@testset "MPI (comm_size=$comm_size): GMRES + Jacobi" begin
    n = 200
    A = sprand(n, n, 0.05) + 10I; A = A'A
    b = rand(n)
    prob = LinearProblem(A, b)
    alg = PETScAlgorithm(:gmres; pc_type = :jacobi, comm = MPI.COMM_WORLD)

    sol = solve(prob, alg; abstol = 1e-10, reltol = 1e-10)

    # solve! now auto-gathers — sol.u is the full solution on all ranks
    if rank == 0
        @test norm(A * sol.u - b) / norm(b) < 1e-6
    end

    PETScExt.cleanup_petsc_cache!(sol)
end

@testset "MPI (comm_size=$comm_size): CG" begin
    n = 200
    A = sprand(n, n, 0.05) + 10I; A = A'A; b = rand(n)
    prob = LinearProblem(A, b)
    alg = PETScAlgorithm(:cg; pc_type = :jacobi, comm = MPI.COMM_WORLD)
    sol = solve(prob, alg; abstol = 1e-10, reltol = 1e-10)

    if rank == 0
        @test norm(A * sol.u - b) / norm(b) < 1e-6
    end

    PETScExt.cleanup_petsc_cache!(sol)
end

@testset "MPI (comm_size=$comm_size): Dense Matrix" begin
    n = 80
    A = rand(n, n) + 10I; A = A'A; b = rand(n)
    prob = LinearProblem(A, b)
    alg = PETScAlgorithm(:gmres; pc_type = :jacobi, comm = MPI.COMM_WORLD)
    sol = solve(prob, alg; abstol = 1e-10, reltol = 1e-10)

    if rank == 0
        @test norm(A * sol.u - b) / norm(b) < 1e-6
    end
    PETScExt.cleanup_petsc_cache!(sol)
end

@testset "MPI (comm_size=$comm_size): Cache Interface" begin
    n = 100
    A = sprand(n, n, 0.1) + 10I; A = A'A
    b1 = rand(n); b2 = rand(n)
    alg = PETScAlgorithm(:gmres; pc_type = :jacobi, comm = MPI.COMM_WORLD)

    cache = SciMLBase.init(LinearProblem(A, b1), alg; abstol = 1e-10)
    sol1 = solve!(cache)

    # No manual gather needed — solve! auto-gathers the full solution
    if rank == 0
        @test norm(A * sol1.u - b1) / norm(b1) < 1e-6
    end

    # Update b and re-solve (copy_into_vec! still available for manual updates)
    cache.b .= b2
    PETScExt.copy_into_vec!(cache.cacheval.petsc_b, cache.b, cache.cacheval.petsclib,
                   cache.cacheval.rstart, cache.cacheval.rend)

    sol2 = solve!(cache)

    if rank == 0
        @test norm(A * sol2.u - b2) / norm(b2) < 1e-6
    end

    PETScExt.cleanup_petsc_cache!(cache)
end
