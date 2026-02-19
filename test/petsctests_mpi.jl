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
    # Ensure each rank owns a unique, non-overlapping range
    # e.g., for 2 ranks, Rank 0 owns 0-50, Rank 1 owns 50-100
    @test pcache.rend - pcache.rstart <= (n ÷ comm_size) + 1

    # Check that all ranks together cover the whole range [0, n)
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
    pcache = sol.cache.cacheval

    # Gather full solution on all ranks
    PETScExt.gather_solution!(sol.u, pcache.petsc_x, pcache; gather_all=true)

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

    pcache = sol.cache.cacheval

    # Gather full solution on all ranks
    PETScExt.gather_solution!(sol.u, pcache.petsc_x, pcache; gather_all=true)

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

    pcache = sol.cache.cacheval
    # Gather full solution on all ranks
    PETScExt.gather_solution!(sol.u, pcache.petsc_x, pcache; gather_all=true)

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

    # Gather solution on all ranks
    PETScExt.gather_solution!(sol1.u, cache.cacheval.petsc_x, cache.cacheval)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test norm(A * sol1.u - b1) / norm(b1) < 1e-6
    end

    # Update b and copy into PETSc distributed vector
    cache.b .= b2
    PETScExt.copy_into_vec!(cache.cacheval.petsc_b, cache.b, cache.cacheval.petsclib,
                   cache.cacheval.rstart, cache.cacheval.rend)

    sol2 = solve!(cache)
    PETScExt.gather_solution!(sol2.u, cache.cacheval.petsc_x, cache.cacheval)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test norm(A * sol2.u - b2) / norm(b2) < 1e-6
    end

    PETScExt.cleanup_petsc_cache!(cache)
end
# @testset "MPI (comm_size=$comm_size): Transposed Solve" begin
#     n = 100
#     A = sprand(n, n, 0.1) + 5I; b = rand(n)
#     alg = PETScAlgorithm(:gmres; transposed = true, comm = MPI.COMM_WORLD)
#     sol = solve(LinearProblem(A, b), alg; abstol = 1e-10, reltol = 1e-10)

#     @test norm(A' * sol.u - b) / norm(b) < 1e-6
#     PETScExt.cleanup_petsc_cache!(sol)
# end

# @testset "MPI (comm_size=$comm_size): Nullspace Constant" begin
#     n = 100
#     D = sparse(1:n, 1:n, 2.0, n, n)
#     D -= sparse(1:n-1, 2:n, 1.0, n, n)
#     D -= sparse(2:n, 1:n-1, 1.0, n, n)
#     D[1,1] = 1.0; D[end,end] = 1.0
#     b = rand(n); b .-= sum(b)/n

#     alg = PETScAlgorithm(:cg; pc_type = :jacobi, nullspace = :constant,
#                           comm = MPI.COMM_WORLD)
#     sol = solve(LinearProblem(D, b), alg; abstol = 1e-10)
#     @test sol.retcode == SciMLBase.ReturnCode.Success
#     @test norm(D * sol.u - b) / norm(b) < 1e-6
#     PETScExt.cleanup_petsc_cache!(sol)
# end

# @testset "MPI (comm_size=$comm_size): Warm Start" begin
#     n = 200
#     A = sprand(n, n, 0.02) + 10I; A = A'A; b = rand(n)
#     prob = LinearProblem(A, b)

#     # Cold start baseline
#     sol_cold = solve(prob,
#         PETScAlgorithm(:cg; initial_guess_nonzero = false, comm = MPI.COMM_WORLD);
#         reltol = 1e-12)
#     iters_cold = sol_cold.iters
#     PETScExt.cleanup_petsc_cache!(sol_cold)

#     # Warm start
#     alg_warm = PETScAlgorithm(:cg; initial_guess_nonzero = true, comm = MPI.COMM_WORLD)
#     cache_warm = SciMLBase.init(prob, alg_warm; reltol = 1e-12)
#     solve!(cache_warm)

#     cache_warm.b = b + rand(n) * 0.01
#     sol_warm = solve!(cache_warm)

#     @test sol_warm.iters < iters_cold
#     PETScExt.cleanup_petsc_cache!(cache_warm)
# end

# @testset "MPI (comm_size=$comm_size): Cleanup" begin
#     n = 50
#     A = sprand(n, n, 0.1) + 10I; A = A'A; b = rand(n)
#     alg = PETScAlgorithm(:gmres; pc_type = :jacobi, comm = MPI.COMM_WORLD)

#     sol = solve(LinearProblem(A, b), alg)
#     pcache = sol.cache.cacheval
#     @test pcache.ksp !== nothing
#     @test pcache.comm !== nothing

#     PETScExt.cleanup_petsc_cache!(sol)
#     @test pcache.ksp === nothing
#     @test pcache.petsc_A === nothing

#     # Idempotent
#     PETScExt.cleanup_petsc_cache!(sol)
#     @test pcache.ksp === nothing
# end

# @testset "MPI (comm_size=$comm_size): Serial fallback (comm=nothing)" begin
#     # Verify that comm=nothing still works identically to the old serial code
#     n = 50
#     A = sprand(n, n, 0.1) + 10I; A = A'A; b = rand(n)
#     sol = solve(LinearProblem(A, b), PETScAlgorithm(:gmres); abstol = 1e-10)
#     @test norm(A * sol.u - b) / norm(b) < 1e-6
#     PETScExt.cleanup_petsc_cache!(sol)
# end
