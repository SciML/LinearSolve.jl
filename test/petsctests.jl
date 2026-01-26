using LinearAlgebra
using LinearSolve
using MPI
using PETSc
using SparseArrays
using Test

MPI.Init()

# Get a PETSc library and initialize
petsclib = first(PETSc.petsclibs)
PETSc.initialized(petsclib) || PETSc.initialize(petsclib)

@testset "PETScAlgorithm Basic Tests" begin
    n = 100

    # Create a sparse positive definite matrix
    A = sprand(n, n, 0.05) + 10I
    A = A'A  # Make symmetric positive definite
    b = rand(n)

    prob = LinearProblem(A, b)

    @testset "GMRES solver" begin
        alg = PETScAlgorithm(:gmres)
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
    end

    @testset "CG solver" begin
        alg = PETScAlgorithm(:cg)
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
    end

    @testset "BiCGSTAB solver" begin
        alg = PETScAlgorithm(:bcgs)
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
    end

    @testset "GMRES with Jacobi preconditioner" begin
        alg = PETScAlgorithm(:gmres; pc_type = :jacobi)
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
    end

    @testset "GMRES with ILU preconditioner" begin
        alg = PETScAlgorithm(:gmres; pc_type = :ilu)
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
    end
end

@testset "PETScAlgorithm Dense Matrix" begin
    n = 50
    A = rand(n, n) + 10I
    A = A'A
    b = rand(n)

    prob = LinearProblem(A, b)
    alg = PETScAlgorithm(:gmres)
    sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
    @test norm(A * sol.u - b) / norm(b) < 1.0e-6
end

@testset "PETScAlgorithm Cache Interface" begin
    n = 50
    A = sprand(n, n, 0.1) + 10I
    A = A'A
    b1 = rand(n)

    prob = LinearProblem(A, b1)
    alg = PETScAlgorithm(:gmres)

    # Initialize cache
    cache = SciMLBase.init(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)

    # First solve
    sol1 = solve!(cache)
    @test norm(A * sol1.u - b1) / norm(b1) < 1.0e-6

    # Update b and solve again
    b2 = rand(n)
    cache.b = b2
    sol2 = solve!(cache)
    @test norm(A * sol2.u - b2) / norm(b2) < 1.0e-6
end

PETSc.finalize(petsclib)
