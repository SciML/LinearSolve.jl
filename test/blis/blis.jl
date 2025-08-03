using LinearSolve, blis_jll, LAPACK_jll, LinearAlgebra, Test
using LinearSolve: BLISLUFactorization

@testset "BLIS + Reference LAPACK Tests" begin
    # Test basic functionality with multiple types
    test_types = [Float32, Float64, ComplexF32, ComplexF64]
    
    for T in test_types
        @testset "Type: $T" begin
            n = 100
            A = rand(T, n, n)
            b = rand(T, n)
            
            # Make A well-conditioned by adding diagonal dominance
            A += I * maximum(abs.(A)) * 0.1
            
            # Test BLIS LU factorization
            prob = LinearProblem(A, b)
            sol = solve(prob, BLISLUFactorization())
            
            # Check accuracy
            residual = norm(A * sol.u - b)
            tol = T <: Union{Float32, ComplexF32} ? 1e-3 : 1e-10
            @test residual < tol
            
            # Test multiple solves with same matrix
            cache = LinearSolve.init(prob, BLISLUFactorization())
            sol1 = solve!(cache)
            
            # Check the first solution
            residual1 = norm(A * sol1.u - b)
            @test residual1 < tol
            
            # Test with a different RHS vector
            b_new = rand(T, n)
            prob_new = LinearProblem(A, b_new)
            sol2 = solve(prob_new, BLISLUFactorization())
            
            residual2 = norm(A * sol2.u - b_new)
            @test residual2 < tol
            
            # Solutions should be different for different RHS
            @test norm(sol1.u - sol2.u) > 1e-6 || norm(b - b_new) < 1e-10
        end
    end
    
    @testset "Comparison with default solver" begin
        n = 50
        A = rand(Float64, n, n) + I * 0.1
        b = rand(Float64, n)
        
        prob = LinearProblem(A, b)
        
        # Solve with BLIS
        sol_blis = solve(prob, BLISLUFactorization())
        
        # Solve with default solver
        sol_default = solve(prob)
        
        # Both should give similar results
        @test norm(sol_blis.u - sol_default.u) < 1e-10
        
        # Both should satisfy the equation
        @test norm(A * sol_blis.u - b) < 1e-10
        @test norm(A * sol_default.u - b) < 1e-10
    end
    
    @testset "Matrix properties" begin
        # Test with different matrix structures
        n = 20
        
        # Symmetric matrix
        A_sym = randn(Float64, n, n)
        A_sym = A_sym + A_sym' + I * 0.1
        b = randn(Float64, n)
        
        prob_sym = LinearProblem(A_sym, b)
        sol_sym = solve(prob_sym, BLISLUFactorization())
        @test norm(A_sym * sol_sym.u - b) < 1e-10
        
        # Sparse matrix (converted to dense for BLIS)
        using SparseArrays
        A_sparse = sprand(Float64, n, n, 0.3) + I * 0.1
        A_dense = Matrix(A_sparse)
        
        prob_sparse = LinearProblem(A_dense, b)
        sol_sparse = solve(prob_sparse, BLISLUFactorization())
        @test norm(A_dense * sol_sparse.u - b) < 1e-10
    end
end