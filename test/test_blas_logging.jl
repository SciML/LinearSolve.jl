using LinearSolve
using LinearAlgebra
using Test
using SciMLLogging: Verbosity

@testset "BLAS Return Code Interpretation" begin
    # Test interpretation of various BLAS return codes
    @testset "Return Code Interpretation" begin
        # Test successful operation
        category, message, details = LinearSolve.interpret_blas_code(:dgetrf, 0)
        @test category == :success
        @test message == "Operation completed successfully"
        
        # Test invalid argument
        category, message, details = LinearSolve.interpret_blas_code(:dgetrf, -3)
        @test category == :invalid_argument
        @test occursin("Argument 3", details)
        
        # Test singular matrix in LU
        category, message, details = LinearSolve.interpret_blas_code(:dgetrf, 2)
        @test category == :singular_matrix
        @test occursin("U(2,2)", details)
        
        # Test not positive definite in Cholesky
        category, message, details = LinearSolve.interpret_blas_code(:dpotrf, 3)
        @test category == :not_positive_definite
        @test occursin("minor of order 3", details)
        
        # Test SVD convergence failure
        category, message, details = LinearSolve.interpret_blas_code(:dgesvd, 5)
        @test category == :convergence_failure
        @test occursin("5 off-diagonal", details)
    end
    
    @testset "BLAS Operation Info" begin
        # Test getting operation info without condition number
        A = rand(10, 10)
        b = rand(10)
        info = LinearSolve.get_blas_operation_info(:dgetrf, A, b)
        
        @test info[:matrix_size] == (10, 10)
        @test info[:element_type] == Float64
        @test !haskey(info, :condition_number)  # Should not compute by default
        @test info[:memory_usage_MB] >= 0  # Memory can be 0 for very small matrices
        
        # Test with condition number computation enabled
        info_with_cond = LinearSolve.get_blas_operation_info(:dgetrf, A, b; compute_condition=true)
        @test haskey(info_with_cond, :condition_number)
    end
    
    @testset "Verbosity Integration" begin
        # Test with singular matrix
        A = [1.0 2.0; 2.0 4.0]  # Singular matrix
        b = [1.0, 2.0]
        
        # Test with warnings enabled
        verbose = LinearVerbosity(
            blas_errors = Verbosity.Warn(),
            blas_info = Verbosity.None()
        )
        
        prob = LinearProblem(A, b)
        
        # This should fail due to singularity but not throw
        sol = solve(prob, LUFactorization(); verbose=verbose)
        @test sol.retcode == ReturnCode.Failure
        
        # Test with all logging enabled
        verbose_all = LinearVerbosity(
            blas_errors = Verbosity.Info(),
            blas_info = Verbosity.Info()
        )
        
        # Non-singular matrix for successful operation
        A_good = [1.0 0.0; 0.0 1.0]
        b_good = [1.0, 1.0]
        prob_good = LinearProblem(A_good, b_good)
        
        sol_good = solve(prob_good, LUFactorization(); verbose=verbose_all)
        @test sol_good.retcode == ReturnCode.Success
        @test sol_good.u ≈ b_good
    end
    
    @testset "Error Categories" begin
        # Test different error categories are properly identified
        test_cases = [
            (:dgetrf, 1, :singular_matrix),
            (:dpotrf, 2, :not_positive_definite),
            (:dgeqrf, 3, :numerical_issue),
            (:dgesdd, 4, :convergence_failure),
            (:dsyev, 5, :convergence_failure),
            (:dsytrf, 6, :singular_matrix),
            (:dgetrs, 1, :unexpected_error),
            (:unknown_func, 1, :unknown_error)
        ]
        
        for (func, code, expected_category) in test_cases
            category, _, _ = LinearSolve.interpret_blas_code(func, code)
            @test category == expected_category
        end
    end
end

