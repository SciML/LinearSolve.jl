using LinearSolve, RecursiveFactorization, LinearAlgebra, SparseArrays, Test

@test LinearSolve.defaultalg(nothing, zeros(3)).alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
prob = LinearProblem(rand(3, 3), rand(3))
solve(prob)

if LinearSolve.appleaccelerate_isavailable()
    @test LinearSolve.defaultalg(nothing, zeros(50)).alg ===
          LinearSolve.DefaultAlgorithmChoice.AppleAccelerateLUFactorization
else
    @test LinearSolve.defaultalg(nothing, zeros(50)).alg ===
          LinearSolve.DefaultAlgorithmChoice.RFLUFactorization
end
prob = LinearProblem(rand(50, 50), rand(50))
solve(prob)

if LinearSolve.usemkl
    @test LinearSolve.defaultalg(nothing, zeros(600)).alg ===
          LinearSolve.DefaultAlgorithmChoice.MKLLUFactorization
elseif LinearSolve.appleaccelerate_isavailable()
    @test LinearSolve.defaultalg(nothing, zeros(600)).alg ===
          LinearSolve.DefaultAlgorithmChoice.AppleAccelerateLUFactorization
else
    @test LinearSolve.defaultalg(nothing, zeros(600)).alg ===
          LinearSolve.DefaultAlgorithmChoice.LUFactorization
end

prob = LinearProblem(rand(600, 600), rand(600))
solve(prob)

@test LinearSolve.defaultalg(LinearAlgebra.Diagonal(zeros(5)), zeros(5)).alg ===
      LinearSolve.DefaultAlgorithmChoice.DiagonalFactorization

@test LinearSolve.defaultalg(nothing, zeros(5),
    LinearSolve.OperatorAssumptions(false)).alg ===
      LinearSolve.DefaultAlgorithmChoice.QRFactorization

A = spzeros(2, 2)
# test that solving a singular problem doesn't error
prob = LinearProblem(A, ones(2))
@test solve(prob, UMFPACKFactorization()).retcode == ReturnCode.Infeasible
@test solve(prob, KLUFactorization()).retcode == ReturnCode.Infeasible

@test LinearSolve.defaultalg(sprand(10^4, 10^4, 1e-5) + I, zeros(1000)).alg ===
      LinearSolve.DefaultAlgorithmChoice.KLUFactorization
prob = LinearProblem(sprand(1000, 1000, 0.5), zeros(1000))
solve(prob)

@test LinearSolve.defaultalg(sprand(11000, 11000, 0.001), zeros(11000)).alg ===
      LinearSolve.DefaultAlgorithmChoice.UMFPACKFactorization
prob = LinearProblem(sprand(11000, 11000, 0.5), zeros(11000))
solve(prob)

# Test inference
A = rand(4, 4)
b = rand(4)
prob = LinearProblem(A, b)
prob = LinearProblem(sparse(A), b)
@inferred solve(prob)
@inferred init(prob, nothing)

prob = LinearProblem(big.(rand(10, 10)), big.(zeros(10)))
solve(prob)

## Operator defaults
## https://github.com/SciML/LinearSolve.jl/issues/414

m, n = 2, 2
A = rand(m, n)
b = rand(m)
x = rand(n)
f = (w, v, u, p, t) -> mul!(w, A, v)
fadj = (w, v, u, p, t) -> mul!(w, A', v)
funcop = FunctionOperator(f, x, b; op_adjoint = fadj)
prob = LinearProblem(funcop, b)
sol1 = solve(prob)
sol2 = solve(prob, LinearSolve.KrylovJL_GMRES())
@test sol1.u == sol2.u

m, n = 3, 2
A = rand(m, n)
b = rand(m)
x = rand(n)
f = (w, v, u, p, t) -> mul!(w, A, v)
fadj = (w, v, u, p, t) -> mul!(w, A', v)
funcop = FunctionOperator(f, x, b; op_adjoint = fadj)
prob = LinearProblem(funcop, b)
sol1 = solve(prob)
sol2 = solve(prob, LinearSolve.KrylovJL_LSMR())
@test sol1.u == sol2.u

m, n = 2, 3
A = rand(m, n)
b = rand(m)
x = rand(n)
f = (w, v, u, p, t) -> mul!(w, A, v)
fadj = (w, v, u, p, t) -> mul!(w, A', v)
funcop = FunctionOperator(f, x, b; op_adjoint = fadj)
prob = LinearProblem(funcop, b)
sol1 = solve(prob)
sol2 = solve(prob, LinearSolve.KrylovJL_CRAIGMR())
@test sol1.u == sol2.u

# Default for Underdetermined problem but the size is a long rectangle
A = [2.0 1.0
     0.0 0.0
     0.0 0.0]
b = [1.0, 0.0, 0.0]
prob = LinearProblem(A, b)
sol = solve(prob)

@test !SciMLBase.successful_retcode(sol.retcode)

## Show that we cannot select a default alg once by checking the rank, since it might change
## later in the cache
## Common occurrence for iterative nonlinear solvers using linear solve
A = [2.0 1.0
     1.0 1.0
     0.0 0.0]
b = [1.0, 1.0, 0.0]
prob = LinearProblem(A, b)

cache = init(prob)
sol = solve!(cache)

@test sol.u ≈ [0.0, 1.0]

cache.A = [2.0 1.0
           0.0 0.0
           0.0 0.0]

sol = solve!(cache)

@test !SciMLBase.successful_retcode(sol.retcode)

## Non-square Sparse Defaults 
# https://github.com/SciML/NonlinearSolve.jl/issues/599
A = SparseMatrixCSC{Float64, Int64}([1.0 0.0
                                     1.0 1.0])
b = ones(2)
A2 = hcat(A, A)
prob = LinearProblem(A, b)
@test SciMLBase.successful_retcode(solve(prob))

prob2 = LinearProblem(A2, b)
@test SciMLBase.successful_retcode(solve(prob2))

A = SparseMatrixCSC{Float64, Int32}([1.0 0.0
                                     1.0 1.0])
b = ones(2)
A2 = hcat(A, A)
prob = LinearProblem(A, b)
@test_broken SciMLBase.successful_retcode(solve(prob))

prob2 = LinearProblem(A2, b)
@test SciMLBase.successful_retcode(solve(prob2))

# Column-Pivoted QR fallback on failed LU
A = [1.0 0 0 0
     0 1 0 0
     0 0 1 0
     0 0 0 0]
b = rand(4)
prob = LinearProblem(A, b)
sol = solve(prob,
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.LUFactorization; safetyfallback = false))
@test sol.retcode === ReturnCode.Failure
@test sol.u == zeros(4)

sol = solve(prob)
@test sol.u ≈ svd(A)\b

# Test that dual preference system integration works correctly
@testset "Autotune Dual Preference System Integration" begin
    using Preferences
    
    # Clear any existing preferences
    target_eltypes = ["Float32", "Float64", "ComplexF32", "ComplexF64"]
    size_categories = ["tiny", "small", "medium", "large", "big"]
    
    for eltype in target_eltypes
        for size_cat in size_categories
            for pref_type in ["best_algorithm", "best_always_loaded"]
                pref_key = "$(pref_type)_$(eltype)_$(size_cat)"
                if Preferences.has_preference(LinearSolve, pref_key)
                    Preferences.delete_preferences!(LinearSolve, pref_key; force = true)
                end
            end
        end
    end
    
    @testset "Dual Preference Storage and Retrieval" begin
        # Test that we can store and retrieve both types of preferences
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_medium" => "RFLUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_medium" => "MKLLUFactorization"; force = true)
        
        # Verify preference storage is correct
        @test Preferences.load_preference(LinearSolve, "best_algorithm_Float64_medium", nothing) == "RFLUFactorization"
        @test Preferences.load_preference(LinearSolve, "best_always_loaded_Float64_medium", nothing) == "MKLLUFactorization"
        
        # Test with different element types and sizes
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float32_small" => "LUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float32_small" => "LUFactorization"; force = true)
        
        @test Preferences.load_preference(LinearSolve, "best_algorithm_Float32_small", nothing) == "LUFactorization"
        @test Preferences.load_preference(LinearSolve, "best_always_loaded_Float32_small", nothing) == "LUFactorization"
    end
    
    @testset "Default Algorithm Selection with Dual Preferences" begin
        # Test that default solver works correctly when preferences are set
        # This verifies the infrastructure is ready for the preference integration
        
        test_scenarios = [
            (Float64, 150, "RFLUFactorization", "LUFactorization"),    # medium size
            (Float32, 80, "LUFactorization", "LUFactorization"),       # small size  
            (ComplexF64, 100, "LUFactorization", "LUFactorization")    # small size, conservative
        ]
        
        for (eltype, matrix_size, best_alg, fallback_alg) in test_scenarios
            # Determine size category for preferences
            size_category = if matrix_size <= 128
                "small"
            elseif matrix_size <= 256  
                "medium"
            elseif matrix_size <= 512
                "large"
            else
                "big"
            end
            
            # Set preferences for this scenario
            eltype_str = string(eltype)
            Preferences.set_preferences!(LinearSolve, "best_algorithm_$(eltype_str)_$(size_category)" => best_alg; force = true)
            Preferences.set_preferences!(LinearSolve, "best_always_loaded_$(eltype_str)_$(size_category)" => fallback_alg; force = true)
            
            # Verify preferences are stored correctly
            @test Preferences.has_preference(LinearSolve, "best_algorithm_$(eltype_str)_$(size_category)")
            @test Preferences.has_preference(LinearSolve, "best_always_loaded_$(eltype_str)_$(size_category)")
            
            stored_best = Preferences.load_preference(LinearSolve, "best_algorithm_$(eltype_str)_$(size_category)", nothing)
            stored_fallback = Preferences.load_preference(LinearSolve, "best_always_loaded_$(eltype_str)_$(size_category)", nothing)
            
            @test stored_best == best_alg
            @test stored_fallback == fallback_alg
            
            # Create test problem and verify it can be solved
            A = rand(eltype, matrix_size, matrix_size) + I(matrix_size)
            b = rand(eltype, matrix_size)
            prob = LinearProblem(A, b)
            
            # Test that default solver works (infrastructure ready for preference integration)
            sol = solve(prob)
            @test sol.retcode == ReturnCode.Success
            @test norm(A * sol.u - b) < (eltype <: AbstractFloat ? 1e-6 : 1e-8)
            
            # Test that preferred algorithms work individually
            if best_alg == "LUFactorization"
                sol_best = solve(prob, LUFactorization())
                @test sol_best.retcode == ReturnCode.Success
                @test norm(A * sol_best.u - b) < (eltype <: AbstractFloat ? 1e-6 : 1e-8)
            elseif best_alg == "RFLUFactorization" && LinearSolve.userecursivefactorization(A)
                sol_best = solve(prob, RFLUFactorization())
                @test sol_best.retcode == ReturnCode.Success
                @test norm(A * sol_best.u - b) < (eltype <: AbstractFloat ? 1e-6 : 1e-8)
            end
            
            if fallback_alg == "LUFactorization"
                sol_fallback = solve(prob, LUFactorization())
                @test sol_fallback.retcode == ReturnCode.Success
                @test norm(A * sol_fallback.u - b) < (eltype <: AbstractFloat ? 1e-6 : 1e-8)
            end
        end
    end
    
    @testset "Preference System Robustness" begin
        # Test that default solver remains robust with invalid preferences
        
        # Set invalid preferences
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_medium" => "NonExistentAlgorithm"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_medium" => "AnotherNonExistentAlgorithm"; force = true)
        
        # Create test problem
        A = rand(Float64, 150, 150) + I(150)
        b = rand(Float64, 150)
        prob = LinearProblem(A, b)
        
        # Should still solve successfully using existing heuristics
        sol = solve(prob)
        @test sol.retcode == ReturnCode.Success
        @test norm(A * sol.u - b) < 1e-8
        
        # Test that preference infrastructure doesn't break default behavior
        @test Preferences.has_preference(LinearSolve, "best_algorithm_Float64_medium")
        @test Preferences.has_preference(LinearSolve, "best_always_loaded_Float64_medium")
    end
    
    # Clean up all test preferences
    for eltype in target_eltypes
        for size_cat in size_categories
            for pref_type in ["best_algorithm", "best_always_loaded"]
                pref_key = "$(pref_type)_$(eltype)_$(size_cat)"
                if Preferences.has_preference(LinearSolve, pref_key)
                    Preferences.delete_preferences!(LinearSolve, pref_key; force = true)
                end
            end
        end
    end
end
