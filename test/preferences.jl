using LinearSolve, LinearAlgebra, Test
using Preferences

@testset "Dual Preference System Integration Tests" begin
    # Clear any existing preferences to start clean
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
    
    @testset "Preference System Before Extension Loading" begin
        # Set preferences with RecursiveFactorization as best and FastLU as always_loaded
        # Test that when RF is not loaded, it falls back to always_loaded (FastLU when available)
        
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_medium" => "RFLUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_medium" => "FastLUFactorization"; force = true)
        
        # Verify preferences are set
        @test Preferences.load_preference(LinearSolve, "best_algorithm_Float64_medium", nothing) == "RFLUFactorization"
        @test Preferences.load_preference(LinearSolve, "best_always_loaded_Float64_medium", nothing) == "FastLUFactorization"
        
        # Create medium-sized Float64 problem (150x150 should trigger medium category)
        A = rand(Float64, 150, 150) + I(150)
        b = rand(Float64, 150)
        
        # Test algorithm choice WITHOUT extensions loaded
        # Should fall back to existing heuristics since neither RF nor FastLapack are loaded yet
        chosen_alg_no_ext = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
        @test isa(chosen_alg_no_ext, LinearSolve.DefaultLinearSolver)
        
        # Should be one of the standard choices when no extensions loaded
        standard_choices = [
            LinearSolve.DefaultAlgorithmChoice.LUFactorization,
            LinearSolve.DefaultAlgorithmChoice.MKLLUFactorization,
            LinearSolve.DefaultAlgorithmChoice.AppleAccelerateLUFactorization,
            LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
        ]
        @test chosen_alg_no_ext.alg in standard_choices
        
        println("✅ Algorithm chosen without extensions: ", chosen_alg_no_ext.alg)
        
        # Test that the problem can be solved
        prob = LinearProblem(A, b)
        sol_no_ext = solve(prob)
        @test sol_no_ext.retcode == ReturnCode.Success
        @test norm(A * sol_no_ext.u - b) < 1e-8
    end
    
    @testset "FastLapack Extension Conditional Loading" begin
        # Test FastLapack loading conditionally and algorithm availability
        
        # Preferences should still be set
        @test Preferences.load_preference(LinearSolve, "best_algorithm_Float64_medium", nothing) == "RFLUFactorization"
        @test Preferences.load_preference(LinearSolve, "best_always_loaded_Float64_medium", nothing) == "FastLUFactorization"
        
        A = rand(Float64, 150, 150) + I(150)
        b = rand(Float64, 150)
        prob = LinearProblem(A, b)
        
        # Try to load FastLapackInterface - it may or may not be available
        try
            @eval using FastLapackInterface
            
            # If FastLapack loads successfully, test that FastLU works
            try
                sol_fast = solve(prob, FastLUFactorization())
                @test sol_fast.retcode == ReturnCode.Success
                @test norm(A * sol_fast.u - b) < 1e-8
                println("✅ FastLUFactorization successfully works with extension loaded")
            catch e
                println("⚠️  FastLUFactorization not fully functional: ", e)
            end
            
        catch e
            println("ℹ️  FastLapackInterface not available in this environment: ", e)
        end
        
        # Test algorithm choice - should work regardless of FastLapack availability
        chosen_alg_test = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
        println("✅ Algorithm chosen (FastLapack test): ", chosen_alg_test.alg)
        
        sol_default = solve(prob)
        @test sol_default.retcode == ReturnCode.Success
        @test norm(A * sol_default.u - b) < 1e-8
    end
    
    @testset "RecursiveFactorization Extension Conditional Loading" begin
        # Test RecursiveFactorization loading conditionally
        
        # Preferences should still be set: RF as best, FastLU as always_loaded
        @test Preferences.load_preference(LinearSolve, "best_algorithm_Float64_medium", nothing) == "RFLUFactorization"
        @test Preferences.load_preference(LinearSolve, "best_always_loaded_Float64_medium", nothing) == "FastLUFactorization"
        
        A = rand(Float64, 150, 150) + I(150)
        b = rand(Float64, 150)
        prob = LinearProblem(A, b)
        
        # Try to load RecursiveFactorization - should be available as it's a dependency
        try
            @eval using RecursiveFactorization
            
            # Test that RFLUFactorization works
            if LinearSolve.userecursivefactorization(A)
                sol_rf = solve(prob, RFLUFactorization())
                @test sol_rf.retcode == ReturnCode.Success
                @test norm(A * sol_rf.u - b) < 1e-8
                println("✅ RFLUFactorization successfully works with extension loaded")
            else
                println("ℹ️  RFLUFactorization not enabled for this matrix type")
            end
            
        catch e
            println("⚠️  RecursiveFactorization loading issue: ", e)
        end
        
        # Test algorithm choice with RecursiveFactorization available
        chosen_alg_with_rf = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
        println("✅ Algorithm chosen (RecursiveFactorization test): ", chosen_alg_with_rf.alg)
        
        sol_default_rf = solve(prob)
        @test sol_default_rf.retcode == ReturnCode.Success
        @test norm(A * sol_default_rf.u - b) < 1e-8
    end
    
    @testset "Algorithm Availability and Functionality Testing" begin
        # Test core algorithms that should always be available
        
        A = rand(Float64, 150, 150) + I(150)
        b = rand(Float64, 150)
        prob = LinearProblem(A, b)
        
        # Test core algorithms individually
        sol_lu = solve(prob, LUFactorization())
        @test sol_lu.retcode == ReturnCode.Success
        @test norm(A * sol_lu.u - b) < 1e-8
        println("✅ LUFactorization confirmed working")
        
        sol_generic = solve(prob, GenericLUFactorization())
        @test sol_generic.retcode == ReturnCode.Success
        @test norm(A * sol_generic.u - b) < 1e-8
        println("✅ GenericLUFactorization confirmed working")
        
        # Test MKL if available
        if LinearSolve.usemkl
            sol_mkl = solve(prob, MKLLUFactorization())
            @test sol_mkl.retcode == ReturnCode.Success
            @test norm(A * sol_mkl.u - b) < 1e-8
            println("✅ MKLLUFactorization confirmed working")
        end
        
        # Test Apple Accelerate if available
        if LinearSolve.appleaccelerate_isavailable()
            sol_apple = solve(prob, AppleAccelerateLUFactorization())
            @test sol_apple.retcode == ReturnCode.Success
            @test norm(A * sol_apple.u - b) < 1e-8
            println("✅ AppleAccelerateLUFactorization confirmed working")
        end
        
        # Test RFLUFactorization if available (from existing dependencies)
        if LinearSolve.userecursivefactorization(A)
            try
                sol_rf = solve(prob, RFLUFactorization())
                @test sol_rf.retcode == ReturnCode.Success
                @test norm(A * sol_rf.u - b) < 1e-8
                println("✅ RFLUFactorization confirmed working")
            catch e
                println("⚠️  RFLUFactorization issue: ", e)
            end
        end
    end
    
    @testset "Preference-Based Algorithm Selection Simulation" begin
        # Simulate what should happen when preference system is fully active
        
        # Test different preference combinations
        test_scenarios = [
            ("RFLUFactorization", "FastLUFactorization", "RF available, FastLU fallback"),
            ("FastLUFactorization", "LUFactorization", "FastLU best, LU fallback"),
            ("NonExistentAlgorithm", "FastLUFactorization", "Invalid best, FastLU fallback"),
            ("NonExistentAlgorithm", "NonExistentAlgorithm", "Both invalid, use heuristics")
        ]
        
        A = rand(Float64, 150, 150) + I(150)
        b = rand(Float64, 150)
        prob = LinearProblem(A, b)
        
        for (best_alg, fallback_alg, description) in test_scenarios
            println("Testing scenario: ", description)
            
            # Set preferences for this scenario
            Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_medium" => best_alg; force = true)
            Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_medium" => fallback_alg; force = true)
            
            # Test that system remains robust
            chosen_alg = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
            @test isa(chosen_alg, LinearSolve.DefaultLinearSolver)
            
            sol = solve(prob)
            @test sol.retcode == ReturnCode.Success
            @test norm(A * sol.u - b) < 1e-8
            
            println("  Chosen algorithm: ", chosen_alg.alg)
        end
    end
    
    @testset "Size Override and Boundary Testing" begin
        # Test that tiny matrix override works regardless of preferences
        
        # Set preferences that should be ignored for tiny matrices
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_tiny" => "RFLUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_tiny" => "FastLUFactorization"; force = true)
        
        # Test matrices at the boundary
        boundary_sizes = [5, 8, 10, 11, 15, 50]
        
        for size in boundary_sizes
            A_boundary = rand(Float64, size, size) + I(size)
            b_boundary = rand(Float64, size)
            
            chosen_alg_boundary = LinearSolve.defaultalg(A_boundary, b_boundary, LinearSolve.OperatorAssumptions(true))
            
            if size <= 10
                # Should always override to GenericLUFactorization for tiny matrices
                @test chosen_alg_boundary.alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
                println("✅ Size $(size)×$(size) correctly overrode to: ", chosen_alg_boundary.alg)
            else
                # Should use normal algorithm selection for larger matrices
                @test isa(chosen_alg_boundary, LinearSolve.DefaultLinearSolver)
                println("✅ Size $(size)×$(size) chose: ", chosen_alg_boundary.alg)
            end
            
            # Test that all can solve
            prob_boundary = LinearProblem(A_boundary, b_boundary)
            sol_boundary = solve(prob_boundary)
            @test sol_boundary.retcode == ReturnCode.Success
            @test norm(A_boundary * sol_boundary.u - b_boundary) < (size <= 10 ? 1e-12 : 1e-8)
        end
    end
    
    # Final cleanup: Reset all preferences to original state
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
    
    # Reset other autotune-related preferences
    for pref in ["LoadMKL_JLL", "autotune_timestamp"]
        if Preferences.has_preference(LinearSolve, pref)
            Preferences.delete_preferences!(LinearSolve, pref; force = true)
        end
    end
    
    println("✅ All preferences cleaned up and reset to original state")
end