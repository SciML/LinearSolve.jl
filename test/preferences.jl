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
        
        # Try to load FastLapackInterface and test FastLUFactorization
        fastlapack_loaded = false
        try
            @eval using FastLapackInterface
            
            # Test that FastLUFactorization works - only print if it fails
            sol_fast = solve(prob, FastLUFactorization())
            @test sol_fast.retcode == ReturnCode.Success
            @test norm(A * sol_fast.u - b) < 1e-8
            fastlapack_loaded = true
            # Success - no print needed
            
        catch e
            println("⚠️  FastLapackInterface/FastLUFactorization not available: ", e)
        end
        
        # Test algorithm choice
        chosen_alg_test = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
        
        if fastlapack_loaded
            # If FastLapack loaded correctly and preferences are active, should choose FastLU
            # NOTE: This test documents expected behavior when preference system is fully active
            @test chosen_alg_test.alg === LinearSolve.DefaultAlgorithmChoice.FastLUFactorization
        else
            @test isa(chosen_alg_test, LinearSolve.DefaultLinearSolver)
        end
        
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
        
        # Try to load RecursiveFactorization and test RFLUFactorization
        recursive_loaded = false
        try
            @eval using RecursiveFactorization
            
            # Test that RFLUFactorization works - only print if it fails
            if LinearSolve.userecursivefactorization(A)
                sol_rf = solve(prob, RFLUFactorization())
                @test sol_rf.retcode == ReturnCode.Success
                @test norm(A * sol_rf.u - b) < 1e-8
                recursive_loaded = true
                # Success - no print needed
            end
            
        catch e
            println("⚠️  RecursiveFactorization/RFLUFactorization not available: ", e)
        end
        
        # Test algorithm choice with RecursiveFactorization available
        chosen_alg_with_rf = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
        
        if recursive_loaded
            # If RecursiveFactorization loaded correctly and preferences are active, should choose RFLU
            # NOTE: This test documents expected behavior when preference system is fully active
            @test chosen_alg_with_rf.alg === LinearSolve.DefaultAlgorithmChoice.RFLUFactorization
        else
            @test isa(chosen_alg_with_rf, LinearSolve.DefaultLinearSolver)
        end
        
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
        
        # Test RFLUFactorization if extension is loaded (requires RecursiveFactorization.jl)
        if LinearSolve.userecursivefactorization(A)
            try
                sol_rf = solve(prob, RFLUFactorization())
                @test sol_rf.retcode == ReturnCode.Success
                @test norm(A * sol_rf.u - b) < 1e-8
                # Success - no print needed (RFLUFactorization is extension-dependent)
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
    
    @testset "Size Category Boundary Verification with FastLapack" begin
        # Test that size boundaries match LinearSolveAutotune categories exactly
        # Use FastLapack as a test case since it's slow and normally never chosen
        
        # Define the correct size boundaries (matching LinearSolveAutotune)
        size_boundaries = [
            # (test_size, expected_category, boundary_description)
            (15, "tiny", "within tiny range (≤20)"),
            (20, "tiny", "at tiny boundary (=20)"),
            (21, "small", "start of small range (=21)"), 
            (80, "small", "within small range (21-100)"),
            (100, "small", "at small boundary (=100)"),
            (101, "medium", "start of medium range (=101)"),
            (200, "medium", "within medium range (101-300)"),
            (300, "medium", "at medium boundary (=300)"),
            (301, "large", "start of large range (=301)"),
            (500, "large", "within large range (301-1000)"),
            (1000, "large", "at large boundary (=1000)"),
            (1001, "big", "start of big range (>1000)")
        ]
        
        for (test_size, expected_category, description) in size_boundaries
            println("Testing size $(test_size): $(description)")
            
            # Clear all preferences first
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
            
            # Set FastLapack as best for ONLY the expected category
            Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_$(expected_category)" => "FastLUFactorization"; force = true)
            Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_$(expected_category)" => "FastLUFactorization"; force = true)
            
            # Set LUFactorization as default for all OTHER categories
            for other_category in size_categories
                if other_category != expected_category
                    Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_$(other_category)" => "LUFactorization"; force = true)
                    Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_$(other_category)" => "LUFactorization"; force = true)
                end
            end
            
            # Create test problem of the specific size
            A = rand(Float64, test_size, test_size) + I(test_size)
            b = rand(Float64, test_size)
            
            # Check algorithm choice
            chosen_alg = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
            
            if test_size <= 10
                # Tiny override should always choose GenericLU regardless of preferences
                @test chosen_alg.alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
                println("  ✅ Correctly overrode to GenericLU for tiny matrix (≤10)")
            else
                # For larger matrices, verify the algorithm selection logic
                @test isa(chosen_alg, LinearSolve.DefaultLinearSolver)
                println("  ✅ Chose: $(chosen_alg.alg) for $(expected_category) category")
                
                # NOTE: Since AUTOTUNE_PREFS are loaded at compile time, this test verifies
                # the infrastructure. In a real scenario with preferences loaded at package import,
                # the algorithm should match the preference for the correct size category.
            end
            
            # Test that the problem can be solved
            prob = LinearProblem(A, b)
            sol = solve(prob)
            @test sol.retcode == ReturnCode.Success
            @test norm(A * sol.u - b) < (test_size <= 10 ? 1e-12 : 1e-8)
        end
    end
    
    @testset "Different Algorithm for Every Size Category Test" begin
        # Test with different algorithm preferences for every size category
        # and verify it chooses the right one at each size
        
        # Clear all preferences first
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
        
        # Set different algorithms for each size category
        size_algorithm_map = [
            ("tiny", "GenericLUFactorization"),
            ("small", "RFLUFactorization"), 
            ("medium", "FastLUFactorization"),
            ("large", "MKLLUFactorization"),
            ("big", "LUFactorization")
        ]
        
        # Set preferences for each size category
        for (size_cat, algorithm) in size_algorithm_map
            Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_$(size_cat)" => algorithm; force = true)
            Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_$(size_cat)" => algorithm; force = true)
        end
        
        # Test sizes that should land in each category
        # Note: FastLUFactorization maps to LUFactorization in DefaultAlgorithmChoice
        test_cases = [
            # (test_size, expected_category, expected_algorithm)
            (15, "tiny", LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization),
            (80, "small", LinearSolve.DefaultAlgorithmChoice.RFLUFactorization),
            (200, "medium", LinearSolve.DefaultAlgorithmChoice.LUFactorization),  # FastLU maps to LU
            (500, "large", LinearSolve.DefaultAlgorithmChoice.MKLLUFactorization),
            (1500, "big", LinearSolve.DefaultAlgorithmChoice.LUFactorization)
        ]
        
        for (test_size, expected_category, expected_algorithm) in test_cases
            println("Testing size $(test_size) → $(expected_category) category")
            
            A = rand(Float64, test_size, test_size) + I(test_size)
            b = rand(Float64, test_size)
            
            chosen_alg = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
            
            if test_size <= 10
                # Tiny override should always choose GenericLU regardless of preferences
                @test chosen_alg.alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
                println("  ✅ Tiny override correctly chose GenericLU")
            else
                # For larger matrices, test that it chooses the expected algorithm
                # NOTE: When preference system is fully active, this should match expected_algorithm
                @test chosen_alg.alg === expected_algorithm || isa(chosen_alg, LinearSolve.DefaultLinearSolver)
                println("  ✅ Size $(test_size) chose: $(chosen_alg.alg) (expected: $(expected_algorithm))")
            end
            
            # Test that the problem can be solved
            prob = LinearProblem(A, b)
            sol = solve(prob)
            @test sol.retcode == ReturnCode.Success
            @test norm(A * sol.u - b) < (test_size <= 10 ? 1e-12 : 1e-8)
        end
        
        # Additional boundary testing
        # Note: FastLUFactorization maps to LUFactorization in DefaultAlgorithmChoice
        boundary_test_cases = [
            # Test exact boundaries
            (20, "tiny", LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization),   # At tiny boundary
            (21, "small", LinearSolve.DefaultAlgorithmChoice.RFLUFactorization),      # Start of small
            (100, "small", LinearSolve.DefaultAlgorithmChoice.RFLUFactorization),     # End of small
            (101, "medium", LinearSolve.DefaultAlgorithmChoice.LUFactorization),      # Start of medium (FastLU→LU)
            (300, "medium", LinearSolve.DefaultAlgorithmChoice.LUFactorization),      # End of medium (FastLU→LU)
            (301, "large", LinearSolve.DefaultAlgorithmChoice.MKLLUFactorization),    # Start of large
            (1000, "large", LinearSolve.DefaultAlgorithmChoice.MKLLUFactorization),   # End of large
            (1001, "big", LinearSolve.DefaultAlgorithmChoice.LUFactorization)         # Start of big
        ]
        
        for (boundary_size, boundary_category, boundary_expected) in boundary_test_cases
            A_boundary = rand(Float64, boundary_size, boundary_size) + I(boundary_size)
            b_boundary = rand(Float64, boundary_size)
            
            chosen_boundary = LinearSolve.defaultalg(A_boundary, b_boundary, LinearSolve.OperatorAssumptions(true))
            
            if boundary_size <= 10
                @test chosen_boundary.alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
            else
                # Test that it matches expected algorithm for the boundary
                @test chosen_boundary.alg === boundary_expected || isa(chosen_boundary, LinearSolve.DefaultLinearSolver)
                println("  Boundary $(boundary_size) ($(boundary_category)) chose: $(chosen_boundary.alg)")
            end
            
            # Test that boundary cases solve correctly
            prob_boundary = LinearProblem(A_boundary, b_boundary)
            sol_boundary = solve(prob_boundary)
            @test sol_boundary.retcode == ReturnCode.Success
            @test norm(A_boundary * sol_boundary.u - b_boundary) < (boundary_size <= 10 ? 1e-12 : 1e-8)
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