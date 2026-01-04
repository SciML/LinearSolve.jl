using LinearSolve, LinearAlgebra, Test
using Preferences

@testset "Dual Preference System Integration Tests" begin
    # Make preferences dynamic for testing verification
    LinearSolve.make_preferences_dynamic!()

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
            LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization,
        ]
        @test chosen_alg_no_ext.alg in standard_choices

        println("✅ Algorithm chosen without extensions: ", chosen_alg_no_ext.alg)

        # Test that the problem can be solved
        prob = LinearProblem(A, b)
        sol_no_ext = solve(prob)
        @test sol_no_ext.retcode == ReturnCode.Success
        @test norm(A * sol_no_ext.u - b) < 1.0e-8
    end

    @testset "FastLapack Extension Conditional Loading" begin
        # Test FastLapack loading conditionally and algorithm availability

        # Set preferences with GenericLU as always_loaded so it can be hit correctly
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_medium" => "FastLUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_medium" => "GenericLUFactorization"; force = true)

        # Verify preferences are set
        @test Preferences.load_preference(LinearSolve, "best_algorithm_Float64_medium", nothing) == "FastLUFactorization"
        @test Preferences.load_preference(LinearSolve, "best_always_loaded_Float64_medium", nothing) == "GenericLUFactorization"

        A = rand(Float64, 150, 150) + I(150)
        b = rand(Float64, 150)
        prob = LinearProblem(A, b)

        # Try to load FastLapackInterface and test FastLUFactorization
        fastlapack_loaded = false
        try
            @eval using FastLapackInterface

            # Test that FastLUFactorization works - only print if it fails
            sol_fast = solve(prob, FastLUFactorization())
            @test sol_fast.retcode == ReturnCode.Default
            @test norm(A * sol_fast.u - b) < 1.0e-8
            fastlapack_loaded = true
            # Success - no print needed

        catch e
            println("⚠️  FastLapackInterface/FastLUFactorization not available: ", e)
        end

        # Test algorithm choice (testing mode enabled at test start)
        chosen_alg_test = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))

        if fastlapack_loaded
            # If FastLapack loaded correctly and preferences are active, should choose LU (FastLU maps to LU)
            @test chosen_alg_test.alg === LinearSolve.DefaultAlgorithmChoice.LUFactorization
        else
            # Should choose GenericLUFactorization (always_loaded preference)
            @test chosen_alg_test.alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
        end

        sol_default = solve(prob)
        @test sol_default.retcode == ReturnCode.Success
        @test norm(A * sol_default.u - b) < 1.0e-8
    end

    @testset "RecursiveFactorization Extension Conditional Loading" begin
        # Clear all preferences first for this test
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

        # Set preferences for this test: RF as best, LU as always_loaded
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_small" => "RFLUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_small" => "LUFactorization"; force = true)

        # Verify preferences are set
        @test Preferences.load_preference(LinearSolve, "best_algorithm_Float64_small", nothing) == "RFLUFactorization"
        @test Preferences.load_preference(LinearSolve, "best_always_loaded_Float64_small", nothing) == "LUFactorization"

        A = rand(Float64, 80, 80) + I(80)  # Small category (21-100)
        b = rand(Float64, 80)
        prob = LinearProblem(A, b)

        # Try to load RecursiveFactorization and test RFLUFactorization
        recursive_loaded = false
        try
            @eval using RecursiveFactorization

            # Test that RFLUFactorization works - only print if it fails
            if LinearSolve.userecursivefactorization(A)
                sol_rf = solve(prob, RFLUFactorization())
                @test sol_rf.retcode == ReturnCode.Success
                @test norm(A * sol_rf.u - b) < 1.0e-8
                recursive_loaded = true
                # Success - no print needed
            end

        catch e
            println("⚠️  RecursiveFactorization/RFLUFactorization not available: ", e)
        end

        # Test algorithm choice with RecursiveFactorization available (testing mode enabled at test start)
        chosen_alg_with_rf = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))

        if recursive_loaded
            # If RecursiveFactorization loaded correctly and preferences are active, should choose RFLU
            @test chosen_alg_with_rf.alg === LinearSolve.DefaultAlgorithmChoice.RFLUFactorization
        else
            # Should choose LUFactorization (always_loaded preference)
            @test chosen_alg_with_rf.alg === LinearSolve.DefaultAlgorithmChoice.LUFactorization
        end

        sol_default_rf = solve(prob)
        @test sol_default_rf.retcode == ReturnCode.Success
        @test norm(A * sol_default_rf.u - b) < 1.0e-8
    end

    @testset "Algorithm Availability and Functionality Testing" begin
        # Test core algorithms that should always be available

        A = rand(Float64, 150, 150) + I(150)
        b = rand(Float64, 150)
        prob = LinearProblem(A, b)

        # Test core algorithms individually
        sol_lu = solve(prob, LUFactorization())
        @test sol_lu.retcode == ReturnCode.Success
        @test norm(A * sol_lu.u - b) < 1.0e-8
        println("✅ LUFactorization confirmed working")

        sol_generic = solve(prob, GenericLUFactorization())
        @test sol_generic.retcode == ReturnCode.Success
        @test norm(A * sol_generic.u - b) < 1.0e-8
        println("✅ GenericLUFactorization confirmed working")

        # Test MKL if available
        if LinearSolve.usemkl
            sol_mkl = solve(prob, MKLLUFactorization())
            @test sol_mkl.retcode == ReturnCode.Success
            @test norm(A * sol_mkl.u - b) < 1.0e-8
            println("✅ MKLLUFactorization confirmed working")
        end

        # Test OpenBLAS if available
        if LinearSolve.useopenblas
            sol_openblas = solve(prob, OpenBLASLUFactorization())
            @test sol_openblas.retcode == ReturnCode.Success
            @test norm(A * sol_openblas.u - b) < 1.0e-8
            println("✅ OpenBLASLUFactorization confirmed working")
        end

        # Test Apple Accelerate if available
        if LinearSolve.appleaccelerate_isavailable()
            sol_apple = solve(prob, AppleAccelerateLUFactorization())
            @test sol_apple.retcode == ReturnCode.Success
            @test norm(A * sol_apple.u - b) < 1.0e-8
            println("✅ AppleAccelerateLUFactorization confirmed working")
        end

        # Test RFLUFactorization if extension is loaded (requires RecursiveFactorization.jl)
        if LinearSolve.userecursivefactorization(A)
            try
                sol_rf = solve(prob, RFLUFactorization())
                @test sol_rf.retcode == ReturnCode.Success
                @test norm(A * sol_rf.u - b) < 1.0e-8
                # Success - no print needed (RFLUFactorization is extension-dependent)
            catch e
                println("⚠️  RFLUFactorization issue: ", e)
            end
        end
    end


    @testset "RFLU vs GenericLU Size Category Verification" begin
        # Test by setting one size to RFLU and all others to GenericLU
        # Rotate through each size category to verify preferences work correctly

        # Test cases: one size gets RFLU, others get GenericLU
        rflu_test_scenarios = [
            # (rflu_size, rflu_category, test_sizes_with_categories)
            (15, "tiny", [(50, "small"), (200, "medium"), (500, "large"), (1500, "big")]),
            (50, "small", [(15, "tiny"), (200, "medium"), (500, "large"), (1500, "big")]),
            (200, "medium", [(15, "tiny"), (50, "small"), (500, "large"), (1500, "big")]),
            (500, "large", [(15, "tiny"), (50, "small"), (200, "medium"), (1500, "big")]),
            (1500, "big", [(15, "tiny"), (50, "small"), (200, "medium"), (500, "large")]),
        ]

        for (rflu_size, rflu_category, other_test_sizes) in rflu_test_scenarios
            println("Testing RFLU at $(rflu_category) category (size $(rflu_size))")

            # Clear all preferences
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

            # Set RFLU for the target category
            Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_$(rflu_category)" => "RFLUFactorization"; force = true)
            Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_$(rflu_category)" => "RFLUFactorization"; force = true)

            # Set GenericLU for all other categories
            for other_category in size_categories
                if other_category != rflu_category
                    Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_$(other_category)" => "GenericLUFactorization"; force = true)
                    Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_$(other_category)" => "GenericLUFactorization"; force = true)
                end
            end

            # Test the RFLU size
            A_rflu = rand(Float64, rflu_size, rflu_size) + I(rflu_size)
            b_rflu = rand(Float64, rflu_size)
            chosen_rflu = LinearSolve.defaultalg(A_rflu, b_rflu, LinearSolve.OperatorAssumptions(true))

            if rflu_size <= 10
                # Tiny override should always choose GenericLU
                @test chosen_rflu.alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
                println("  ✅ Tiny override: size $(rflu_size) chose GenericLU (as expected)")
            else
                # Should choose RFLU based on preference
                @test chosen_rflu.alg === LinearSolve.DefaultAlgorithmChoice.RFLUFactorization
                println("  ✅ RFLU preference: size $(rflu_size) chose RFLUFactorization")
            end

            # Test other sizes should choose GenericLU
            for (other_size, other_category) in other_test_sizes
                A_other = rand(Float64, other_size, other_size) + I(other_size)
                b_other = rand(Float64, other_size)
                chosen_other = LinearSolve.defaultalg(A_other, b_other, LinearSolve.OperatorAssumptions(true))

                if other_size <= 10
                    # Tiny override
                    @test chosen_other.alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
                    println("  ✅ Tiny override: size $(other_size) chose GenericLU")
                else
                    # Should choose GenericLU based on preference
                    @test chosen_other.alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
                    println("  ✅ GenericLU preference: size $(other_size) chose GenericLUFactorization")
                end

                # Test that problems solve
                prob_other = LinearProblem(A_other, b_other)
                sol_other = solve(prob_other)
                @test sol_other.retcode == ReturnCode.Success
                @test norm(A_other * sol_other.u - b_other) < (other_size <= 10 ? 1.0e-12 : 1.0e-6)
            end

            # Test that RFLU size problem solves
            prob_rflu = LinearProblem(A_rflu, b_rflu)
            sol_rflu = solve(prob_rflu)
            @test sol_rflu.retcode == ReturnCode.Success
            @test norm(A_rflu * sol_rflu.u - b_rflu) < (rflu_size <= 10 ? 1.0e-12 : 1.0e-6)
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
