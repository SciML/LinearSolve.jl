using Test
using LinearSolve

if isempty(VERSION.prerelease)
    using LinearSolveAutotune
    using DataFrames
    using Random

    @testset "LinearSolveAutotune.jl Tests" begin

        @testset "Algorithm Detection" begin
            cpu_algs, cpu_names = LinearSolveAutotune.get_available_algorithms()
            @test !isempty(cpu_algs)
            @test !isempty(cpu_names)
            @test length(cpu_algs) == length(cpu_names)

            # Test that we have at least basic algorithms
            @test "LUFactorization" in cpu_names
            @test "GenericLUFactorization" in cpu_names

            gpu_algs, gpu_names = LinearSolveAutotune.get_gpu_algorithms()
            @test length(gpu_algs) == length(gpu_names)
            # GPU algorithms might be empty if no GPU packages loaded
        end

        @testset "Element Type Compatibility Testing" begin
            cpu_algs, cpu_names = LinearSolveAutotune.get_available_algorithms()

            # Test Float64 compatibility (should work with all algorithms)
            compatible_algs, compatible_names = LinearSolveAutotune.filter_compatible_algorithms(
                cpu_algs, cpu_names, Float64
            )
            @test !isempty(compatible_algs)
            @test length(compatible_algs) == length(compatible_names)

            # Test Float32 compatibility
            compatible_algs_f32, compatible_names_f32 = LinearSolveAutotune.filter_compatible_algorithms(
                cpu_algs, cpu_names, Float32
            )
            @test !isempty(compatible_algs_f32)

            # Test ComplexF64 compatibility
            compatible_algs_c64, compatible_names_c64 = LinearSolveAutotune.filter_compatible_algorithms(
                cpu_algs, cpu_names, ComplexF64
            )
            @test !isempty(compatible_algs_c64)

            # Test BigFloat compatibility - should exclude BLAS algorithms but include pure Julia ones
            compatible_algs_bf, compatible_names_bf = LinearSolveAutotune.filter_compatible_algorithms(
                cpu_algs, cpu_names, BigFloat
            )
            @test !isempty(compatible_algs_bf)
            # Should include GenericLUFactorization (pure Julia)
            @test "GenericLUFactorization" in compatible_names_bf
            # Should include SimpleLUFactorization (pure Julia)
            @test "SimpleLUFactorization" in compatible_names_bf
            # Should include RFLUFactorization if available (pure Julia)
            if "RFLUFactorization" in cpu_names
                @test "RFLUFactorization" in compatible_names_bf
            end

            # Test individual algorithm compatibility
            for (alg, name) in zip(cpu_algs[1:min(3, end)], cpu_names[1:min(3, end)])
                result = LinearSolveAutotune.test_algorithm_compatibility(alg, Float64)
                @test isa(result, Bool)
            end
        end

        @testset "Benchmark Size Generation" begin
            # Test new size categories
            tiny_sizes = LinearSolveAutotune.get_benchmark_sizes([:tiny])
            @test !isempty(tiny_sizes)
            @test minimum(tiny_sizes) == 5
            @test maximum(tiny_sizes) == 20

            small_sizes = LinearSolveAutotune.get_benchmark_sizes([:small])
            @test !isempty(small_sizes)
            @test minimum(small_sizes) == 20
            @test maximum(small_sizes) == 100

            medium_sizes = LinearSolveAutotune.get_benchmark_sizes([:medium])
            @test !isempty(medium_sizes)
            @test minimum(medium_sizes) == 100
            @test maximum(medium_sizes) == 300

            large_sizes = LinearSolveAutotune.get_benchmark_sizes([:large])
            @test !isempty(large_sizes)
            @test minimum(large_sizes) == 300
            @test maximum(large_sizes) == 1000

            # Test combination
            combined_sizes = LinearSolveAutotune.get_benchmark_sizes([:tiny, :small])
            @test length(combined_sizes) == length(unique(combined_sizes))
            @test minimum(combined_sizes) == 5
            @test maximum(combined_sizes) == 100
        end

        @testset "Small Scale Benchmarking" begin
            # Test with a very small benchmark to ensure functionality
            cpu_algs, cpu_names = LinearSolveAutotune.get_available_algorithms()

            # Use only first 2 algorithms and small sizes for fast testing
            test_algs = cpu_algs[1:min(2, end)]
            test_names = cpu_names[1:min(2, end)]
            test_sizes = [5, 10]  # Very small sizes for fast testing
            test_eltypes = (Float64,)  # Single element type for speed

            results_df = LinearSolveAutotune.benchmark_algorithms(
                test_sizes, test_algs, test_names, test_eltypes;
                samples = 1, seconds = 0.1, sizes = [:tiny]
            )

            @test isa(results_df, DataFrame)
            @test nrow(results_df) > 0
            @test hasproperty(results_df, :size)
            @test hasproperty(results_df, :algorithm)
            @test hasproperty(results_df, :eltype)
            @test hasproperty(results_df, :gflops)
            @test hasproperty(results_df, :success)
            @test hasproperty(results_df, :error)

            # Test that we have results for both sizes and element types
            @test length(unique(results_df.size)) <= length(test_sizes)
            @test all(eltype -> eltype in ["Float64"], unique(results_df.eltype))

            # Check that successful results have positive GFLOPs
            successful_results = filter(row -> row.success, results_df)
            if nrow(successful_results) > 0
                @test all(gflops -> gflops > 0, successful_results.gflops)
            end
        end

        @testset "Result Categorization" begin
            # Create mock results data for testing
            mock_data = [
                (size = 50, algorithm = "TestAlg1", eltype = "Float64", gflops = 10.0, success = true, error = ""),
                (size = 100, algorithm = "TestAlg1", eltype = "Float64", gflops = 12.0, success = true, error = ""),
                (size = 200, algorithm = "TestAlg1", eltype = "Float64", gflops = 8.0, success = true, error = ""),
                (size = 50, algorithm = "TestAlg2", eltype = "Float64", gflops = 8.0, success = true, error = ""),
                (size = 100, algorithm = "TestAlg2", eltype = "Float64", gflops = 15.0, success = true, error = ""),
                (size = 200, algorithm = "TestAlg2", eltype = "Float64", gflops = 14.0, success = true, error = ""),
                (size = 50, algorithm = "TestAlg1", eltype = "Float32", gflops = 9.0, success = true, error = ""),
                (size = 100, algorithm = "TestAlg1", eltype = "Float32", gflops = 11.0, success = true, error = ""),
            ]

            test_df = DataFrame(mock_data)
            categories = LinearSolveAutotune.categorize_results(test_df)

            @test isa(categories, Dict{String, String})
            @test !isempty(categories)

            # Check that categories are properly formatted with element types
            for (key, value) in categories
                @test contains(key, "_")  # Should have element type prefix
                @test !isempty(value)
            end
        end

        @testset "Plotting Functions" begin
            # Create mock results for plotting tests
            mock_data = [
                (size = 50, algorithm = "TestAlg1", eltype = "Float64", gflops = 10.0, success = true, error = ""),
                (size = 100, algorithm = "TestAlg1", eltype = "Float64", gflops = 12.0, success = true, error = ""),
                (size = 50, algorithm = "TestAlg2", eltype = "Float64", gflops = 8.0, success = true, error = ""),
                (size = 100, algorithm = "TestAlg2", eltype = "Float64", gflops = 15.0, success = true, error = ""),
                (size = 50, algorithm = "TestAlg1", eltype = "Float32", gflops = 9.0, success = true, error = ""),
                (size = 100, algorithm = "TestAlg1", eltype = "Float32", gflops = 11.0, success = true, error = ""),
            ]

            test_df = DataFrame(mock_data)

            # Test multi-element type plotting
            plots_dict = LinearSolveAutotune.create_benchmark_plots(test_df)
            @test isa(plots_dict, Dict)
            @test !isempty(plots_dict)
            @test haskey(plots_dict, "Float64")
            @test haskey(plots_dict, "Float32")

            # Test backward compatibility plotting
            single_plot = LinearSolveAutotune.create_benchmark_plot(test_df)
            @test single_plot !== nothing

            # Test with empty data
            empty_df = DataFrame(
                size = Int[], algorithm = String[], eltype = String[],
                gflops = Float64[], success = Bool[], error = String[]
            )
            empty_plots = LinearSolveAutotune.create_benchmark_plots(empty_df)
            @test isa(empty_plots, Dict)
            @test isempty(empty_plots)
        end

        @testset "System Information" begin
            system_info = LinearSolveAutotune.get_system_info()
            @test isa(system_info, Dict)

            # Check required fields
            required_fields = [
                "julia_version", "os", "arch", "cpu_name", "num_cores",
                "num_threads", "blas_vendor", "has_cuda", "has_metal",
                "mkl_available", "apple_accelerate_available",
            ]

            for field in required_fields
                @test haskey(system_info, field)
            end

            # Check types
            @test isa(system_info["julia_version"], String)
            @test isa(system_info["num_cores"], Int)
            @test isa(system_info["num_threads"], Int)
            @test isa(system_info["has_cuda"], Bool)
            @test isa(system_info["has_metal"], Bool)
        end

        @testset "Algorithm Classification" begin
            # Test is_always_loaded_algorithm function
            @test LinearSolveAutotune.is_always_loaded_algorithm("LUFactorization") == true
            @test LinearSolveAutotune.is_always_loaded_algorithm("GenericLUFactorization") == true
            @test LinearSolveAutotune.is_always_loaded_algorithm("MKLLUFactorization") == true
            @test LinearSolveAutotune.is_always_loaded_algorithm("AppleAccelerateLUFactorization") == true
            @test LinearSolveAutotune.is_always_loaded_algorithm("SimpleLUFactorization") == true

            # Test extension-dependent algorithms
            @test LinearSolveAutotune.is_always_loaded_algorithm("RFLUFactorization") == false
            @test LinearSolveAutotune.is_always_loaded_algorithm("FastLUFactorization") == false
            @test LinearSolveAutotune.is_always_loaded_algorithm("BLISLUFactorization") == false
            @test LinearSolveAutotune.is_always_loaded_algorithm("CudaOffloadLUFactorization") == false
            @test LinearSolveAutotune.is_always_loaded_algorithm("MetalLUFactorization") == false

            # Test unknown algorithm
            @test LinearSolveAutotune.is_always_loaded_algorithm("UnknownAlgorithm") == false
        end

        @testset "Best Always-Loaded Algorithm Finding" begin
            # Create mock benchmark data with both always-loaded and extension-dependent algorithms
            mock_data = [
                (size = 150, algorithm = "RFLUFactorization", eltype = "Float64", gflops = 50.0, success = true, error = ""),
                (size = 150, algorithm = "LUFactorization", eltype = "Float64", gflops = 30.0, success = true, error = ""),
                (size = 150, algorithm = "MKLLUFactorization", eltype = "Float64", gflops = 40.0, success = true, error = ""),
                (size = 150, algorithm = "GenericLUFactorization", eltype = "Float64", gflops = 20.0, success = true, error = ""),
                # Add Float32 data
                (size = 150, algorithm = "LUFactorization", eltype = "Float32", gflops = 25.0, success = true, error = ""),
                (size = 150, algorithm = "MKLLUFactorization", eltype = "Float32", gflops = 35.0, success = true, error = ""),
                (size = 150, algorithm = "GenericLUFactorization", eltype = "Float32", gflops = 15.0, success = true, error = ""),
            ]

            test_df = DataFrame(mock_data)

            # Test finding best always-loaded algorithm for Float64 medium size
            best_always_loaded = LinearSolveAutotune.find_best_always_loaded_algorithm(
                test_df, "Float64", "medium (100-300)"
            )
            @test best_always_loaded == "MKLLUFactorization"  # Best among always-loaded (40.0 > 30.0 > 20.0)

            # Test finding best always-loaded algorithm for Float32 medium size
            best_always_loaded_f32 = LinearSolveAutotune.find_best_always_loaded_algorithm(
                test_df, "Float32", "medium (100-300)"
            )
            @test best_always_loaded_f32 == "MKLLUFactorization"  # Best among always-loaded (35.0 > 25.0 > 15.0)

            # Test with no data for a size range
            no_result = LinearSolveAutotune.find_best_always_loaded_algorithm(
                test_df, "Float64", "large (300-1000)"
            )
            @test no_result === nothing

            # Test with unknown element type
            no_result_et = LinearSolveAutotune.find_best_always_loaded_algorithm(
                test_df, "ComplexF64", "medium (100-300)"
            )
            @test no_result_et === nothing
        end

        @testset "Dual Preference System" begin
            # Clear any existing preferences first
            LinearSolveAutotune.clear_algorithm_preferences()

            # Create mock benchmark data
            mock_data = [
                (size = 150, algorithm = "RFLUFactorization", eltype = "Float64", gflops = 50.0, success = true, error = ""),
                (size = 150, algorithm = "LUFactorization", eltype = "Float64", gflops = 30.0, success = true, error = ""),
                (size = 150, algorithm = "MKLLUFactorization", eltype = "Float64", gflops = 40.0, success = true, error = ""),
                (size = 150, algorithm = "GenericLUFactorization", eltype = "Float64", gflops = 20.0, success = true, error = ""),
                # Add Float32 data where MKL is best overall
                (size = 150, algorithm = "LUFactorization", eltype = "Float32", gflops = 25.0, success = true, error = ""),
                (size = 150, algorithm = "MKLLUFactorization", eltype = "Float32", gflops = 45.0, success = true, error = ""),
                (size = 150, algorithm = "GenericLUFactorization", eltype = "Float32", gflops = 15.0, success = true, error = ""),
            ]

            test_df = DataFrame(mock_data)

            # Test categories: RFLU best for Float64, MKL best for Float32
            test_categories = Dict{String, String}(
                "Float64_medium (100-300)" => "RFLUFactorization",
                "Float32_medium (100-300)" => "MKLLUFactorization"
            )

            # Set preferences with benchmark data for intelligent fallback selection
            LinearSolveAutotune.set_algorithm_preferences(test_categories, test_df)

            # Get preferences back
            retrieved_prefs = LinearSolveAutotune.get_algorithm_preferences()
            @test isa(retrieved_prefs, Dict{String, Any})
            @test !isempty(retrieved_prefs)

            # Test Float64 preferences
            @test haskey(retrieved_prefs, "Float64_medium")
            float64_prefs = retrieved_prefs["Float64_medium"]
            @test isa(float64_prefs, Dict)
            @test haskey(float64_prefs, "best")
            @test haskey(float64_prefs, "always_loaded")
            @test float64_prefs["best"] == "RFLUFactorization"  # Best overall
            @test float64_prefs["always_loaded"] == "MKLLUFactorization"  # Best always-loaded

            # Test Float32 preferences
            @test haskey(retrieved_prefs, "Float32_medium")
            float32_prefs = retrieved_prefs["Float32_medium"]
            @test isa(float32_prefs, Dict)
            @test haskey(float32_prefs, "best")
            @test haskey(float32_prefs, "always_loaded")
            @test float32_prefs["best"] == "MKLLUFactorization"  # Best overall
            @test float32_prefs["always_loaded"] == "MKLLUFactorization"  # Same as best (already always-loaded)

            # Test that both preference types are actually set in LinearSolve
            using Preferences
            @test Preferences.has_preference(LinearSolve, "best_algorithm_Float64_medium")
            @test Preferences.has_preference(LinearSolve, "best_always_loaded_Float64_medium")
            @test Preferences.has_preference(LinearSolve, "best_algorithm_Float32_medium")
            @test Preferences.has_preference(LinearSolve, "best_always_loaded_Float32_medium")

            # Verify the actual preference values
            @test Preferences.load_preference(LinearSolve, "best_algorithm_Float64_medium") == "RFLUFactorization"
            @test Preferences.load_preference(LinearSolve, "best_always_loaded_Float64_medium") == "MKLLUFactorization"
            @test Preferences.load_preference(LinearSolve, "best_algorithm_Float32_medium") == "MKLLUFactorization"
            @test Preferences.load_preference(LinearSolve, "best_always_loaded_Float32_medium") == "MKLLUFactorization"

            # Test clearing dual preferences
            LinearSolveAutotune.clear_algorithm_preferences()
            cleared_prefs = LinearSolveAutotune.get_algorithm_preferences()
            @test isempty(cleared_prefs)

            # Verify preferences are actually cleared from LinearSolve
            @test !Preferences.has_preference(LinearSolve, "best_algorithm_Float64_medium")
            @test !Preferences.has_preference(LinearSolve, "best_always_loaded_Float64_medium")
            @test !Preferences.has_preference(LinearSolve, "best_algorithm_Float32_medium")
            @test !Preferences.has_preference(LinearSolve, "best_always_loaded_Float32_medium")
        end

        @testset "Dual Preference Fallback Logic" begin
            # Test fallback logic when no benchmark data is provided
            LinearSolveAutotune.clear_algorithm_preferences()

            # Test categories with extension-dependent algorithms but no benchmark data
            test_categories_no_data = Dict{String, String}(
                "Float64_medium (100-300)" => "RFLUFactorization",
                "ComplexF64_medium (100-300)" => "RFLUFactorization"
            )

            # Set preferences WITHOUT benchmark data (should use fallback logic)
            LinearSolveAutotune.set_algorithm_preferences(test_categories_no_data, nothing)

            # Get preferences back
            retrieved_prefs = LinearSolveAutotune.get_algorithm_preferences()

            # Test Float64 fallback logic
            @test haskey(retrieved_prefs, "Float64_medium")
            float64_prefs = retrieved_prefs["Float64_medium"]
            @test float64_prefs["best"] == "RFLUFactorization"
            # Should fall back to LUFactorization for real types when no MKL detected
            @test float64_prefs["always_loaded"] == "LUFactorization"

            # Test ComplexF64 fallback logic
            @test haskey(retrieved_prefs, "ComplexF64_medium")
            complex_prefs = retrieved_prefs["ComplexF64_medium"]
            @test complex_prefs["best"] == "RFLUFactorization"
            # Should fall back to LUFactorization for complex types (conservative)
            @test complex_prefs["always_loaded"] == "LUFactorization"

            # Clean up
            LinearSolveAutotune.clear_algorithm_preferences()
        end

        @testset "Integration: Dual Preferences Set in autotune_setup" begin
            # Test that autotune_setup actually sets dual preferences
            LinearSolveAutotune.clear_algorithm_preferences()

            # Run a minimal autotune that sets preferences
            result = LinearSolveAutotune.autotune_setup(
                sizes = [:tiny],
                set_preferences = true,  # KEY: Must be true to test preference setting
                samples = 1,
                seconds = 0.1,
                eltypes = (Float64,)
            )

            @test isa(result, AutotuneResults)

            # Check if any preferences were set
            prefs_after_autotune = LinearSolveAutotune.get_algorithm_preferences()

            # If autotune found and categorized results, we should have dual preferences
            if !isempty(prefs_after_autotune)
                # Pick the first preference set to test
                first_key = first(keys(prefs_after_autotune))
                first_prefs = prefs_after_autotune[first_key]

                @test isa(first_prefs, Dict)
                @test haskey(first_prefs, "best")
                @test haskey(first_prefs, "always_loaded")
                @test first_prefs["best"] !== nothing
                @test first_prefs["always_loaded"] !== nothing

                # Both should be valid algorithm names
                @test isa(first_prefs["best"], String)
                @test isa(first_prefs["always_loaded"], String)
                @test !isempty(first_prefs["best"])
                @test !isempty(first_prefs["always_loaded"])

                # The always_loaded algorithm should indeed be always loaded
                @test LinearSolveAutotune.is_always_loaded_algorithm(first_prefs["always_loaded"])
            end

            # Clean up
            LinearSolveAutotune.clear_algorithm_preferences()
        end

        @testset "AutotuneResults Type" begin
            # Create mock data for AutotuneResults
            mock_data = [
                (size = 50, algorithm = "TestAlg1", eltype = "Float64", gflops = 10.0, success = true, error = ""),
                (size = 100, algorithm = "TestAlg2", eltype = "Float64", gflops = 15.0, success = true, error = ""),
            ]

            test_df = DataFrame(mock_data)
            test_sysinfo = Dict(
                "cpu_name" => "Test CPU", "os" => "TestOS",
                "julia_version" => "1.0.0", "num_threads" => 4
            )

            results = AutotuneResults(test_df, test_sysinfo)

            @test isa(results, AutotuneResults)
            @test results.results_df == test_df
            @test results.sysinfo == test_sysinfo

            # Test that display works without error
            io = IOBuffer()
            show(io, results)
            display_output = String(take!(io))
            @test contains(display_output, "LinearSolve.jl Autotune Results")
            @test contains(display_output, "Test CPU")
        end

        @testset "Integration Test - Mini Autotune with New API" begin
            # Test the full autotune_setup function with minimal parameters
            # This is an integration test with very small scale to ensure everything works together

            # Skip telemetry and use minimal settings for testing
            result = LinearSolveAutotune.autotune_setup(
                sizes = [:tiny],
                set_preferences = false,
                samples = 1,
                seconds = 0.1,
                eltypes = (Float64,)  # Single element type for speed
            )

            @test isa(result, AutotuneResults)
            @test isa(result.results_df, DataFrame)
            @test isa(result.sysinfo, Dict)
            @test nrow(result.results_df) > 0
            @test hasproperty(result.results_df, :size)
            @test hasproperty(result.results_df, :algorithm)
            @test hasproperty(result.results_df, :eltype)
            @test hasproperty(result.results_df, :gflops)
            @test hasproperty(result.results_df, :success)

            # Test with multiple element types
            result_multi = LinearSolveAutotune.autotune_setup(
                sizes = [:tiny],
                set_preferences = false,
                samples = 1,
                seconds = 0.1,
                eltypes = (Float64, Float32)
            )

            @test isa(result_multi, AutotuneResults)
            df = result_multi.results_df
            @test nrow(df) > 0

            # Check that we have results for multiple element types
            eltypes_in_results = unique(df.eltype)
            @test length(eltypes_in_results) >= 1  # At least one element type should work
        end
    end
end
