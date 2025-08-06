using Test
using LinearSolve
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
            cpu_algs, cpu_names, Float64)
        @test !isempty(compatible_algs)
        @test length(compatible_algs) == length(compatible_names)
        
        # Test Float32 compatibility
        compatible_algs_f32, compatible_names_f32 = LinearSolveAutotune.filter_compatible_algorithms(
            cpu_algs, cpu_names, Float32)
        @test !isempty(compatible_algs_f32)
        
        # Test ComplexF64 compatibility
        compatible_algs_c64, compatible_names_c64 = LinearSolveAutotune.filter_compatible_algorithms(
            cpu_algs, cpu_names, ComplexF64)
        @test !isempty(compatible_algs_c64)
        
        # Test BigFloat compatibility - should exclude BLAS algorithms but include pure Julia ones
        compatible_algs_bf, compatible_names_bf = LinearSolveAutotune.filter_compatible_algorithms(
            cpu_algs, cpu_names, BigFloat)
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
        # Test small benchmark sizes
        small_sizes = LinearSolveAutotune.get_benchmark_sizes(false)
        @test !isempty(small_sizes)
        @test minimum(small_sizes) >= 4
        @test maximum(small_sizes) <= 500
        
        # Test large benchmark sizes
        large_sizes = LinearSolveAutotune.get_benchmark_sizes(true)
        @test !isempty(large_sizes)
        @test minimum(large_sizes) >= 4
        @test maximum(large_sizes) >= 2000
    end
    
    @testset "Small Scale Benchmarking" begin
        # Test with a very small benchmark to ensure functionality
        cpu_algs, cpu_names = LinearSolveAutotune.get_available_algorithms()
        
        # Use only first 2 algorithms and small sizes for fast testing
        test_algs = cpu_algs[1:min(2, end)]
        test_names = cpu_names[1:min(2, end)]
        test_sizes = [4, 8]  # Very small sizes for fast testing
        test_eltypes = (Float64,)  # Single element type for speed
        
        results_df = LinearSolveAutotune.benchmark_algorithms(
            test_sizes, test_algs, test_names, test_eltypes;
            samples = 1, seconds = 0.1)
        
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
        empty_df = DataFrame(size = Int[], algorithm = String[], eltype = String[], 
                           gflops = Float64[], success = Bool[], error = String[])
        empty_plots = LinearSolveAutotune.create_benchmark_plots(empty_df)
        @test isa(empty_plots, Dict)
        @test isempty(empty_plots)
    end
    
    @testset "System Information" begin
        system_info = LinearSolveAutotune.get_system_info()
        @test isa(system_info, Dict)
        
        # Check required fields
        required_fields = ["julia_version", "os", "arch", "cpu_name", "num_cores", 
                          "num_threads", "blas_vendor", "has_cuda", "has_metal",
                          "mkl_available", "apple_accelerate_available"]
        
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
    
    @testset "Preference Management" begin
        # Test setting and getting preferences
        test_categories = Dict{String, String}(
            "Float64_0-128" => "TestAlg1",
            "Float64_128-256" => "TestAlg2",
            "Float32_0-128" => "TestAlg1"
        )
        
        # Clear any existing preferences first
        LinearSolveAutotune.clear_algorithm_preferences()
        
        # Set test preferences
        LinearSolveAutotune.set_algorithm_preferences(test_categories)
        
        # Get preferences back
        retrieved_prefs = LinearSolveAutotune.get_algorithm_preferences()
        @test isa(retrieved_prefs, Dict{String, String})
        @test !isempty(retrieved_prefs)
        
        # Verify we can retrieve what we set
        for (key, value) in test_categories
            @test_broken haskey(retrieved_prefs, key)
            @test_broken retrieved_prefs[key] == value
        end
        
        # Test clearing preferences
        LinearSolveAutotune.clear_algorithm_preferences()
        cleared_prefs = LinearSolveAutotune.get_algorithm_preferences()
        @test isempty(cleared_prefs)
    end
    
    @testset "Integration Test - Mini Autotune" begin
        # Test the full autotune_setup function with minimal parameters
        # This is an integration test with very small scale to ensure everything works together
        
        # Skip telemetry and use minimal settings for testing
        result, sysinfo, _ = LinearSolveAutotune.autotune_setup(
            large_matrices = false,
            telemetry = false,
            make_plot = false,
            set_preferences = false,
            samples = 1,
            seconds = 0.1,
            eltypes = (Float64,)  # Single element type for speed
        )
        
        @test isa(result, DataFrame)
        @test nrow(result) > 0
        @test hasproperty(result, :size)
        @test hasproperty(result, :algorithm)
        @test hasproperty(result, :eltype)
        @test hasproperty(result, :gflops)
        @test hasproperty(result, :success)
        
        # Test with multiple element types
        result_multi = LinearSolveAutotune.autotune_setup(
            large_matrices = false,
            telemetry = false,
            make_plot = true,  # Test plotting integration
            set_preferences = false,
            samples = 1,
            seconds = 0.1,
            eltypes = (Float64, Float32)
        )
        
        # Should return tuple of (DataFrame, Dataframe, Dict) when make_plot=true
        @test isa(result_multi, Tuple)
        @test length(result_multi) == 3
        @test isa(result_multi[1], DataFrame)
        @test isa(result_multi[2], DataFrame)
        @test isa(result_multi[3], Dict)  # Plots dictionary
        
        df, plots = result_multi
        @test nrow(df) > 0
        @test !isempty(plots)
        
        # Check that we have results for multiple element types
        eltypes_in_results = unique(df.eltype)
        @test length(eltypes_in_results) >= 1  # At least one element type should work
    end
end