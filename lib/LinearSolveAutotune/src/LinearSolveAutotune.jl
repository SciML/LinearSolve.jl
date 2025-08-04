module LinearSolveAutotune

using LinearSolve
using BenchmarkTools
using CSV
using DataFrames
using PrettyTables
using Preferences
using Statistics
using Random
using LinearAlgebra
using Printf
using Dates
using RecursiveFactorization  # Hard dependency to ensure RFLUFactorization is available

# Optional dependencies for telemetry and plotting
using GitHub
using Plots

export autotune_setup

include("algorithms.jl")
include("gpu_detection.jl")
include("benchmarking.jl")
include("plotting.jl")
include("telemetry.jl")
include("preferences.jl")

"""
    autotune_setup(; 
        large_matrices::Bool = false,
        telemetry::Bool = true,
        make_plot::Bool = true,
        set_preferences::Bool = true,
        samples::Int = 5,
        seconds::Float64 = 0.5,
        eltypes = (Float32, Float64, ComplexF32, ComplexF64),
        skip_missing_algs::Bool = false)

Run a comprehensive benchmark of all available LU factorization methods and optionally:

  - Create performance plots for each element type
  - Upload results to GitHub telemetry  
  - Set Preferences for optimal algorithm selection
  - Support both CPU and GPU algorithms based on hardware detection
  - Test algorithm compatibility with different element types

# Arguments

  - `large_matrices::Bool = false`: Include larger matrix sizes for GPU benchmarking
  - `telemetry::Bool = true`: Share results to GitHub issue for community data
  - `make_plot::Bool = true`: Generate performance plots for each element type
  - `set_preferences::Bool = true`: Update LinearSolve preferences with optimal algorithms
  - `samples::Int = 5`: Number of benchmark samples per algorithm/size
  - `seconds::Float64 = 0.5`: Maximum time per benchmark
  - `eltypes = (Float32, Float64, ComplexF32, ComplexF64)`: Element types to benchmark
  - `skip_missing_algs::Bool = false`: If false, error when expected algorithms are missing; if true, warn instead

# Returns

  - `DataFrame`: Detailed benchmark results with performance data for all element types
  - `Dict` or `Plot`: Performance visualizations by element type (if `make_plot=true`)

# Examples

```julia
using LinearSolve
using LinearSolveAutotune

# Basic autotune with default settings (4 element types)
results = autotune_setup()

# Custom autotune for GPU systems with larger matrices
results = autotune_setup(large_matrices = true, samples = 10, seconds = 1.0)

# Autotune with only Float64 and ComplexF64
results = autotune_setup(eltypes = (Float64, ComplexF64))

# Test with BigFloat (note: most BLAS algorithms will be excluded)
results = autotune_setup(eltypes = (BigFloat,), telemetry = false)

# Allow missing algorithms (useful for incomplete setups)
results = autotune_setup(skip_missing_algs = true)
```
"""
function autotune_setup(;
        large_matrices::Bool = true,
        telemetry::Bool = true,
        make_plot::Bool = true,
        set_preferences::Bool = true,
        samples::Int = 5,
        seconds::Float64 = 0.5,
        eltypes = (Float32, Float64, ComplexF32, ComplexF64),
        skip_missing_algs::Bool = false)
    @info "Starting LinearSolve.jl autotune setup..."
    @info "Configuration: large_matrices=$large_matrices, telemetry=$telemetry, make_plot=$make_plot, set_preferences=$set_preferences"
    @info "Element types to benchmark: $(join(eltypes, ", "))"

    # Set up GitHub authentication early if telemetry is enabled
    github_auth = nothing
    if telemetry
        @info "🔗 Checking GitHub authentication for telemetry..."
        github_auth = setup_github_authentication()
        if github_auth === nothing
            @info "📊 Continuing with benchmarking (results will be saved locally)"
        end
    end

    # Get system information
    system_info = get_system_info()
    @info "System detected: $(system_info["os"]) $(system_info["arch"]) with $(system_info["num_cores"]) cores"

    # Get available algorithms
    cpu_algs, cpu_names = get_available_algorithms(; skip_missing_algs = skip_missing_algs)
    @info "Found $(length(cpu_algs)) CPU algorithms: $(join(cpu_names, ", "))"

    # Add GPU algorithms if available
    gpu_algs, gpu_names = get_gpu_algorithms(; skip_missing_algs = skip_missing_algs)
    if !isempty(gpu_algs)
        @info "Found $(length(gpu_algs)) GPU algorithms: $(join(gpu_names, ", "))"
    end

    # Combine all algorithms
    all_algs = vcat(cpu_algs, gpu_algs)
    all_names = vcat(cpu_names, gpu_names)

    if isempty(all_algs)
        error("No algorithms found! This shouldn't happen.")
    end

    # Get benchmark sizes
    sizes = collect(get_benchmark_sizes(large_matrices))
    @info "Benchmarking $(length(sizes)) matrix sizes from $(minimum(sizes)) to $(maximum(sizes))"

    # Run benchmarks
    @info "Running benchmarks (this may take several minutes)..."
    results_df = benchmark_algorithms(sizes, all_algs, all_names, eltypes;
        samples = samples, seconds = seconds, large_matrices = large_matrices)

    # Display results table
    successful_results = filter(row -> row.success, results_df)
    if nrow(successful_results) > 0
        @info "Benchmark completed successfully!"

        # Create summary table for display
        summary = combine(groupby(successful_results, :algorithm),
            :gflops => mean => :avg_gflops,
            :gflops => maximum => :max_gflops,
            nrow => :num_tests)
        sort!(summary, :avg_gflops, rev = true)

        println("\n" * "="^60)
        println("BENCHMARK RESULTS SUMMARY")
        println("="^60)
        pretty_table(summary,
            header = ["Algorithm", "Avg GFLOPs", "Max GFLOPs", "Tests"],
            formatters = ft_printf("%.2f", [2, 3]),
            crop = :none)
    else
        @warn "No successful benchmark results!"
        return results_df, nothing
    end

    # Categorize results and find best algorithms per size range
    categories = categorize_results(results_df)

    # Set preferences if requested
    if set_preferences && !isempty(categories)
        set_algorithm_preferences(categories)
    end

    # Create plots if requested
    plots_dict = nothing
    plot_files = nothing
    if make_plot
        @info "Creating performance plots..."
        plots_dict = create_benchmark_plots(results_df)
        if !isempty(plots_dict)
            plot_files = save_benchmark_plots(plots_dict)
        end
    end

    # Upload telemetry if requested
    if telemetry && nrow(successful_results) > 0
        @info "📤 Preparing telemetry data for community sharing..."
        markdown_content = format_results_for_github(results_df, system_info, categories)
        upload_to_github(markdown_content, plot_files, github_auth, results_df, system_info, categories)
    end

    @info "Autotune setup completed!"

    # Return results and plots
    if make_plot && plots_dict !== nothing && !isempty(plots_dict)
        return results_df, plots_dict
    else
        return results_df
    end
end

end
