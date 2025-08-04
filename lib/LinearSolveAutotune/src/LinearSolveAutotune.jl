module LinearSolveAutotune

using LinearSolve
using BenchmarkTools
using DataFrames
using PrettyTables
using Preferences
using Statistics
using Random
using LinearAlgebra
using Printf
using Dates

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
        seconds::Float64 = 0.5)

Run a comprehensive benchmark of all available LU factorization methods and optionally:

  - Create performance plots
  - Upload results to GitHub telemetry
  - Set Preferences for optimal algorithm selection
  - Support both CPU and GPU algorithms based on hardware detection

# Arguments

  - `large_matrices::Bool = false`: Include larger matrix sizes for GPU benchmarking
  - `telemetry::Bool = true`: Share results to GitHub issue for community data
  - `make_plot::Bool = true`: Generate performance plots
  - `set_preferences::Bool = true`: Update LinearSolve preferences with optimal algorithms
  - `samples::Int = 5`: Number of benchmark samples per algorithm/size
  - `seconds::Float64 = 0.5`: Maximum time per benchmark

# Returns

  - `DataFrame`: Detailed benchmark results with performance data
  - `Plot`: Performance visualization (if `make_plot=true`)

# Examples

```julia
using LinearSolve
using LinearSolveAutotune

# Basic autotune with default settings
results = autotune_setup()

# Custom autotune for GPU systems with larger matrices
results = autotune_setup(large_matrices = true, samples = 10, seconds = 1.0)

# Autotune without telemetry sharing
results = autotune_setup(telemetry = false)
```
"""
function autotune_setup(;
        large_matrices::Bool = false,
        telemetry::Bool = true,
        make_plot::Bool = true,
        set_preferences::Bool = true,
        samples::Int = 5,
        seconds::Float64 = 0.5)
    @info "Starting LinearSolve.jl autotune setup..."
    @info "Configuration: large_matrices=$large_matrices, telemetry=$telemetry, make_plot=$make_plot, set_preferences=$set_preferences"

    # Get system information
    system_info = get_system_info()
    @info "System detected: $(system_info["os"]) $(system_info["arch"]) with $(system_info["num_cores"]) cores"

    # Get available algorithms
    cpu_algs, cpu_names = get_available_algorithms()
    @info "Found $(length(cpu_algs)) CPU algorithms: $(join(cpu_names, ", "))"

    # Add GPU algorithms if available
    gpu_algs, gpu_names = get_gpu_algorithms()
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
    results_df = benchmark_algorithms(sizes, all_algs, all_names;
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

    # Create plot if requested
    plot_obj = nothing
    plot_files = nothing
    if make_plot
        @info "Creating performance plots..."
        plot_obj = create_benchmark_plot(results_df)
        if plot_obj !== nothing
            plot_files = save_benchmark_plot(plot_obj)
        end
    end

    # Upload telemetry if requested
    if telemetry && nrow(successful_results) > 0
        @info "Preparing telemetry data for GitHub..."
        markdown_content = format_results_for_github(results_df, system_info, categories)
        upload_to_github(markdown_content, plot_files)
    end

    @info "Autotune setup completed!"

    # Return results and plot
    if make_plot && plot_obj !== nothing
        return results_df, plot_obj
    else
        return results_df
    end
end

end
