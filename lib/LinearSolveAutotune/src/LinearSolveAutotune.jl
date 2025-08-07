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
using Base64

# Hard dependency to ensure RFLUFactorization others solvers are available
using RecursiveFactorization  
using blis_jll
using LAPACK_jll
using CUDA
using Metal


# Optional dependencies for telemetry and plotting
using GitHub
using Plots

export autotune_setup, share_results

include("algorithms.jl")
include("gpu_detection.jl")
include("benchmarking.jl")
include("plotting.jl")
include("telemetry.jl")
include("preferences.jl")

"""
    autotune_setup(; 
        sizes = [:small, :medium],
        make_plot::Bool = true,
        set_preferences::Bool = true,
        samples::Int = 5,
        seconds::Float64 = 0.5,
        eltypes = (Float32, Float64, ComplexF32, ComplexF64),
        skip_missing_algs::Bool = false)

Run a comprehensive benchmark of all available LU factorization methods and optionally:

  - Create performance plots for each element type
  - Set Preferences for optimal algorithm selection
  - Support both CPU and GPU algorithms based on hardware detection
  - Test algorithm compatibility with different element types

# Arguments

  - `sizes = [:small, :medium]`: Size categories to test. Options: :small (5-20), :medium (20-100), :large (100-1000), :big (10000-100000)
  - `make_plot::Bool = true`: Generate performance plots for each element type
  - `set_preferences::Bool = true`: Update LinearSolve preferences with optimal algorithms
  - `samples::Int = 5`: Number of benchmark samples per algorithm/size
  - `seconds::Float64 = 0.5`: Maximum time per benchmark
  - `eltypes = (Float32, Float64, ComplexF32, ComplexF64)`: Element types to benchmark
  - `skip_missing_algs::Bool = false`: If false, error when expected algorithms are missing; if true, warn instead

# Returns

  - `DataFrame`: Detailed benchmark results with performance data for all element types
  - `Dict`: System information about the benchmark environment
  - `Dict` or `Plot`: Performance visualizations by element type (if `make_plot=true`)

# Examples

```julia
using LinearSolve
using LinearSolveAutotune

# Basic autotune with small and medium sizes
results, sysinfo, plots = autotune_setup()

# Test all size ranges
results, sysinfo, plots = autotune_setup(sizes = [:small, :medium, :large, :big])

# Large matrices only
results, sysinfo, plots = autotune_setup(sizes = [:large, :big], samples = 10, seconds = 1.0)

# After running autotune, share results (requires gh CLI or GitHub token)
share_results(results, sysinfo, plots)
```
"""
function autotune_setup(;
        sizes = [:small, :medium],
        make_plot::Bool = true,
        set_preferences::Bool = true,
        samples::Int = 5,
        seconds::Float64 = 0.5,
        eltypes = (Float32, Float64, ComplexF32, ComplexF64),
        skip_missing_algs::Bool = false)
    @info "Starting LinearSolve.jl autotune setup..."
    @info "Configuration: sizes=$sizes, make_plot=$make_plot, set_preferences=$set_preferences"
    @info "Element types to benchmark: $(join(eltypes, ", "))"

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

    # Get benchmark sizes based on size categories
    matrix_sizes = collect(get_benchmark_sizes(sizes))
    @info "Benchmarking $(length(matrix_sizes)) matrix sizes from $(minimum(matrix_sizes)) to $(maximum(matrix_sizes))"

    # Run benchmarks
    @info "Running benchmarks (this may take several minutes)..."
    results_df = benchmark_algorithms(matrix_sizes, all_algs, all_names, eltypes;
        samples = samples, seconds = seconds, sizes = sizes)

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

    @info "Autotune setup completed!"

    sysinfo = get_detailed_system_info()

    @info "To share your results with the community, run: share_results(results_df, sysinfo, plots_dict)"
    
    # Return results and plots
    if make_plot && plots_dict !== nothing && !isempty(plots_dict)
        return results_df, sysinfo, plots_dict
    else
        return results_df, sysinfo, nothing
    end
end

"""
    share_results(results_df::DataFrame, sysinfo::Dict, plots_dict=nothing)

Share your benchmark results with the LinearSolve.jl community to help improve 
automatic algorithm selection across different hardware configurations.

This function will authenticate with GitHub (using gh CLI or token) and post
your results as a comment to the community benchmark collection issue.

# Setup Instructions

## Method 1: GitHub CLI (Recommended)
1. Install GitHub CLI: https://cli.github.com/
2. Run: `gh auth login` 
3. Follow the prompts to authenticate
4. Run this function - it will automatically use your gh session

## Method 2: GitHub Token
1. Go to: https://github.com/settings/tokens/new
2. Add description: "LinearSolve.jl Telemetry"
3. Select scope: "public_repo" (for commenting on issues)
4. Click "Generate token" and copy it
5. Set environment variable: `ENV["GITHUB_TOKEN"] = "your_token_here"`
6. Run this function

# Arguments
- `results_df`: Benchmark results DataFrame from autotune_setup
- `sysinfo`: System information Dict from autotune_setup
- `plots_dict`: Optional plots dictionary from autotune_setup

# Examples
```julia
# Run benchmarks
results, sysinfo, plots = autotune_setup()

# Share results with the community
share_results(results, sysinfo, plots)
```
"""
function share_results(results_df::DataFrame, sysinfo::Dict, plots_dict=nothing)
    @info "📤 Preparing to share benchmark results with the community..."
    
    # Get system info if not provided
    system_info = if haskey(sysinfo, "os")
        sysinfo
    else
        get_system_info()
    end
    
    # Categorize results
    categories = categorize_results(results_df)
    
    # Set up authentication
    @info "🔗 Setting up GitHub authentication..."
    @info "ℹ️  For setup instructions, see the documentation or visit:"
    @info "    https://cli.github.com/ (for gh CLI)"
    @info "    https://github.com/settings/tokens/new (for token)"
    
    github_auth = setup_github_authentication()
    
    if github_auth === nothing || github_auth[1] === nothing
        @warn "❌ GitHub authentication not available."
        @info "📝 To share results, please set up authentication:"
        @info "    Option 1: Install gh CLI and run: gh auth login"
        @info "    Option 2: Create a GitHub token and set: ENV[\"GITHUB_TOKEN\"] = \"your_token\""
        
        # Save results locally as fallback
        timestamp = replace(string(Dates.now()), ":" => "-")
        fallback_file = "autotune_results_$(timestamp).md"
        markdown_content = format_results_for_github(results_df, system_info, categories)
        open(fallback_file, "w") do f
            write(f, markdown_content)
        end
        @info "📁 Results saved locally to $fallback_file"
        @info "    You can manually share this file on the issue tracker."
        return
    end
    
    # Format results
    markdown_content = format_results_for_github(results_df, system_info, categories)
    
    # Process plots if available
    plot_files = nothing
    if plots_dict !== nothing && !isempty(plots_dict)
        plot_files = save_benchmark_plots(plots_dict)
    end
    
    # Upload to GitHub
    upload_to_github(markdown_content, plot_files, github_auth, results_df, system_info, categories)
    
    @info "✅ Thank you for contributing to the LinearSolve.jl community!"
end

end
