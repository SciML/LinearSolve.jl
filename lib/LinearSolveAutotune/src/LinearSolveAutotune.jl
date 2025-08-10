module LinearSolveAutotune

# Ensure MKL is available for benchmarking by setting the preference before loading LinearSolve
using Preferences
using MKL_jll

# Set MKL preference to true for benchmarking if MKL is available
# We need to use UUID instead of the module since LinearSolve isn't loaded yet
const LINEARSOLVE_UUID = Base.UUID("7ed4a6bd-45f5-4d41-b270-4a48e9bafcae")
if MKL_jll.is_available()
    # Force load MKL for benchmarking to ensure we can test MKL algorithms
    # The autotune results will determine the final preference setting
    current_pref = Preferences.load_preference(LINEARSOLVE_UUID, "LoadMKL_JLL", nothing)
    if current_pref !== true
        Preferences.set_preferences!(LINEARSOLVE_UUID, "LoadMKL_JLL" => true; force = true)
        @info "Temporarily setting LoadMKL_JLL=true for benchmarking (was $(current_pref))"
    end
end

using LinearSolve
using BenchmarkTools
using DataFrames
using PrettyTables
using Statistics
using Random
using LinearAlgebra
using Printf
using Dates
using Base64
using ProgressMeter
using CPUSummary

# Hard dependency to ensure RFLUFactorization others solvers are available
using RecursiveFactorization  
using blis_jll
using LAPACK_jll
using CUDA
using Metal


# Optional dependencies for telemetry and plotting
using GitHub
using gh_cli_jll
using Plots

export autotune_setup, share_results, AutotuneResults, plot

include("algorithms.jl")
include("gpu_detection.jl")
include("benchmarking.jl")
include("plotting.jl")
include("telemetry.jl")
include("preferences.jl")

# Define the AutotuneResults struct
struct AutotuneResults
    results_df::DataFrame
    sysinfo::Dict
end

# Display method for AutotuneResults
function Base.show(io::IO, results::AutotuneResults)
    println(io, "="^60)
    println(io, "LinearSolve.jl Autotune Results")
    println(io, "="^60)
    
    # System info summary
    println(io, "\n📊 System Information:")
    println(io, "  • CPU: ", get(results.sysinfo, "cpu_name", "Unknown"))
    println(io, "  • OS: ", get(results.sysinfo, "os_name", "Unknown"), " (", get(results.sysinfo, "os", "Unknown"), ")")
    println(io, "  • Julia: ", get(results.sysinfo, "julia_version", "Unknown"))
    println(io, "  • Threads: ", get(results.sysinfo, "num_threads", "Unknown"), " (BLAS: ", get(results.sysinfo, "blas_num_threads", "Unknown"), ")")
    
    # Results summary
    successful_results = filter(row -> row.success, results.results_df)
    if nrow(successful_results) > 0
        println(io, "\n🏆 Top Performing Algorithms:")
        summary = combine(groupby(successful_results, :algorithm),
            :gflops => mean => :avg_gflops,
            :gflops => maximum => :max_gflops,
            nrow => :num_tests)
        sort!(summary, :avg_gflops, rev = true)
        
        # Show top 5
        for (i, row) in enumerate(eachrow(first(summary, 5)))
            println(io, "  ", i, ". ", row.algorithm, ": ",
                    @sprintf("%.2f GFLOPs avg", row.avg_gflops))
        end
    end
    
    # Element types tested
    eltypes = unique(results.results_df.eltype)
    println(io, "\n🔬 Element Types Tested: ", join(eltypes, ", "))
    
    # Matrix sizes tested
    sizes = unique(results.results_df.size)
    println(io, "📏 Matrix Sizes: ", minimum(sizes), "×", minimum(sizes), 
            " to ", maximum(sizes), "×", maximum(sizes))
    
    # Call to action - reordered
    println(io, "\n" * "="^60)
    println(io, "🚀 For comprehensive results, consider running:")
    println(io, "   results_full = autotune_setup(")
    println(io, "       sizes = [:tiny, :small, :medium, :large, :big],")
    println(io, "       eltypes = (Float32, Float64, ComplexF32, ComplexF64)")
    println(io, "   )")
    println(io, "\n📈 See community results at:")
    println(io, "   https://github.com/SciML/LinearSolve.jl/issues/669")
    println(io, "\n💡 To share your results with the community, run:")
    println(io, "   share_results(results)")
    println(io, "="^60)
end

# Plot method for AutotuneResults
function Plots.plot(results::AutotuneResults; kwargs...)
    # Generate plots from the results data
    plots_dict = create_benchmark_plots(results.results_df)
    
    if plots_dict === nothing || isempty(plots_dict)
        @warn "No data available for plotting"
        return nothing
    end
    
    # Create a composite plot from all element type plots
    plot_list = []
    for (eltype_name, p) in plots_dict
        push!(plot_list, p)
    end
    
    # Create composite plot
    n_plots = length(plot_list)
    if n_plots == 1
        return plot_list[1]
    elseif n_plots == 2
        return plot(plot_list..., layout=(1, 2), size=(1200, 500); kwargs...)
    elseif n_plots <= 4
        return plot(plot_list..., layout=(2, 2), size=(1200, 900); kwargs...)
    else
        ncols = ceil(Int, sqrt(n_plots))
        nrows = ceil(Int, n_plots / ncols)
        return plot(plot_list..., layout=(nrows, ncols), 
                   size=(400*ncols, 400*nrows); kwargs...)
    end
end

"""
    autotune_setup(; 
        sizes = [:small, :medium, :large],
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
  - Automatically manage MKL loading preference based on performance results

!!! note "MKL Preference Management"
    During benchmarking, MKL is temporarily enabled (if available) to test MKL algorithms.
    After benchmarking, the LoadMKL_JLL preference is set based on whether MKL algorithms
    performed best in any category. This optimizes startup time and memory usage.

# Arguments

  - `sizes = [:small, :medium, :large]`: Size categories to test. Options: :small (5-20), :medium (20-300), :large (300-1000), :big (10000-100000)
  - `set_preferences::Bool = true`: Update LinearSolve preferences with optimal algorithms
  - `samples::Int = 5`: Number of benchmark samples per algorithm/size
  - `seconds::Float64 = 0.5`: Maximum time per benchmark
  - `eltypes = (Float32, Float64, ComplexF32, ComplexF64)`: Element types to benchmark
  - `skip_missing_algs::Bool = false`: If false, error when expected algorithms are missing; if true, warn instead

# Returns

  - `AutotuneResults`: Object containing benchmark results, system info, and plots

# Examples

```julia
using LinearSolve
using LinearSolveAutotune

# Basic autotune with default sizes
results = autotune_setup()

# Test all size ranges
results = autotune_setup(sizes = [:small, :medium, :large, :big])

# Large matrices only
results = autotune_setup(sizes = [:large, :big], samples = 10, seconds = 1.0)

# After running autotune, share results (requires gh CLI or GitHub token)
share_results(results)
```
"""
function autotune_setup(;
        sizes = [:tiny, :small, :medium, :large],
        set_preferences::Bool = true,
        samples::Int = 5,
        seconds::Float64 = 0.5,
        eltypes = (Float64,),
        skip_missing_algs::Bool = false)
    @info "Starting LinearSolve.jl autotune setup..."
    @info "Configuration: sizes=$sizes, set_preferences=$set_preferences"
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

    @info "Autotune setup completed!"

    sysinfo_df = get_detailed_system_info()
    # Convert DataFrame to Dict for AutotuneResults
    sysinfo = Dict{String, Any}()
    if nrow(sysinfo_df) > 0
        for col in names(sysinfo_df)
            sysinfo[col] = sysinfo_df[1, col]
        end
    end

    # Return AutotuneResults object
    return AutotuneResults(results_df, sysinfo)
end

"""
    share_results(results::AutotuneResults; auto_login::Bool = true)

Share your benchmark results with the LinearSolve.jl community to help improve 
automatic algorithm selection across different hardware configurations.

This function will authenticate with GitHub (using gh CLI or token) and post
your results as a comment to the community benchmark collection issue.

If authentication is not found and `auto_login` is true, the function will
offer to run `gh auth login` interactively to set up authentication.

# Arguments
- `results`: AutotuneResults object from autotune_setup
- `auto_login`: If true, prompts to authenticate if not already authenticated (default: true)

# Authentication Methods

## Automatic (New!)
If gh is not authenticated, the function will offer to run authentication for you.

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

# Examples
```julia
# Run benchmarks
results = autotune_setup()

# Share results with automatic authentication prompt
share_results(results)

# Share results without authentication prompt
share_results(results; auto_login = false)
```
"""
function share_results(results::AutotuneResults; auto_login::Bool = true)
    @info "📤 Preparing to share benchmark results with the community..."
    
    # Extract from AutotuneResults
    results_df = results.results_df
    sysinfo = results.sysinfo
    
    # Get system info
    system_info = sysinfo
    
    # Categorize results
    categories = categorize_results(results_df)
    
    # Set up authentication (with auto-login prompt if enabled)
    @info "🔗 Checking GitHub authentication..."
    
    github_auth = setup_github_authentication(; auto_login = auto_login)
    
    if github_auth === nothing || github_auth[1] === nothing
        # Save results locally as fallback
        timestamp = replace(string(Dates.now()), ":" => "-")
        fallback_file = "autotune_results_$(timestamp).md"
        markdown_content = format_results_for_github(results_df, system_info, categories)
        open(fallback_file, "w") do f
            write(f, markdown_content)
        end
        @info "📁 Results saved locally to $fallback_file"
        @info "    You can manually share this file on the issue tracker:"
        @info "    https://github.com/SciML/LinearSolve.jl/issues/669"
        return
    end
    
    # Format results
    markdown_content = format_results_for_github(results_df, system_info, categories)
    
    # Upload to GitHub (without plots)
    upload_to_github(markdown_content, nothing, github_auth, results_df, system_info, categories)
    
    @info "✅ Thank you for contributing to the LinearSolve.jl community!"
end

end
