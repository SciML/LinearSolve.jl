# Automatic Algorithm Selection with LinearSolveAutotune

LinearSolve.jl includes an automatic tuning system that benchmarks all available linear algebra algorithms on your specific hardware and automatically selects optimal algorithms for different problem sizes and data types. This tutorial will show you how to use the `LinearSolveAutotune` sublibrary to optimize your linear solve performance.

!!! warn
    The autotuning system is under active development. While benchmarking and result sharing are fully functional, automatic preference setting for algorithm selection is still being refined.

## Quick Start

The simplest way to use the autotuner is to run it with default settings:

```julia
using LinearSolve
using LinearSolveAutotune

# Run autotune with default settings
results = autotune_setup()

# View the results
display(results)

# Generate performance plots
plot(results)

# Share results with the community (optional, requires GitHub authentication)
share_results(results)
```

This will:
- Benchmark algorithms for `Float64` matrices by default
- Test matrix sizes from tiny (5×5) through large (1000×1000) 
- Display a summary of algorithm performance
- Return an `AutotuneResults` object containing all benchmark data

## Understanding the Results

The `autotune_setup()` function returns an `AutotuneResults` object containing:
- `results_df`: A DataFrame with detailed benchmark results
- `sysinfo`: System information dictionary

You can explore the results in several ways:

```julia
# Get the results
results = autotune_setup()

# Display a formatted summary
display(results)

# Access the raw benchmark data
df = results.results_df

# View system information
sysinfo = results.sysinfo

# Generate performance plots
plot(results)

# Filter to see successful benchmarks only
using DataFrames
successful = filter(row -> row.success, df)
```

## Customizing the Autotune Process

### Size Categories

Control which matrix size ranges to test:

```julia
# Available size categories:
# :tiny   - 5×5 to 20×20 (very small problems)
# :small  - 20×20 to 100×100 (small problems)  
# :medium - 100×100 to 300×300 (typical problems)
# :large  - 300×300 to 1000×1000 (larger problems)
# :big    - 10000×10000 to 100000×100000 (GPU/HPC scale)

# Default: test tiny through large
results = autotune_setup()  # uses [:tiny, :small, :medium, :large]

# Test only medium and large sizes
results = autotune_setup(sizes = [:medium, :large])

# Include huge matrices (for GPU systems)
results = autotune_setup(sizes = [:large, :big])

# Test all size categories
results = autotune_setup(sizes = [:tiny, :small, :medium, :large, :big])
```

### Element Types

Specify which numeric types to benchmark:

```julia
# Default: Float64 only
results = autotune_setup()  # equivalent to eltypes = (Float64,)

# Test standard floating point types
results = autotune_setup(eltypes = (Float32, Float64))

# Include complex numbers
results = autotune_setup(eltypes = (Float64, ComplexF64))

# Test all standard BLAS types
results = autotune_setup(eltypes = (Float32, Float64, ComplexF32, ComplexF64))

# Test arbitrary precision (excludes some BLAS algorithms)
results = autotune_setup(eltypes = (BigFloat,), skip_missing_algs = true)
```

### Benchmark Quality vs Speed

Adjust the thoroughness of benchmarking:

```julia
# Quick benchmark (fewer samples, less time per test)
results = autotune_setup(samples = 1, seconds = 0.1)

# Default benchmark (balanced)
results = autotune_setup(samples = 5, seconds = 0.5)

# Thorough benchmark (more samples, more time per test)
results = autotune_setup(samples = 10, seconds = 2.0)

# Production-quality benchmark for final tuning
results = autotune_setup(
    samples = 20,
    seconds = 5.0,
    sizes = [:small, :medium, :large],
    eltypes = (Float32, Float64, ComplexF32, ComplexF64)
)
```

### Missing Algorithm Handling

By default, autotune expects all algorithms to be available to ensure complete benchmarking. You can relax this requirement:

```julia
# Default: error if expected algorithms are missing
results = autotune_setup()  # Will error if RFLUFactorization is missing

# Allow missing algorithms (useful for incomplete setups)
results = autotune_setup(skip_missing_algs = true)  # Will warn instead of error
```

### Preferences Setting

Control whether the autotuner updates LinearSolve preferences:

```julia
# Default: set preferences based on benchmark results
results = autotune_setup(set_preferences = true)

# Benchmark only, don't change preferences
results = autotune_setup(set_preferences = false)
```

## GPU Systems

On systems with CUDA or Metal GPU support, the autotuner will automatically detect and benchmark GPU algorithms:

```julia
# Enable large matrix testing for GPUs
results = autotune_setup(
    sizes = [:large, :big],
    samples = 3,
    seconds = 1.0
)
```

GPU algorithms tested (when available):
- **CudaOffloadFactorization**: CUDA GPU acceleration
- **MetalLUFactorization**: Apple Metal GPU acceleration

## Sharing Results with the Community

The autotuner includes a telemetry feature that allows you to share your benchmark results with the LinearSolve.jl community. This helps improve algorithm selection across different hardware configurations.

### Setting Up GitHub Authentication

To share results, you need to authenticate with GitHub. There are two methods:

#### Method 1: GitHub CLI (Recommended)

1. **Install GitHub CLI**
   - macOS: `brew install gh`
   - Windows: `winget install --id GitHub.cli`
   - Linux: See [cli.github.com](https://cli.github.com/manual/installation)

2. **Authenticate**
   ```bash
   gh auth login
   ```
   Follow the prompts to authenticate with your GitHub account.

3. **Verify authentication**
   ```bash
   gh auth status
   ```

#### Method 2: GitHub Personal Access Token

1. Go to [GitHub Settings > Tokens](https://github.com/settings/tokens/new)
2. Add description: "LinearSolve.jl Telemetry"
3. Select scope: `public_repo` (for commenting on issues)
4. Click "Generate token" and copy it
5. In Julia:
   ```julia
   ENV["GITHUB_TOKEN"] = "your_token_here"
   ```

### Sharing Your Results

Once authenticated, sharing is simple:

```julia
# Run benchmarks
results = autotune_setup()

# Share with the community
share_results(results)
```

This will:
1. Format your benchmark results as a markdown report
2. Create performance plots if enabled
3. Post the results as a comment to the [community benchmark collection issue](https://github.com/SciML/LinearSolve.jl/issues/669)
4. Upload plots as GitHub Gists for easy viewing

!!! info "Privacy Note"
    - Sharing is completely optional
    - Only benchmark performance data and system specifications are shared
    - No personal information is collected
    - All shared data is publicly visible on GitHub
    - If authentication fails, results are saved locally for manual sharing

## Working with Results

### Examining Performance Data

```julia
using DataFrames
using Statistics

results = autotune_setup()

# Access the raw DataFrame
df = results.results_df

# Filter successful results
successful = filter(row -> row.success, df)

# Summary by algorithm
summary = combine(groupby(successful, [:algorithm, :eltype]), 
                 :gflops => mean => :avg_gflops,
                 :gflops => maximum => :max_gflops)
sort!(summary, :avg_gflops, rev=true)
println(summary)

# Best algorithm for each size category
by_size = combine(groupby(successful, [:size_category, :eltype])) do group
    best_row = argmax(group.gflops)
    return (algorithm = group.algorithm[best_row],
            gflops = group.gflops[best_row])
end
println(by_size)
```

### Performance Visualization

Generate and save performance plots:

```julia
results = autotune_setup()

# Generate plots (returns a combined plot)
p = plot(results)
display(p)

# Save the plot
using Plots
savefig(p, "benchmark_results.png")
```

### Accessing System Information

```julia
results = autotune_setup()

# System information is stored in the results
sysinfo = results.sysinfo
println("CPU: ", sysinfo["cpu_name"])
println("Cores: ", sysinfo["num_cores"])
println("Julia: ", sysinfo["julia_version"])
println("OS: ", sysinfo["os"])
```

## Advanced Usage

### Custom Benchmark Pipeline

For complete control over the benchmarking process:

```julia
# Step 1: Run benchmarks without plotting or sharing
results = autotune_setup(
    sizes = [:medium, :large],
    eltypes = (Float64, ComplexF64),
    set_preferences = false,  # Don't change preferences yet
    samples = 10,
    seconds = 1.0
)

# Step 2: Analyze results
df = results.results_df
# ... perform custom analysis ...

# Step 3: Generate plots
p = plot(results)
savefig(p, "my_benchmarks.png")

# Step 4: Optionally share results
share_results(results)
```

### Batch Testing Multiple Configurations

```julia
# Test different element types separately
configs = [
    (eltypes = (Float32,), name = "float32"),
    (eltypes = (Float64,), name = "float64"),
    (eltypes = (ComplexF64,), name = "complex64")
]

all_results = Dict()
for config in configs
    println("Testing $(config.name)...")
    results = autotune_setup(
        eltypes = config.eltypes,
        sizes = [:small, :medium],
        samples = 3
    )
    all_results[config.name] = results
end
```

## Preferences Integration

!!! warn
    Automatic preference setting is still under development and may not affect algorithm selection in the current version.

The autotuner can set preferences that LinearSolve.jl will use for automatic algorithm selection:

```julia
using LinearSolveAutotune

# View current preferences (if any)
LinearSolveAutotune.show_current_preferences()

# Run autotune and set preferences
results = autotune_setup(set_preferences = true)

# Clear all autotune preferences
LinearSolveAutotune.clear_algorithm_preferences()

# Manually set custom preferences
custom_categories = Dict(
    "Float64_0-128" => "RFLUFactorization",
    "Float64_128-256" => "LUFactorization"
)
LinearSolveAutotune.set_algorithm_preferences(custom_categories)
```

## Troubleshooting

### Common Issues

1. **Missing algorithms error**
   ```julia
   # If you get errors about missing algorithms:
   results = autotune_setup(skip_missing_algs = true)
   ```

2. **GitHub authentication fails**
   - Ensure gh CLI is installed and authenticated: `gh auth status`
   - Or set a valid GitHub token: `ENV["GITHUB_TOKEN"] = "your_token"`
   - Results will be saved locally if authentication fails

3. **Out of memory on large matrices**
   ```julia
   # Use smaller size categories
   results = autotune_setup(sizes = [:tiny, :small, :medium])
   ```

4. **Benchmarks taking too long**
   ```julia
   # Reduce samples and time per benchmark
   results = autotune_setup(samples = 1, seconds = 0.1)
   ```

## Summary

LinearSolveAutotune provides a comprehensive system for benchmarking and optimizing LinearSolve.jl performance on your specific hardware. Key features include:

- Flexible size categories from tiny to GPU-scale matrices
- Support for all standard numeric types
- Automatic GPU algorithm detection
- Community result sharing via GitHub
- Performance visualization
- Preference setting for automatic algorithm selection (in development)

By running autotune and optionally sharing your results, you help improve LinearSolve.jl's performance for everyone in the Julia community.