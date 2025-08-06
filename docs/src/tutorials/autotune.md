# Automatic Algorithm Selection with LinearSolveAutotune

LinearSolve.jl includes an automatic tuning system that benchmarks all available linear algebra algorithms on your specific hardware and automatically selects optimal algorithms for different problem sizes and data types. This tutorial will show you how to use the `LinearSolveAutotune` sublibrary to optimize your linear solve performance.

!!! warn

    This is still in development. At this point the tuning will not result in different settings
    but it will run the benchmarking and create plots of the performance of the algorithms. A
    future version will use the results to set preferences for the algorithms.

## Quick Start

The simplest way to use the autotuner is to run it with default settings:

```julia
using LinearSolve
using LinearSolveAutotune

# Run autotune with default settings
results, sysinfo, plots = autotune_setup()
```

This will:
- Benchmark 4 element types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`
- Test matrix sizes from small (4×4), medium (500×500), to large (10,000×10,000)
- Create performance plots for each element type
- Set preferences for optimal algorithm selection
- Share results with the community (if desired)

## Understanding the Results

The autotune process returns benchmark results and creates several outputs:

```julia
# Basic usage returns just the DataFrame of results and system information
results, sysinfo, _ = autotune_setup(make_plot=false)

# With plotting enabled, returns (DataFrame, System Info, Plots)
results, sysinfo, plots = autotune_setup(make_plot=true)

# Examine the results
println("Algorithms tested: ", unique(results.algorithm))
println("Element types: ", unique(results.eltype))
println("Size range: ", minimum(results.size), " to ", maximum(results.size))
```

## Customizing the Autotune Process

### Element Types

You can specify which element types to benchmark:

```julia
# Test only Float64 and ComplexF64
results, sysinfo, _ = autotune_setup(eltypes = (Float64, ComplexF64))

# Test arbitrary precision types (excludes BLAS algorithms)
results, sysinfo, _ = autotune_setup(eltypes = (BigFloat,), telemetry = false)

# Test high precision float
results, sysinfo, _ = autotune_setup(eltypes = (Float64, BigFloat))
```

### Matrix Sizes

Control the range of matrix sizes tested:

```julia
# Default: small to medium matrices (4×4 to 500×500)
results, sysinfo, _ = autotune_setup(large_matrices = false)

# Large matrices: includes sizes up to 10,000×10,000 (good for GPU systems)
results, sysinfo, _ = autotune_setup(large_matrices = true)
```

### Benchmark Quality vs Speed

Adjust the thoroughness of benchmarking:

```julia
# Quick benchmark (fewer samples, less time per test)
results, sysinfo, _ = autotune_setup(samples = 1, seconds = 0.1)

# Thorough benchmark (more samples, more time per test)  
results, sysinfo, _ = autotune_setup(samples = 10, seconds = 2.0)
```

### Privacy and Telemetry

!!! warn

    Telemetry implementation is still in development.

The telemetry featrure of LinearSolveAutotune allows sharing performance results 
with the community to improve algorithm selection. Minimal data is collected, including:

- System information (OS, CPU, core count)
- Algorithm performance results 

and shared via public GitHub. This helps the community understand performance across 
different hardware configurations and further improve the default algorithm selection
and research in improved algorithms.

However, if your system has privacy concerns or you prefer not to share data, you can disable telemetry:

```julia
# Disable telemetry (no data shared)
results, sysinfo, _ = autotune_setup(telemetry = false)

# Disable preference setting (just benchmark, don't change defaults)
results, sysinfo, _ = autotune_setup(set_preferences = false)

# Disable plotting (faster, less output)
results, sysinfo, _ = autotune_setup(make_plot = false)
```

### Missing Algorithm Handling

By default, autotune is assertive about finding all expected algorithms. This is because
we want to ensure that all possible algorithms on a given hardware are tested in order for
the autotuning histroy/telemetry to be as complete as possible. However, in some cases
you may want to allow missing algorithms, such as when running on a system where the
hardware may not have support due to driver versions or other issues. If that's the case,
you can set `skip_missing_algs = true` to allow missing algorithms without failing the autotune setup:

```julia
# Default behavior: error if expected algorithms are missing
results, sysinfo, _ = autotune_setup()  # Will error if RFLUFactorization missing

# Allow missing algorithms (useful for incomplete setups)
results, sysinfo, _ = autotune_setup(skip_missing_algs = true)  # Will warn instead of error
```

## GPU Systems

On systems with CUDA or Metal GPU support, the autotuner will automatically detect and benchmark GPU algorithms:

```julia
# Enable large matrix testing for GPUs
results, sysinfo, _ = autotune_setup(large_matrices = true, samples = 3, seconds = 1.0)
```

GPU algorithms tested (when available):
- **CudaOffloadFactorization**: CUDA GPU acceleration
- **MetalLUFactorization**: Apple Metal GPU acceleration

## Working with Results

### Examining Performance Data

```julia
using DataFrames
using Statistics

results, sysinfo, _ = autotune_setup(make_plot = false)

# Filter successful results
successful = filter(row -> row.success, results)

# Summary by algorithm
summary = combine(groupby(successful, [:algorithm, :eltype]), 
                 :gflops => mean => :avg_gflops,
                 :gflops => maximum => :max_gflops)
sort!(summary, :avg_gflops, rev=true)
println(summary)
```

### Performance Plots

When `make_plot=true`, you get separate plots for each element type:

```julia
results, sysinfo, plots = autotune_setup()

# plots is a dictionary keyed by element type
for (eltype, plot) in plots
    println("Plot for $eltype available")
    # Plots are automatically saved as PNG and PDF files
    display(plot)
end
```

### Preferences Integration

The autotuner sets preferences that LinearSolve.jl uses for automatic algorithm selection:

```julia
using LinearSolveAutotune

# View current preferences
LinearSolveAutotune.show_current_preferences()

# Clear all autotune preferences
LinearSolveAutotune.clear_algorithm_preferences()

# Set custom preferences
custom_categories = Dict(
    "Float64_0-128" => "RFLUFactorization",
    "Float64_128-256" => "LUFactorization"
)
LinearSolveAutotune.set_algorithm_preferences(custom_categories)
```

## How Preferences Affect LinearSolve.jl

!!! warn

    Usage of autotune preferences is still in development.

After running autotune, LinearSolve.jl will automatically use the optimal algorithms:

```julia
using LinearSolve

# This will now use the algorithm determined by autotune
A = rand(100, 100)  # Float64 matrix in 0-128 size range
b = rand(100)
prob = LinearProblem(A, b)
sol = solve(prob)  # Uses auto-selected optimal algorithm

# For different sizes, different optimal algorithms may be used
A_large = rand(300, 300)  # Different size range
b_large = rand(300)
prob_large = LinearProblem(A_large, b_large)
sol_large = solve(prob_large)  # May use different algorithm
```