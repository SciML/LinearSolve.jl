# Automatic Algorithm Selection with LinearSolveAutotune

LinearSolve.jl includes an automatic tuning system that benchmarks all available linear algebra algorithms on your specific hardware and automatically selects optimal algorithms for different problem sizes and data types. This tutorial will show you how to use the `LinearSolveAutotune` sublibrary to optimize your linear solve performance.

## Quick Start

The simplest way to use the autotuner is to run it with default settings:

```julia
using LinearSolve
using LinearSolveAutotune

# Run autotune with default settings
results = autotune_setup()
```

This will:
- Benchmark 4 element types: `Float32`, `Float64`, `ComplexF32`, `ComplexF64`
- Test matrix sizes from small (4×4) to medium (500×500) 
- Create performance plots for each element type
- Set preferences for optimal algorithm selection
- Share results with the community (if desired)

## Understanding the Results

The autotune process returns benchmark results and creates several outputs:

```julia
# Basic usage returns just the DataFrame of results
results = autotune_setup(make_plot=false)

# With plotting enabled, returns (DataFrame, Plots)
results, plots = autotune_setup(make_plot=true)

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
results = autotune_setup(eltypes = (Float64, ComplexF64))

# Test arbitrary precision types (excludes BLAS algorithms)
results = autotune_setup(eltypes = (BigFloat,), telemetry = false)

# Test high precision float
results = autotune_setup(eltypes = (Float64, BigFloat))
```

### Matrix Sizes

Control the range of matrix sizes tested:

```julia
# Default: small to medium matrices (4×4 to 500×500)
results = autotune_setup(large_matrices = false)

# Large matrices: includes sizes up to 10,000×10,000 (good for GPU systems)
results = autotune_setup(large_matrices = true)
```

### Benchmark Quality vs Speed

Adjust the thoroughness of benchmarking:

```julia
# Quick benchmark (fewer samples, less time per test)
results = autotune_setup(samples = 1, seconds = 0.1)

# Thorough benchmark (more samples, more time per test)  
results = autotune_setup(samples = 10, seconds = 2.0)
```

### Privacy and Telemetry

Control data sharing:

```julia
# Disable telemetry (no data shared)
results = autotune_setup(telemetry = false)

# Disable preference setting (just benchmark, don't change defaults)
results = autotune_setup(set_preferences = false)

# Disable plotting (faster, less output)
results = autotune_setup(make_plot = false)
```

## Understanding Algorithm Compatibility

The autotuner automatically detects which algorithms work with which element types:

### Standard Types (Float32, Float64, ComplexF32, ComplexF64)
- **LUFactorization**: Fast BLAS-based LU decomposition
- **MKLLUFactorization**: Intel MKL optimized (if available)
- **AppleAccelerateLUFactorization**: Apple Accelerate optimized (on macOS)
- **RFLUFactorization**: Recursive factorization (cache-friendly)
- **GenericLUFactorization**: Pure Julia implementation
- **SimpleLUFactorization**: Simple pure Julia LU

### Arbitrary Precision Types (BigFloat, Rational, etc.)
Only pure Julia algorithms work:
- **GenericLUFactorization**: ✅ Compatible
- **RFLUFactorization**: ✅ Compatible  
- **SimpleLUFactorization**: ✅ Compatible
- **LUFactorization**: ❌ Excluded (requires BLAS)

## GPU Systems

On systems with CUDA or Metal GPU support, the autotuner will automatically detect and benchmark GPU algorithms:

```julia
# Enable large matrix testing for GPUs
results = autotune_setup(large_matrices = true, samples = 3, seconds = 1.0)
```

GPU algorithms tested (when available):
- **CudaOffloadFactorization**: CUDA GPU acceleration
- **MetalLUFactorization**: Apple Metal GPU acceleration

## Working with Results

### Examining Performance Data

```julia
using DataFrames
using Statistics

results = autotune_setup(make_plot = false)

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
results, plots = autotune_setup()

# plots is a dictionary keyed by element type
for (eltype, plot) in plots
    println("Plot for $eltype available")
    # Plots are automatically saved as PNG and PDF files
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

## Real-World Examples

### High-Performance Computing

```julia
# For HPC clusters with large problems
results = autotune_setup(
    large_matrices = true,
    samples = 5,
    seconds = 1.0,
    eltypes = (Float64, ComplexF64),
    telemetry = false  # Privacy on shared systems
)
```

### Workstation with GPU

```julia
# Comprehensive benchmark including GPU algorithms
results = autotune_setup(
    large_matrices = true,
    samples = 3,
    seconds = 0.5,
    eltypes = (Float32, Float64, ComplexF32, ComplexF64)
)
```

### Research with Arbitrary Precision

```julia
# Testing arbitrary precision arithmetic
results = autotune_setup(
    eltypes = (Float64, BigFloat),
    samples = 2,
    seconds = 0.2,  # BigFloat is slow
    telemetry = false,
    large_matrices = false
)
```

### Quick Development Testing

```julia
# Fast benchmark for development/testing
results = autotune_setup(
    samples = 1,
    seconds = 0.05,
    eltypes = (Float64,),
    make_plot = false,
    telemetry = false,
    set_preferences = false
)
```

## How Preferences Affect LinearSolve.jl

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

## Best Practices

1. **Run autotune once per system**: Results are system-specific and should be rerun when hardware changes.

2. **Use appropriate matrix sizes**: Set `large_matrices=true` only if you regularly solve large systems.

3. **Consider element types**: Only benchmark the types you actually use to save time.

4. **Benchmark thoroughly for production**: Use higher `samples` and `seconds` values for production systems.

5. **Respect privacy**: Disable telemetry on sensitive or proprietary systems.

6. **Save results**: The DataFrame returned contains valuable performance data for analysis.

## Troubleshooting

### No Algorithms Available
If you get "No algorithms found", ensure LinearSolve.jl is properly installed:
```julia
using Pkg
Pkg.test("LinearSolve")
```

### GPU Algorithms Missing
GPU algorithms require additional packages:
```julia
# For CUDA
using CUDA, LinearSolve

# For Metal (Apple Silicon)  
using Metal, LinearSolve
```

### Preferences Not Applied
Restart Julia after running autotune for preferences to take effect, or check:
```julia
LinearSolveAutotune.show_current_preferences()
```

### Slow BigFloat Performance
This is expected - arbitrary precision arithmetic is much slower than hardware floating point. Consider using `DoubleFloats.jl` or `MultiFloats.jl` for better performance if extreme precision isn't required.

## Community and Telemetry

By default, autotune results are shared with the LinearSolve.jl community to help improve algorithm selection for everyone. The shared data includes:

- System information (OS, CPU, core count, etc.)
- Algorithm performance results
- NO personal information or sensitive data

### GitHub Authentication for Telemetry

When telemetry is enabled, the system will check for GitHub authentication:

```julia
# This will show setup instructions if GITHUB_TOKEN not found
results = autotune_setup(telemetry = true)
```

**Quick Setup (30 seconds):**

1. **Create GitHub Token**: Open [https://github.com/settings/tokens?type=beta](https://github.com/settings/tokens?type=beta)
   - Click "Generate new token"
   - Name: "LinearSolve Autotune"
   - Expiration: 90 days (or longer)
   - Repository access: "Public Repositories (read-only)"
   - Generate and copy the token

2. **Set Environment Variable**:
   ```bash
   export GITHUB_TOKEN=paste_your_token_here
   ```

3. **Restart Julia** and run autotune again

That's it! Your results will automatically be shared to help the community.

### Disabling Telemetry

You can disable telemetry completely:

```julia
# No authentication required
results = autotune_setup(telemetry = false)
```

This helps the community understand performance across different hardware configurations and improves the default algorithm selection for future users, but participation is entirely optional.