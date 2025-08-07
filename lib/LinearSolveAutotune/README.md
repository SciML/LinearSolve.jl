# LinearSolveAutotune.jl

Automatic benchmarking and tuning for LinearSolve.jl algorithms.

## Quick Start

```julia
using LinearSolve, LinearSolveAutotune

# Run benchmarks with default settings (small, medium, and large sizes)
results = autotune_setup()

# View a summary of results
display(results)

# Plot all benchmark results
plot(results)

# Share your results with the community (optional)
share_results(results)
```

## Features

- **Automatic Algorithm Benchmarking**: Tests all available LU factorization methods
- **Multi-size Testing**: Flexible size categories from small to very large matrices
- **Element Type Support**: Tests with Float32, Float64, ComplexF32, ComplexF64
- **GPU Support**: Automatically detects and benchmarks GPU algorithms if available
- **Performance Visualization**: Creates plots showing algorithm performance
- **Community Sharing**: Optional telemetry to help improve algorithm selection

## Size Categories

The package now uses flexible size categories instead of a binary large_matrices flag:

- `:small` - Matrices from 5×5 to 20×20 (quick tests)
- `:medium` - Matrices from 20×20 to 300×300 (typical problems)
- `:large` - Matrices from 300×300 to 1000×1000 (larger problems)
- `:big` - Matrices from 10000×10000 to 100000×100000 (GPU/HPC)

## Usage Examples

### Basic Benchmarking

```julia
# Default: small, medium, and large sizes
results = autotune_setup()

# Test all size ranges
results = autotune_setup(sizes = [:small, :medium, :large, :big])

# Large matrices only (for GPU systems)
results = autotune_setup(sizes = [:large, :big])

# Custom configuration
results = autotune_setup(
    sizes = [:medium, :large],
    samples = 10,
    seconds = 1.0,
    eltypes = (Float64, ComplexF64)
)

# View results and plot
display(results)
plot(results)
```

### Sharing Results

After running benchmarks, you can optionally share your results with the LinearSolve.jl community to help improve automatic algorithm selection:

```julia
# Share your benchmark results
share_results(results)
```

## Setting Up GitHub Authentication

To share results, you need GitHub authentication. We recommend using the GitHub CLI:

### Method 1: GitHub CLI (Recommended)

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

### Method 2: GitHub Personal Access Token

If you prefer using a token:

1. Go to [GitHub Settings > Tokens](https://github.com/settings/tokens/new)
2. Add description: "LinearSolve.jl Telemetry"
3. Select scope: `public_repo`
4. Click "Generate token" and copy it
5. In Julia:
   ```julia
   ENV["GITHUB_TOKEN"] = "your_token_here"
   share_results(results, sysinfo, plots)
   ```

## How It Works

1. **Benchmarking**: The `autotune_setup()` function runs comprehensive benchmarks of all available LinearSolve.jl algorithms across different matrix sizes and element types.

2. **Analysis**: Results are analyzed to find the best-performing algorithm for each size range and element type combination.

3. **Preferences**: Optionally sets Julia preferences to automatically use the best algorithms for your system.

4. **Sharing**: The `share_results()` function allows you to contribute your benchmark data to the community collection at [LinearSolve.jl Issue #669](https://github.com/SciML/LinearSolve.jl/issues/669).

## Privacy and Telemetry

- Sharing results is **completely optional**
- Only benchmark performance data and system specifications are shared
- No personal information is collected
- All shared data is publicly visible on GitHub
- You can review the exact data before sharing

## API Reference

### `autotune_setup`

```julia
autotune_setup(;
    sizes = [:small, :medium, :large],
    make_plot = true,
    set_preferences = true,
    samples = 5,
    seconds = 0.5,
    eltypes = (Float32, Float64, ComplexF32, ComplexF64),
    skip_missing_algs = false
)
```

**Parameters:**
- `sizes`: Vector of size categories to test
- `make_plot`: Generate performance plots
- `set_preferences`: Update LinearSolve preferences
- `samples`: Number of benchmark samples per test
- `seconds`: Maximum time per benchmark
- `eltypes`: Element types to benchmark
- `skip_missing_algs`: Continue if algorithms are missing

**Returns:**
- `results`: AutotuneResults object containing benchmark data, system info, and plots

### `share_results`

```julia
share_results(results)
```

**Parameters:**
- `results`: AutotuneResults object from `autotune_setup`

## Contributing

Your benchmark contributions help improve LinearSolve.jl for everyone! By sharing results from diverse hardware configurations, we can build better automatic algorithm selection heuristics.

## License

Part of the SciML ecosystem. See LinearSolve.jl for license information.