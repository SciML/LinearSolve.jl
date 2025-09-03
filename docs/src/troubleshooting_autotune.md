# Troubleshooting AutoTune Issues

This guide helps resolve common issues with `autotune_setup()` where algorithms fail during benchmarking but work fine for individual solves.

## Common Symptoms

- Individual LU factorization algorithms (like `AppleAccelerateLUFactorization`, `OpenBLASLUFactorization`) work correctly
- `autotune_setup()` reports failures for multiple algorithms with "NaN" performance values
- Only `LUFactorization` succeeds in autotune benchmarking

## Root Causes and Solutions

### 1. Missing Dependencies

**Problem**: Some algorithms require specific packages to be loaded.

**Example Error**: 
```
RFLUFactorization requires that RecursiveFactorization.jl is loaded, i.e. `using RecursiveFactorization`
```

**Solution**: Load required dependencies before running autotune:
```julia
using RecursiveFactorization  # For RFLUFactorization
using LinearSolve, LinearSolveAutotune

results = autotune_setup()
```

### 2. Julia Version and Precompilation Issues (Fixed in LinearSolve v3.39.2+)

**Problem**: Method overwriting errors during module precompilation on Julia 1.10+.

**Example Error**:
```
ERROR: Method overwriting is not permitted during Module precompilation
```

**Solution**: Update LinearSolve.jl to version 3.39.2 or later:
```julia
using Pkg
Pkg.update("LinearSolve")
```

### 3. Platform-Specific Algorithm Availability

**Problem**: Some algorithms are only available on specific platforms.

**Examples**:
- `MetalLUFactorization` - Only on Apple platforms with Metal.jl
- `AppleAccelerateLUFactorization` - Only on macOS with Accelerate framework

**Solution**: This is expected behavior. Use `skip_missing_algs=true`:
```julia
results = autotune_setup(skip_missing_algs=true)
```

## Complete Setup Example

Here's a comprehensive setup that addresses most common issues:

```julia
using Pkg

# Update to latest versions
Pkg.update(["LinearSolve", "LinearSolveAutotune"])

# Load required dependencies
using LinearSolve, LinearSolveAutotune
using RecursiveFactorization  # For RFLUFactorization

# Optional: Load platform-specific dependencies
if Sys.isapple()
    try
        using Metal  # For MetalLUFactorization on Apple Silicon
    catch
        @warn "Metal.jl not available - MetalLUFactorization will be skipped"
    end
end

# Run autotune with appropriate settings
results = autotune_setup(
    sizes = [:tiny, :small, :medium, :large],  # Adjust based on your needs
    skip_missing_algs = true,                  # Skip unavailable algorithms
    maxtime = 30.0                            # Adjust timeout as needed
)

println(results)
```

## Environment Verification

To check your environment before running autotune:

```julia
using LinearSolve, Pkg

# Check versions
println("LinearSolve version: ", Pkg.installed()["LinearSolve"])
println("Julia version: ", VERSION)

# Test individual algorithms
n = 4
A = rand(n, n)
b = rand(n)
prob = LinearProblem(A, b)

algorithms_to_test = [
    ("LUFactorization", LUFactorization()),
    ("AppleAccelerateLUFactorization", AppleAccelerateLUFactorization()),
    ("OpenBLASLUFactorization", OpenBLASLUFactorization()),
    ("GenericLUFactorization", GenericLUFactorization()),
    ("SimpleLUFactorization", SimpleLUFactorization()),
]

for (name, alg) in algorithms_to_test
    try
        linsolve = init(prob, alg)
        result = solve!(linsolve)
        println("✅ $name: Working")
    catch e
        println("❌ $name: Failed - $e")
    end
end
```

## Performance Considerations

If autotune is taking too long or timing out:

```julia
# Reduce scope for faster testing
results = autotune_setup(
    sizes = [:tiny],          # Test only small matrices
    maxtime = 10.0,          # Reduce timeout per algorithm
    samples = 3              # Fewer samples per test
)
```

## Reporting Issues

If you continue to experience problems after following this guide:

1. Include your environment information:
   ```julia
   using Pkg; Pkg.status()
   versioninfo()
   ```

2. Test individual algorithms as shown in "Environment Verification"

3. Try running autotune with verbose output to identify specific failures:
   ```julia
   results = autotune_setup(sizes = [:tiny], maxtime = 5.0)
   ```

4. Report the issue at [LinearSolve.jl Issues](https://github.com/SciML/LinearSolve.jl/issues) or [LinearSolveAutotune.jl Issues](https://github.com/SciML/LinearSolveAutotune.jl/issues) with the above information.