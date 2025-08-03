# LinearSolve.jl BLIS Benchmark

This directory contains a comprehensive benchmark script for testing the performance of various LU factorization algorithms in LinearSolve.jl, including the new BLIS integration.

## Quick Start

```bash
julia --project benchmark_blis.jl
```

This will:
1. Automatically detect available implementations (BLIS, MKL, Apple Accelerate, etc.)
2. Run benchmarks on matrix sizes from 4×4 to 256×256  
3. Generate a performance plot saved as `lu_factorization_benchmark.png`
4. Display results in both console output and a summary table

**Note**: The PNG plot file cannot be included in this gist due to GitHub's binary file restrictions, but it will be generated locally when you run the benchmark.

## What Gets Benchmarked

The script automatically detects and includes algorithms based on what's available, following LinearSolve.jl's detection patterns:

- **LU (OpenBLAS)**: Default BLAS-based LU factorization
- **RecursiveFactorization**: High-performance pure Julia implementation  
- **BLIS**: New BLIS-based implementation (requires `blis_jll` and `LAPACK_jll`)
- **Intel MKL**: Intel's optimized library (automatically detected on x86_64/i686, excludes EPYC CPUs by default)
- **Apple Accelerate**: Apple's framework (macOS only, checks for Accelerate.framework availability)
- **FastLU**: FastLapackInterface.jl implementation (if available)

### Detection Logic

The benchmark uses the same detection patterns as LinearSolve.jl:

- **MKL**: Enabled on x86_64/i686 architectures, disabled on AMD EPYC by default
- **Apple Accelerate**: Checks for macOS and verifies Accelerate.framework can be loaded with required symbols
- **BLIS**: Attempts to load blis_jll and LAPACK_jll, verifies extension loading
- **FastLU**: Attempts to load FastLapackInterface.jl package

## Requirements

### Essential Dependencies
```julia
using Pkg
Pkg.add(["BenchmarkTools", "Plots", "RecursiveFactorization"])
```

### Optional Dependencies for Full Testing
```julia
# For BLIS support
Pkg.add(["blis_jll", "LAPACK_jll"])

# For FastLU support  
Pkg.add("FastLapackInterface")
```

## Sample Output

```
============================================================
LinearSolve.jl LU Factorization Benchmark with BLIS
============================================================

System Information:
  Julia Version: 1.11.6
  OS: Linux x86_64
  CPU Threads: 1
  BLAS Threads: 1
  BLAS Config: LBTConfig([ILP64] libopenblas64_.so)

Available Implementations:
  BLIS: true
  MKL: false  
  Apple Accelerate: false

Results Summary (GFLOPs):
------------------------------------------------------------
Size    LU (OpenBLAS)   RecursiveFactorization  BLIS
4       0.05            0.09                    0.03
8       0.28            0.43                    0.09
16      0.61            1.28                    0.31
32      1.67            4.17                    1.09
64      4.0             9.52                    2.5
128     9.87            16.86                   8.1
256     17.33           28.16                   9.62
```

## Performance Notes

- **RecursiveFactorization** typically performs best for smaller matrices (< 500×500)
- **BLIS** provides an alternative BLAS implementation with different performance characteristics
- **Apple Accelerate** and **Intel MKL** may show significant advantages on supported platforms
- Single-threaded benchmarks are used for consistent comparison

## Customization

You can modify the benchmark by editing `benchmark_blis.jl`:

- **Matrix sizes**: Change the `sizes` parameter in `benchmark_lu_factorizations()`
- **Benchmark parameters**: Adjust `BenchmarkTools` settings (samples, evaluations)
- **Algorithms**: Add/remove algorithms in `build_algorithm_list()`

## Understanding the Results

- **GFLOPs**: Billions of floating-point operations per second (higher is better)
- **Performance scaling**: Look for algorithms that maintain high GFLOPs as matrix size increases
- **Platform differences**: Results vary significantly between systems based on hardware and BLAS libraries

## Integration with SciMLBenchmarks

This benchmark follows the same structure as the [official SciMLBenchmarks LU factorization benchmark](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/LinearSolve/LUFactorization/), making it easy to compare results and contribute to the broader benchmark suite.