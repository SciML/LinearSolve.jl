# Algorithm Selection Guide

LinearSolve.jl automatically selects appropriate algorithms based on your problem characteristics, but understanding how this works can help you make better choices for your specific use case.

## Automatic Algorithm Selection

When you call `solve(prob)` without specifying an algorithm, LinearSolve.jl uses intelligent heuristics to choose the best solver:

```julia
using LinearSolve

# LinearSolve.jl automatically chooses the best algorithm
A = rand(100, 100)
b = rand(100)
prob = LinearProblem(A, b)
sol = solve(prob)  # Automatic algorithm selection
```

The selection process considers:

- **Matrix type**: Dense vs. sparse vs. structured matrices
- **Matrix properties**: Square vs. rectangular, symmetric, positive definite
- **Size**: Small vs. large matrices for performance optimization  
- **Hardware**: CPU vs. GPU arrays
- **Conditioning**: Well-conditioned vs. ill-conditioned systems

## Algorithm Categories

LinearSolve.jl organizes algorithms into several categories:

### Factorization Methods

These algorithms decompose your matrix into simpler components:

- **Dense factorizations**: Best for matrices without special sparsity structure
  - `LUFactorization()`: General-purpose, good balance of speed and stability
  - `QRFactorization()`: More stable for ill-conditioned problems
  - `CholeskyFactorization()`: Fastest for symmetric positive definite matrices

- **Sparse factorizations**: Optimized for matrices with many zeros
  - `UMFPACKFactorization()`: General sparse LU with good fill-in control
  - `KLUFactorization()`: Optimized for circuit simulation problems

### Iterative Methods

These solve the system iteratively without explicit factorization:

- **Krylov methods**: Memory-efficient for large sparse systems
  - `KrylovJL_GMRES()`: General-purpose iterative solver
  - `KrylovJL_CG()`: For symmetric positive definite systems

### Direct Methods

Simple direct approaches:

- `DirectLdiv!()`: Uses Julia's built-in `\` operator
- `DiagonalFactorization()`: Optimized for diagonal matrices

## Performance Characteristics

### Dense Matrices

For dense matrices, algorithm choice depends on size and conditioning:

```julia
# Small matrices (< 100×100): SimpleLUFactorization often fastest
A_small = rand(50, 50)
sol = solve(LinearProblem(A_small, rand(50)), SimpleLUFactorization())

# Medium matrices (100×500): RFLUFactorization often optimal  
A_medium = rand(200, 200)
sol = solve(LinearProblem(A_medium, rand(200)), RFLUFactorization())

# Large matrices (> 500×500): MKLLUFactorization or AppleAccelerate
A_large = rand(1000, 1000) 
sol = solve(LinearProblem(A_large, rand(1000)), MKLLUFactorization())
```

### Sparse Matrices

For sparse matrices, structure matters:

```julia
using SparseArrays

# General sparse matrices
A_sparse = sprand(1000, 1000, 0.01)
sol = solve(LinearProblem(A_sparse, rand(1000)), UMFPACKFactorization())

# Structured sparse (e.g., from discretized PDEs)
# KLUFactorization often better for circuit-like problems
```

### GPU Acceleration

For very large problems, GPU offloading can be beneficial:

```julia
# Requires CUDA.jl
# A_gpu = CuArray(rand(Float32, 2000, 2000))
# sol = solve(LinearProblem(A_gpu, CuArray(rand(Float32, 2000))), 
#            CudaOffloadLUFactorization())
```

## When to Override Automatic Selection

You might want to manually specify an algorithm when:

1. **You know your problem structure**: E.g., you know your matrix is positive definite
   ```julia
   sol = solve(prob, CholeskyFactorization())  # Faster for SPD matrices
   ```

2. **You need maximum stability**: For ill-conditioned problems
   ```julia
   sol = solve(prob, QRFactorization())  # More numerically stable
   ```

3. **You're doing many solves**: Factorization methods amortize cost over multiple solves
   ```julia
   cache = init(prob, LUFactorization())
   for i in 1:1000
       cache.b = new_rhs[i]
       sol = solve!(cache)
   end
   ```

4. **Memory constraints**: Iterative methods use less memory
   ```julia
   sol = solve(prob, KrylovJL_GMRES())  # Lower memory usage
   ```

## Algorithm Selection Flowchart

The automatic selection roughly follows this logic:

```
Is A diagonal? → DiagonalFactorization
Is A tridiagonal/bidiagonal? → DirectLdiv! (Julia 1.11+) or LUFactorization  
Is A symmetric positive definite? → CholeskyFactorization
Is A symmetric indefinite? → BunchKaufmanFactorization
Is A sparse? → UMFPACKFactorization or KLUFactorization
Is A small dense? → RFLUFactorization or SimpleLUFactorization
Is A large dense? → MKLLUFactorization or AppleAccelerateLUFactorization
Is A GPU array? → QRFactorization or LUFactorization
Is A an operator/function? → KrylovJL_GMRES
Is the system overdetermined? → QRFactorization or KrylovJL_LSMR
```

## Custom Functions

For specialized algorithms not covered by the built-in solvers:

```julia
function my_custom_solver(A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)
    # Your custom solving logic here
    return A \ b  # Simple example
end

sol = solve(prob, LinearSolveFunction(my_custom_solver))
```

See the [Custom Linear Solvers](@ref custom) section for more details.

## Tuned Algorithm Selection

LinearSolve.jl includes a sophisticated preference system that can be tuned using LinearSolveAutotune for optimal performance on your specific hardware:

```julia
using LinearSolve
using LinearSolveAutotune

# Run autotune to benchmark algorithms and set preferences
results = autotune_setup(set_preferences = true)

# View what algorithms are now being chosen
show_algorithm_choices()
```

The system automatically sets preferences for:
- **Different matrix sizes**: tiny (≤20), small (21-100), medium (101-300), large (301-1000), big (>1000)
- **Different element types**: Float32, Float64, ComplexF32, ComplexF64
- **Dual preferences**: Best overall algorithm + best always-available fallback

### Viewing Algorithm Choices

Use `show_algorithm_choices()` to see what algorithms are currently being selected:

```julia
using LinearSolve
show_algorithm_choices()
```

This shows:
- Current autotune preferences (if set)
- Algorithm choices for each size category
- System information (available extensions)
- Element type behavior

### Preference System Benefits

- **Automatic optimization**: Uses the fastest algorithms found by benchmarking
- **Intelligent fallbacks**: Falls back to always-available algorithms when extensions aren't loaded
- **Size-specific tuning**: Different algorithms optimized for different matrix sizes
- **Type-specific tuning**: Optimized algorithm selection for different numeric types