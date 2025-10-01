# Internal API Documentation

This page documents LinearSolve.jl's internal API, which is useful for developers who want to understand the package's architecture, contribute to the codebase, or develop custom linear solver algorithms.

## Abstract Type Hierarchy

LinearSolve.jl uses a well-structured type hierarchy to organize different classes of linear solver algorithms:

```@docs
LinearSolve.SciMLLinearSolveAlgorithm
LinearSolve.AbstractFactorization
LinearSolve.AbstractDenseFactorization
LinearSolve.AbstractSparseFactorization
LinearSolve.AbstractKrylovSubspaceMethod
LinearSolve.AbstractSolveFunction
```

## Core Cache System

The caching system is central to LinearSolve.jl's performance and functionality:

```@docs
LinearSolve.LinearCache
LinearSolve.init_cacheval
```

## Algorithm Selection

The automatic algorithm selection is one of LinearSolve.jl's key features:

```@docs
LinearSolve.defaultalg
LinearSolve.get_tuned_algorithm
LinearSolve.is_algorithm_available
LinearSolve.show_algorithm_choices
LinearSolve.make_preferences_dynamic!
```

### Preference System Architecture

The dual preference system provides intelligent algorithm selection with comprehensive fallbacks:

#### **Core Functions**
- **`get_tuned_algorithm`**: Retrieves tuned algorithm preferences based on matrix size and element type
- **`is_algorithm_available`**: Checks if a specific algorithm is currently available (extensions loaded)  
- **`show_algorithm_choices`**: Analysis function displaying algorithm choices for all element types
- **`make_preferences_dynamic!`**: Testing function that enables runtime preference checking

#### **Size Categorization**
The system categorizes matrix sizes to match LinearSolveAutotune benchmarking:
- **tiny**: ≤20 elements (matrices ≤10 always override to GenericLU)
- **small**: 21-100 elements  
- **medium**: 101-300 elements
- **large**: 301-1000 elements
- **big**: >1000 elements

#### **Dual Preference Structure**
For each category and element type (Float32, Float64, ComplexF32, ComplexF64):
- `best_algorithm_{type}_{size}`: Overall fastest algorithm from autotune
- `best_always_loaded_{type}_{size}`: Fastest always-available algorithm (fallback)

#### **Preference File Organization**
All preference-related functionality is consolidated in `src/preferences.jl`:

**Compile-Time Constants**:
- `AUTOTUNE_PREFS`: Preference structure loaded at package import
- `AUTOTUNE_PREFS_SET`: Fast path check for whether any preferences are set
- `_string_to_algorithm_choice`: Mapping from preference strings to algorithm enums

**Runtime Functions**:
- `_get_tuned_algorithm_runtime`: Dynamic preference checking for testing
- `_choose_available_algorithm`: Algorithm availability and fallback logic
- `show_algorithm_choices`: Comprehensive analysis and display function

**Testing Infrastructure**:
- `make_preferences_dynamic!`: Eval-based function redefinition for testing
- Enables runtime preference verification without affecting production performance

#### **Testing Mode Operation**
The testing system uses an elegant eval-based approach:
```julia
# Production: Uses compile-time constants (maximum performance)
get_tuned_algorithm(Float64, Float64, 200)  # → Uses AUTOTUNE_PREFS constants

# Testing: Redefines function to use runtime checking
make_preferences_dynamic!()
get_tuned_algorithm(Float64, Float64, 200)  # → Uses runtime preference loading
```

This approach maintains type stability and inference while enabling comprehensive testing.

#### **Algorithm Support Scope**
The preference system focuses exclusively on LU algorithms for dense matrices:

**Supported LU Algorithms**:
- `LUFactorization`, `GenericLUFactorization`, `RFLUFactorization`
- `MKLLUFactorization`, `AppleAccelerateLUFactorization`
- `SimpleLUFactorization`, `FastLUFactorization` (both map to LU)
- GPU LU variants (CUDA, Metal, AMDGPU - all map to LU)

**Non-LU algorithms** (QR, Cholesky, SVD, etc.) are not included in the preference system
as they serve different use cases and are not typically the focus of dense matrix autotune optimization.

## Trait Functions

These trait functions help determine algorithm capabilities and requirements:

```@docs
LinearSolve.needs_concrete_A
```

## Utility Functions

Various utility functions support the core functionality:

```@docs
LinearSolve.default_tol
LinearSolve.default_alias_A
LinearSolve.default_alias_b
LinearSolve.__init_u0_from_Ab
```

## Solve Functions

For custom solving strategies:

```@docs
LinearSolve.LinearSolveFunction
LinearSolve.DirectLdiv!
```

## Preconditioner Infrastructure

The preconditioner system allows for flexible preconditioning strategies:

```@docs
LinearSolve.ComposePreconditioner
LinearSolve.InvPreconditioner
```

## Internal Algorithm Types

These are internal algorithm implementations:

```@docs
LinearSolve.SimpleLUFactorization
LinearSolve.LUSolver
```

## Developer Notes

### Adding New Algorithms

When adding a new linear solver algorithm to LinearSolve.jl:

1. **Choose the appropriate abstract type**: Inherit from the most specific abstract type that fits your algorithm
2. **Implement required methods**: At minimum, implement `solve!` and possibly `init_cacheval`
3. **Consider trait functions**: Override trait functions like `needs_concrete_A` if needed
4. **Document thoroughly**: Add comprehensive docstrings following the patterns shown here

### Performance Considerations

- The `LinearCache` system is designed for efficient repeated solves
- Use `cache.isfresh` to avoid redundant computations when the matrix hasn't changed
- Consider implementing specialized `init_cacheval` for algorithms that need setup
- Leverage trait functions to optimize dispatch and memory usage

### Testing Guidelines

When adding new functionality:

- Test with various matrix types (dense, sparse, GPU arrays)
- Verify caching behavior works correctly
- Ensure trait functions return appropriate values
- Test integration with the automatic algorithm selection system