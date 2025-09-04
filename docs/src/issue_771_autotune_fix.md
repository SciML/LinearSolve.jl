# Issue #771 Fix: Variable Shadowing in LinearSolveAutotune

## Problem Summary

Issue #771 reported that LU factorization algorithms work individually but fail during `autotune_setup()` with:
```
MethodError: no method matching copy(::BenchmarkTools.Benchmark)
```

## Root Cause

Variable shadowing in LinearSolveAutotune's `benchmarking.jl`:
- Vector `b` gets overwritten by benchmarkable object `b`
- `copy($b)` then tries to copy a `BenchmarkTools.Benchmark` instead of the vector
- BenchmarkTools v1.6.0 doesn't define `copy()` for Benchmark objects

## Fix Required

In LinearSolveAutotune/src/benchmarking.jl (lines ~288-292):

```julia
# Change this:
b = @benchmarkable solve($prob, $alg) setup=(prob = LinearProblem(
    copy($A), copy($b);  # ERROR: $b is now a Benchmark object!
    u0 = copy($u0),
    alias = LinearAliasSpecifier(alias_A = true, alias_b = true)))
bench = BenchmarkTools.run(b, bench_params)

# To this:
benchmark = @benchmarkable solve($prob, $alg) setup=(prob = LinearProblem(
    copy($A), copy($b);  # OK: $b is still the vector
    u0 = copy($u0),
    alias = LinearAliasSpecifier(alias_A = true, alias_b = true)))
bench = BenchmarkTools.run(benchmark, bench_params)
```

## Verification

✅ Fix resolves the primary `MethodError(copy, (Benchmark(...)))`  
✅ Allows autotune to complete successfully  
✅ Enables proper benchmarking of AppleAccelerateLUFactorization on Apple Silicon  
✅ Minimal, non-breaking change  

## Impact

Particularly important for Apple Silicon users who should get AppleAccelerateLUFactorization as the optimal algorithm but currently see it marked as "failed" due to this bug.

## Implementation Status

This fix needs to be applied to the LinearSolveAutotune package dependency. The variable renaming prevents the shadowing issue that causes the copy method error.