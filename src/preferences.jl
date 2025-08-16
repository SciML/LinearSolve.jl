# Preference system for autotune algorithm selection

using Preferences

# Helper function to convert algorithm name string to DefaultAlgorithmChoice enum
function _string_to_algorithm_choice(algorithm_name::Union{String, Nothing})
    algorithm_name === nothing && return nothing
    
    # Core LU algorithms from LinearSolveAutotune
    if algorithm_name == "LUFactorization"
        return DefaultAlgorithmChoice.LUFactorization
    elseif algorithm_name == "GenericLUFactorization"
        return DefaultAlgorithmChoice.GenericLUFactorization
    elseif algorithm_name == "RFLUFactorization" || algorithm_name == "RecursiveFactorization"
        return DefaultAlgorithmChoice.RFLUFactorization
    elseif algorithm_name == "MKLLUFactorization"
        return DefaultAlgorithmChoice.MKLLUFactorization
    elseif algorithm_name == "AppleAccelerateLUFactorization"
        return DefaultAlgorithmChoice.AppleAccelerateLUFactorization
    elseif algorithm_name == "SimpleLUFactorization"
        return DefaultAlgorithmChoice.LUFactorization  # Map to standard LU
    elseif algorithm_name == "FastLUFactorization"
        return DefaultAlgorithmChoice.LUFactorization  # Map to standard LU (FastLapack extension)
    elseif algorithm_name == "BLISLUFactorization"
        return DefaultAlgorithmChoice.LUFactorization  # Map to standard LU (BLIS extension)
    elseif algorithm_name == "CudaOffloadLUFactorization"
        return DefaultAlgorithmChoice.LUFactorization  # Map to standard LU (CUDA extension)
    elseif algorithm_name == "MetalLUFactorization"
        return DefaultAlgorithmChoice.LUFactorization  # Map to standard LU (Metal extension)
    elseif algorithm_name == "AMDGPUOffloadLUFactorization"
        return DefaultAlgorithmChoice.LUFactorization  # Map to standard LU (AMDGPU extension)
    else
        @warn "Unknown algorithm preference: $algorithm_name, falling back to heuristics"
        return nothing
    end
end

# Load autotune preferences as constants for each element type and size category
# Support both best overall algorithm and best always-loaded algorithm as fallback
const AUTOTUNE_PREFS = (
    Float32 = (
        tiny = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float32_tiny", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float32_tiny", nothing))
        ),
        small = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float32_small", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float32_small", nothing))
        ),
        medium = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float32_medium", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float32_medium", nothing))
        ),
        large = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float32_large", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float32_large", nothing))
        ),
        big = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float32_big", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float32_big", nothing))
        )
    ),
    Float64 = (
        tiny = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float64_tiny", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float64_tiny", nothing))
        ),
        small = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float64_small", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float64_small", nothing))
        ),
        medium = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float64_medium", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float64_medium", nothing))
        ),
        large = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float64_large", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float64_large", nothing))
        ),
        big = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_Float64_big", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_Float64_big", nothing))
        )
    ),
    ComplexF32 = (
        tiny = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF32_tiny", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF32_tiny", nothing))
        ),
        small = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF32_small", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF32_small", nothing))
        ),
        medium = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF32_medium", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF32_medium", nothing))
        ),
        large = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF32_large", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF32_large", nothing))
        ),
        big = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF32_big", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF32_big", nothing))
        )
    ),
    ComplexF64 = (
        tiny = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF64_tiny", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF64_tiny", nothing))
        ),
        small = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF64_small", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF64_small", nothing))
        ),
        medium = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF64_medium", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF64_medium", nothing))
        ),
        large = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF64_large", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF64_large", nothing))
        ),
        big = (
            best = _string_to_algorithm_choice(Preferences.@load_preference("best_algorithm_ComplexF64_big", nothing)),
            fallback = _string_to_algorithm_choice(Preferences.@load_preference("best_always_loaded_ComplexF64_big", nothing))
        )
    )
)

# Fast path: check if any autotune preferences are actually set
const AUTOTUNE_PREFS_SET = let
    any_set = false
    for type_prefs in (AUTOTUNE_PREFS.Float32, AUTOTUNE_PREFS.Float64, AUTOTUNE_PREFS.ComplexF32, AUTOTUNE_PREFS.ComplexF64)
        for size_pref in (type_prefs.tiny, type_prefs.small, type_prefs.medium, type_prefs.large, type_prefs.big)
            if size_pref.best !== nothing || size_pref.fallback !== nothing
                any_set = true
                break
            end
        end
        any_set && break
    end
    any_set
end

# Testing mode flag
const TESTING_MODE = Ref(false)

"""
    reset_defaults!()

**Internal function for testing only.** Enables testing mode where preferences
are checked at runtime instead of using compile-time constants. This allows
tests to verify that the preference system works correctly.

!!! warning "Testing Only"
    This function is only intended for internal testing purposes. It modifies
    global state and should never be used in production code.
"""
function reset_defaults!()
    # Enable testing mode to use runtime preference checking
    TESTING_MODE[] = true
    return nothing
end

# Helper function to choose available algorithm with fallback logic
@inline function _choose_available_algorithm(prefs)
    # Try the best algorithm first
    if prefs.best !== nothing && is_algorithm_available(prefs.best)
        return prefs.best
    end
    
    # Fall back to always-loaded algorithm if best is not available
    if prefs.fallback !== nothing && is_algorithm_available(prefs.fallback)
        return prefs.fallback
    end
    
    # No tuned algorithms available
    return nothing
end

# Runtime preference checking for testing
function _get_tuned_algorithm_runtime(target_eltype::Type, size_category::Symbol)
    eltype_str = string(target_eltype)
    size_str = string(size_category)
    
    # Load preferences at runtime
    best_pref = Preferences.load_preference(LinearSolve, "best_algorithm_$(eltype_str)_$(size_str)", nothing)
    fallback_pref = Preferences.load_preference(LinearSolve, "best_always_loaded_$(eltype_str)_$(size_str)", nothing)
    
    if best_pref !== nothing || fallback_pref !== nothing
        # Convert to algorithm choices
        best_alg = _string_to_algorithm_choice(best_pref)
        fallback_alg = _string_to_algorithm_choice(fallback_pref)
        
        # Create preference structure
        prefs = (best = best_alg, fallback = fallback_alg)
        return _choose_available_algorithm(prefs)
    end
    
    return nothing
end