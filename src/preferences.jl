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


"""
    make_preferences_dynamic!()

**Internal function for testing only.** Makes preferences dynamic by redefining
get_tuned_algorithm to check preferences at runtime instead of using compile-time
constants. This allows tests to verify that the preference system works correctly.

!!! warning "Testing Only"
    This function is only intended for internal testing purposes. It modifies
    global state and should never be used in production code.
"""
function make_preferences_dynamic!()
    # Redefine get_tuned_algorithm to use runtime preference checking for testing
    @eval function get_tuned_algorithm(::Type{eltype_A}, ::Type{eltype_b}, matrix_size::Integer) where {eltype_A, eltype_b}
        # Determine the element type to use for preference lookup
        target_eltype = eltype_A !== Nothing ? eltype_A : eltype_b
        
        # Determine size category based on matrix size (matching LinearSolveAutotune categories)
        size_category = if matrix_size <= 20
            :tiny
        elseif matrix_size <= 100
            :small
        elseif matrix_size <= 300
            :medium
        elseif matrix_size <= 1000
            :large
        else
            :big
        end
        
        # Use runtime preference checking for testing
        return _get_tuned_algorithm_runtime(target_eltype, size_category)
    end
    
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

"""
    show_algorithm_choices()

Display what algorithm choices are actually made by the default solver for 
representative matrix sizes. Shows current preferences and system information.
"""
function show_algorithm_choices()
    println("="^60)
    println("LinearSolve.jl Algorithm Choice Analysis")
    println("="^60)
    
    # Show current preferences for all element types
    println("📋 Current Preferences:")
    println("-"^60)
    
    any_prefs_set = false
    for eltype in ["Float32", "Float64", "ComplexF32", "ComplexF64"]
        for size_cat in ["tiny", "small", "medium", "large", "big"]
            best_pref = Preferences.load_preference(LinearSolve, "best_algorithm_$(eltype)_$(size_cat)", nothing)
            fallback_pref = Preferences.load_preference(LinearSolve, "best_always_loaded_$(eltype)_$(size_cat)", nothing)
            
            if best_pref !== nothing || fallback_pref !== nothing
                any_prefs_set = true
                println("$(eltype) $(size_cat):")
                if best_pref !== nothing
                    println("  Best: $(best_pref)")
                end
                if fallback_pref !== nothing
                    println("  Always-loaded: $(fallback_pref)")
                end
            end
        end
    end
    
    if !any_prefs_set
        println("No autotune preferences currently set.")
    end
    
    # Show algorithm choices for all element types and all sizes
    println("\n📊 Default Algorithm Choices:")
    println("-"^80)
    println("Size       Category    Float32            Float64            ComplexF32         ComplexF64")
    println("-"^80)
    
    # One representative size per category
    test_cases = [
        (8, "tiny"),      # ≤10 override
        (50, "small"),    # 21-100
        (200, "medium"),  # 101-300
        (500, "large"),   # 301-1000
        (1500, "big")     # >1000
    ]
    
    for (size, expected_category) in test_cases
        size_str = lpad("$(size)×$(size)", 10)
        cat_str = rpad(expected_category, 11)
        
        # Get algorithm choice for each element type
        alg_choices = []
        for eltype in [Float32, Float64, ComplexF32, ComplexF64]
            A = rand(eltype, size, size) + I(size)
            b = rand(eltype, size)
            chosen_alg = defaultalg(A, b, OperatorAssumptions(true))
            push!(alg_choices, rpad(string(chosen_alg.alg), 18))
        end
        
        println("$(size_str) $(cat_str) $(alg_choices[1]) $(alg_choices[2]) $(alg_choices[3]) $(alg_choices[4])")
    end
    
    # Show system information
    println("\n🖥️  System Information:")
    println("-"^60)
    println("MKL available: ", usemkl)
    println("Apple Accelerate available: ", appleaccelerate_isavailable())
    println("RecursiveFactorization enabled: ", userecursivefactorization(nothing))
    
    println("\n💡 Size Categories:")
    println("tiny (≤20), small (21-100), medium (101-300), large (301-1000), big (>1000)")
    println("Matrices ≤10 elements always use GenericLUFactorization override")
    
    println("="^60)
end