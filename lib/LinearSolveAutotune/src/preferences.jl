# Preferences management for storing optimal algorithms in LinearSolve.jl

"""
    set_algorithm_preferences(categories::Dict{String, String})

Set LinearSolve preferences based on the categorized benchmark results.
These preferences are stored in the main LinearSolve.jl package.
Handles element type-specific preferences with keys like "Float64_0-128".
"""
function set_algorithm_preferences(categories::Dict{String, String})
    @info "Setting LinearSolve preferences based on benchmark results..."

    for (category_key, algorithm) in categories
        # Handle element type specific keys like "Float64_0-128"
        # Convert to safe preference key format
        pref_key = "best_algorithm_$(replace(category_key, "+" => "plus", "-" => "_"))"
        
        # Set preferences in LinearSolve.jl, not LinearSolveAutotune (force=true allows overwriting)
        Preferences.set_preferences!(LinearSolve, pref_key => algorithm; force = true)
        @info "Set preference $pref_key = $algorithm in LinearSolve.jl"
    end

    # Set a timestamp for when these preferences were created
    Preferences.set_preferences!(LinearSolve, "autotune_timestamp" => string(Dates.now()); force = true)

    @info "Preferences updated in LinearSolve.jl. You may need to restart Julia for changes to take effect."
end

"""
    get_algorithm_preferences()

Get the current algorithm preferences from LinearSolve.jl.
Handles both legacy and element type-specific preferences.
"""
function get_algorithm_preferences()
    prefs = Dict{String, String}()

    # Get all LinearSolve preferences by checking common preference patterns
    # Since there's no direct way to get all preferences, we'll check for known patterns
    common_patterns = [
        # Element type + size range combinations
        "Float64_0_128", "Float64_128_256", "Float64_256_512", "Float64_512plus",
        "Float32_0_128", "Float32_128_256", "Float32_256_512", "Float32_512plus", 
        "ComplexF64_0_128", "ComplexF64_128_256", "ComplexF64_256_512", "ComplexF64_512plus",
        "ComplexF32_0_128", "ComplexF32_128_256", "ComplexF32_256_512", "ComplexF32_512plus",
        "BigFloat_0_128", "BigFloat_128_256", "BigFloat_256_512", "BigFloat_512plus",
        # Legacy patterns without element type
        "0_128", "128_256", "256_512", "512plus"
    ]
    
    for pattern in common_patterns
        pref_key = "best_algorithm_$pattern"
        value = Preferences.load_preference(LinearSolve, pref_key, nothing)
        if value !== nothing
            # Convert back to human-readable key
            readable_key = replace(pattern, "_" => "-", "plus" => "+")
            prefs[readable_key] = value
        end
    end

    return prefs
end

"""
    clear_algorithm_preferences()

Clear all autotune-related preferences from LinearSolve.jl.
Handles both legacy and element type-specific preferences.
"""
function clear_algorithm_preferences()
    @info "Clearing LinearSolve autotune preferences..."

    # Clear known preference patterns
    common_patterns = [
        # Element type + size range combinations
        "Float64_0_128", "Float64_128_256", "Float64_256_512", "Float64_512plus",
        "Float32_0_128", "Float32_128_256", "Float32_256_512", "Float32_512plus", 
        "ComplexF64_0_128", "ComplexF64_128_256", "ComplexF64_256_512", "ComplexF64_512plus",
        "ComplexF32_0_128", "ComplexF32_128_256", "ComplexF32_256_512", "ComplexF32_512plus",
        "BigFloat_0_128", "BigFloat_128_256", "BigFloat_256_512", "BigFloat_512plus",
        # Legacy patterns without element type
        "0_128", "128_256", "256_512", "512plus"
    ]
    
    for pattern in common_patterns
        pref_key = "best_algorithm_$pattern"
        # Check if preference exists before trying to delete
        if Preferences.has_preference(LinearSolve, pref_key)
            Preferences.delete_preferences!(LinearSolve, pref_key; force = true)
            @info "Cleared preference: $pref_key"
        end
    end

    # Clear timestamp
    if Preferences.has_preference(LinearSolve, "autotune_timestamp")
        Preferences.delete_preferences!(LinearSolve, "autotune_timestamp"; force = true)
    end

    @info "Preferences cleared from LinearSolve.jl."
end

"""
    show_current_preferences()

Display the current algorithm preferences from LinearSolve.jl in a readable format.
"""
function show_current_preferences()
    prefs = get_algorithm_preferences()

    if isempty(prefs)
        println("No autotune preferences currently set in LinearSolve.jl.")
        return
    end

    println("Current LinearSolve.jl autotune preferences:")
    println("="^50)

    for (range, algorithm) in sort(prefs)
        println("  Size range $range: $algorithm")
    end

    timestamp = Preferences.load_preference(LinearSolve, "autotune_timestamp", "unknown")
    println("  Last updated: $timestamp")
end
