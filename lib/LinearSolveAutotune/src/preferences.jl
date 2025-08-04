# Preferences management for storing optimal algorithms

"""
    set_algorithm_preferences(categories::Dict{String, String})

Set LinearSolve preferences based on the categorized benchmark results.
"""
function set_algorithm_preferences(categories::Dict{String, String})
    @info "Setting LinearSolve preferences based on benchmark results..."

    for (range, algorithm) in categories
        pref_key = "best_algorithm_$(replace(range, "+" => "plus", "-" => "_"))"
        @set_preferences!(pref_key => algorithm)
        @info "Set preference $pref_key = $algorithm"
    end

    # Set a timestamp for when these preferences were created
    @set_preferences!("autotune_timestamp" => string(Dates.now()))

    @info "Preferences updated. You may need to restart Julia for changes to take effect."
end

"""
    get_algorithm_preferences()

Get the current algorithm preferences.
"""
function get_algorithm_preferences()
    prefs = Dict{String, String}()

    ranges = ["0_128", "128_256", "256_512", "512plus"]

    for range in ranges
        pref_key = "best_algorithm_$range"
        value = @load_preference(pref_key, nothing)
        if value !== nothing
            # Convert back to human-readable range name
            readable_range = replace(range, "_" => "-", "plus" => "+")
            prefs[readable_range] = value
        end
    end

    return prefs
end

"""
    clear_algorithm_preferences()

Clear all autotune-related preferences.
"""
function clear_algorithm_preferences()
    @info "Clearing LinearSolve autotune preferences..."

    ranges = ["0_128", "128_256", "256_512", "512plus"]

    for range in ranges
        pref_key = "best_algorithm_$range"
        @delete_preferences!(pref_key)
    end

    @delete_preferences!("autotune_timestamp")

    @info "Preferences cleared."
end

"""
    show_current_preferences()

Display the current algorithm preferences in a readable format.
"""
function show_current_preferences()
    prefs = get_algorithm_preferences()

    if isempty(prefs)
        println("No autotune preferences currently set.")
        return
    end

    println("Current LinearSolve autotune preferences:")
    println("="^50)

    for (range, algorithm) in sort(prefs)
        println("  Size range $range: $algorithm")
    end

    timestamp = @load_preference("autotune_timestamp", "unknown")
    println("  Last updated: $timestamp")
end
