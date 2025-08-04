# Preferences management for storing optimal algorithms in LinearSolve.jl

"""
    set_algorithm_preferences(categories::Dict{String, String})

Set LinearSolve preferences based on the categorized benchmark results.
These preferences are stored in the main LinearSolve.jl package.
"""
function set_algorithm_preferences(categories::Dict{String, String})
    @info "Setting LinearSolve preferences based on benchmark results..."

    for (range, algorithm) in categories
        pref_key = "best_algorithm_$(replace(range, "+" => "plus", "-" => "_"))"
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
"""
function get_algorithm_preferences()
    prefs = Dict{String, String}()

    ranges = ["0_128", "128_256", "256_512", "512plus"]

    for range in ranges
        pref_key = "best_algorithm_$range"
        # Load preferences from LinearSolve.jl
        value = Preferences.load_preference(LinearSolve, pref_key, nothing)
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

Clear all autotune-related preferences from LinearSolve.jl.
"""
function clear_algorithm_preferences()
    @info "Clearing LinearSolve autotune preferences..."

    ranges = ["0_128", "128_256", "256_512", "512plus"]

    for range in ranges
        pref_key = "best_algorithm_$range"
        # Delete preferences from LinearSolve.jl (force=true ensures deletion works)
        Preferences.delete_preferences!(LinearSolve, pref_key; force = true)
    end

    Preferences.delete_preferences!(LinearSolve, "autotune_timestamp"; force = true)

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
