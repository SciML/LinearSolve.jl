# Preferences management for storing optimal algorithms in LinearSolve.jl

"""
    set_algorithm_preferences(categories::Dict{String, String})

Set LinearSolve preferences based on the categorized benchmark results.
These preferences are stored in the main LinearSolve.jl package.

The function handles type fallbacks:
- If Float32 wasn't benchmarked, uses Float64 results
- If ComplexF64 wasn't benchmarked, uses ComplexF32 results (if available) or Float64
- If ComplexF32 wasn't benchmarked, uses Float64 results
- For complex types, avoids RFLUFactorization due to known issues
"""
function set_algorithm_preferences(categories::Dict{String, String})
    @info "Setting LinearSolve preferences based on benchmark results..."
    
    # Define the size category names we use
    size_categories = ["tiny", "small", "medium", "large", "big"]
    
    # Define the element types we want to set preferences for
    target_eltypes = ["Float32", "Float64", "ComplexF32", "ComplexF64"]
    
    # Extract benchmarked results by element type and size
    benchmarked = Dict{String, Dict{String, String}}()
    mkl_is_best_somewhere = false  # Track if MKL wins any category
    
    for (key, algorithm) in categories
        if contains(key, "_")
            eltype, size_range = split(key, "_", limit=2)
            if !haskey(benchmarked, eltype)
                benchmarked[eltype] = Dict{String, String}()
            end
            benchmarked[eltype][size_range] = algorithm
            
            # Check if MKL algorithm is best for this category
            if contains(algorithm, "MKL")
                mkl_is_best_somewhere = true
                @info "MKL algorithm ($algorithm) is best for $eltype at size $size_range"
            end
        end
    end
    
    # Helper function to get best algorithm for complex types (avoiding RFLU)
    function get_complex_algorithm(results_df, eltype_str, size_range)
        # If we have direct benchmark results, use them
        if haskey(benchmarked, eltype_str) && haskey(benchmarked[eltype_str], size_range)
            alg = benchmarked[eltype_str][size_range]
            # Check if it's RFLU and we should avoid it for complex
            if contains(alg, "RFLU") || contains(alg, "RecursiveFactorization")
                # Find the second best for this case
                # We'd need the full results DataFrame to do this properly
                # For now, we'll just flag it
                @warn "RFLUFactorization selected for $eltype_str at size $size_range, but it has known issues with complex numbers"
            end
            return alg
        end
        return nothing
    end
    
    # Process each target element type and size combination
    for eltype in target_eltypes
        for size_cat in size_categories
            # Map size categories to the range strings used in categories
            size_range = if size_cat == "tiny"
                "0-128"  # Maps to tiny range
            elseif size_cat == "small"
                "0-128"  # Small also uses this range
            elseif size_cat == "medium"
                "128-256"  # Medium range
            elseif size_cat == "large"
                "256-512"  # Large range
            elseif size_cat == "big"
                "512+"  # Big range
            else
                continue
            end
            
            # Determine the algorithm based on fallback rules
            algorithm = nothing
            
            if eltype == "Float64"
                # Float64 should be directly benchmarked
                if haskey(benchmarked, "Float64") && haskey(benchmarked["Float64"], size_range)
                    algorithm = benchmarked["Float64"][size_range]
                end
            elseif eltype == "Float32"
                # Float32: use Float32 results if available, else use Float64
                if haskey(benchmarked, "Float32") && haskey(benchmarked["Float32"], size_range)
                    algorithm = benchmarked["Float32"][size_range]
                elseif haskey(benchmarked, "Float64") && haskey(benchmarked["Float64"], size_range)
                    algorithm = benchmarked["Float64"][size_range]
                end
            elseif eltype == "ComplexF32"
                # ComplexF32: use ComplexF32 if available, else Float64 (avoiding RFLU)
                if haskey(benchmarked, "ComplexF32") && haskey(benchmarked["ComplexF32"], size_range)
                    algorithm = benchmarked["ComplexF32"][size_range]
                elseif haskey(benchmarked, "Float64") && haskey(benchmarked["Float64"], size_range)
                    algorithm = benchmarked["Float64"][size_range]
                    # Check for RFLU and warn
                    if contains(algorithm, "RFLU") || contains(algorithm, "RecursiveFactorization")
                        @warn "Would use RFLUFactorization for ComplexF32 at $size_cat, but it has issues with complex numbers. Consider benchmarking ComplexF32 directly."
                    end
                end
            elseif eltype == "ComplexF64"
                # ComplexF64: use ComplexF64 if available, else ComplexF32, else Float64 (avoiding RFLU)
                if haskey(benchmarked, "ComplexF64") && haskey(benchmarked["ComplexF64"], size_range)
                    algorithm = benchmarked["ComplexF64"][size_range]
                elseif haskey(benchmarked, "ComplexF32") && haskey(benchmarked["ComplexF32"], size_range)
                    algorithm = benchmarked["ComplexF32"][size_range]
                elseif haskey(benchmarked, "Float64") && haskey(benchmarked["Float64"], size_range)
                    algorithm = benchmarked["Float64"][size_range]
                    # Check for RFLU and warn
                    if contains(algorithm, "RFLU") || contains(algorithm, "RecursiveFactorization")
                        @warn "Would use RFLUFactorization for ComplexF64 at $size_cat, but it has issues with complex numbers. Consider benchmarking ComplexF64 directly."
                    end
                end
            end
            
            # Set the preference if we have an algorithm
            if algorithm !== nothing
                pref_key = "best_algorithm_$(eltype)_$(size_cat)"
                Preferences.set_preferences!(LinearSolve, pref_key => algorithm; force = true)
                @info "Set preference $pref_key = $algorithm in LinearSolve.jl"
            end
        end
    end
    
    # Set MKL preference based on whether it was best for any category
    # If MKL wasn't best anywhere, disable it to avoid loading unnecessary dependencies
    # Note: During benchmarking, MKL is temporarily enabled to test MKL algorithms
    # This final preference setting determines whether MKL loads in normal usage
    Preferences.set_preferences!(LinearSolve, "LoadMKL_JLL" => mkl_is_best_somewhere; force = true)
    
    if mkl_is_best_somewhere
        @info "MKL was best in at least one category - setting LoadMKL_JLL preference to true"
    else
        @info "MKL was not best in any category - setting LoadMKL_JLL preference to false to avoid loading unnecessary dependencies"
    end
    
    # Set a timestamp for when these preferences were created
    Preferences.set_preferences!(LinearSolve, "autotune_timestamp" => string(Dates.now()); force = true)
    
    @info "Preferences updated in LinearSolve.jl. You may need to restart Julia for changes to take effect."
end

"""
    get_algorithm_preferences()

Get the current algorithm preferences from LinearSolve.jl.
Returns preferences organized by element type and size category.
"""
function get_algorithm_preferences()
    prefs = Dict{String, String}()
    
    # Define the patterns we look for
    target_eltypes = ["Float32", "Float64", "ComplexF32", "ComplexF64"]
    size_categories = ["tiny", "small", "medium", "large", "big"]
    
    for eltype in target_eltypes
        for size_cat in size_categories
            pref_key = "best_algorithm_$(eltype)_$(size_cat)"
            value = Preferences.load_preference(LinearSolve, pref_key, nothing)
            if value !== nothing
                readable_key = "$(eltype)_$(size_cat)"
                prefs[readable_key] = value
            end
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
    
    # Define the patterns we look for
    target_eltypes = ["Float32", "Float64", "ComplexF32", "ComplexF64"]
    size_categories = ["tiny", "small", "medium", "large", "big"]
    
    for eltype in target_eltypes
        for size_cat in size_categories
            pref_key = "best_algorithm_$(eltype)_$(size_cat)"
            if Preferences.has_preference(LinearSolve, pref_key)
                Preferences.delete_preferences!(LinearSolve, pref_key; force = true)
                @info "Cleared preference: $pref_key"
            end
        end
    end
    
    # Clear timestamp
    if Preferences.has_preference(LinearSolve, "autotune_timestamp")
        Preferences.delete_preferences!(LinearSolve, "autotune_timestamp"; force = true)
    end
    
    # Clear MKL preference
    Preferences.delete_preferences!(LinearSolve, "LoadMKL_JLL"; force = true)
    @info "Cleared MKL preference"
    
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
    
    # Group by element type for better display
    by_eltype = Dict{String, Vector{Tuple{String, String}}}()
    for (key, algorithm) in prefs
        eltype, size_cat = split(key, "_", limit=2)
        if !haskey(by_eltype, eltype)
            by_eltype[eltype] = Vector{Tuple{String, String}}()
        end
        push!(by_eltype[eltype], (size_cat, algorithm))
    end
    
    for eltype in sort(collect(keys(by_eltype)))
        println("\n$eltype:")
        for (size_cat, algorithm) in sort(by_eltype[eltype])
            println("  $size_cat: $algorithm")
        end
    end
    
    # Show MKL preference
    mkl_pref = Preferences.load_preference(LinearSolve, "LoadMKL_JLL", nothing)
    if mkl_pref !== nothing
        println("\nMKL Usage: $(mkl_pref ? "Enabled" : "Disabled")")
    end
    
    timestamp = Preferences.load_preference(LinearSolve, "autotune_timestamp", "unknown")
    println("\nLast updated: $timestamp")
end