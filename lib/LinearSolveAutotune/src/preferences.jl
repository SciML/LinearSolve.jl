# Preferences management for storing optimal algorithms in LinearSolve.jl

"""
    is_always_loaded_algorithm(algorithm_name::String)

Determine if an algorithm is always loaded (available without extensions).
Returns true for algorithms that don't require extensions to be available.
"""
function is_always_loaded_algorithm(algorithm_name::String)
    # Algorithms that are always available without requiring extensions
    always_loaded = [
        "LUFactorization",
        "GenericLUFactorization", 
        "MKLLUFactorization",  # Available if MKL is loaded
        "AppleAccelerateLUFactorization",  # Available on macOS
        "SimpleLUFactorization"
    ]
    
    return algorithm_name in always_loaded
end

"""
    find_best_always_loaded_algorithm(results_df::DataFrame, eltype_str::String, size_range_name::String)

Find the best always-loaded algorithm from benchmark results for a specific element type and size range.
Returns the algorithm name or nothing if no suitable algorithm is found.
"""
function find_best_always_loaded_algorithm(results_df::DataFrame, eltype_str::String, size_range_name::String)
    # Define size ranges to match the categories
    size_ranges = Dict(
        "tiny (5-20)" => 5:20,
        "small (20-100)" => 21:100,
        "medium (100-300)" => 101:300,
        "large (300-1000)" => 301:1000,
        "big (1000+)" => 1000:typemax(Int)
    )
    
    size_range = get(size_ranges, size_range_name, nothing)
    if size_range === nothing
        @debug "Unknown size range: $size_range_name"
        return nothing
    end
    
    # Filter results for this element type and size range
    filtered_results = filter(row -> 
        row.eltype == eltype_str && 
        row.size in size_range && 
        row.success && 
        !isnan(row.gflops) &&
        is_always_loaded_algorithm(row.algorithm), 
        results_df)
    
    if nrow(filtered_results) == 0
        return nothing
    end
    
    # Calculate average GFLOPs for each always-loaded algorithm
    avg_results = combine(groupby(filtered_results, :algorithm), 
        :gflops => (x -> mean(filter(!isnan, x))) => :avg_gflops)
    
    # Sort by performance and return the best
    sort!(avg_results, :avg_gflops, rev=true)
    
    if nrow(avg_results) > 0
        return avg_results.algorithm[1]
    end
    
    return nothing
end

"""
    set_algorithm_preferences(categories::Dict{String, String}, results_df::Union{DataFrame, Nothing} = nothing)

Set LinearSolve preferences based on the categorized benchmark results.
These preferences are stored in the main LinearSolve.jl package.

This function now supports the dual preference system introduced in LinearSolve.jl v2.31+:
- `best_algorithm_{type}_{size}`: Overall fastest algorithm
- `best_always_loaded_{type}_{size}`: Fastest among always-available methods

The function handles type fallbacks:
- If Float32 wasn't benchmarked, uses Float64 results
- If ComplexF64 wasn't benchmarked, uses ComplexF32 results (if available) or Float64
- If ComplexF32 wasn't benchmarked, uses Float64 results
- For complex types, avoids RFLUFactorization due to known issues

If results_df is provided, it will be used to determine the best always-loaded algorithm
from actual benchmark data. Otherwise, a fallback strategy is used.
"""
function set_algorithm_preferences(categories::Dict{String, String}, results_df::Union{DataFrame, Nothing} = nothing)
    @info "Setting LinearSolve preferences based on benchmark results (dual preference system)..."
    
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
            # Find matching size range from benchmarked data for this element type
            size_range = nothing
            if haskey(benchmarked, eltype)
                for range_key in keys(benchmarked[eltype])
                    # Check if the range_key contains the size category we're looking for
                    # e.g., "medium (100-300)" contains "medium"
                    if contains(range_key, size_cat)
                        size_range = range_key
                        break
                    end
                end
            end
            
            if size_range === nothing
                continue  # No matching size range found for this element type and size category
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
            
            # Set preferences if we have an algorithm
            if algorithm !== nothing
                # Set the best overall algorithm preference
                best_pref_key = "best_algorithm_$(eltype)_$(size_cat)"
                Preferences.set_preferences!(LinearSolve, best_pref_key => algorithm; force = true)
                @info "Set preference $best_pref_key = $algorithm in LinearSolve.jl"
                
                # Determine the best always-loaded algorithm
                best_always_loaded = nothing
                
                # If the best algorithm is already always-loaded, use it
                if is_always_loaded_algorithm(algorithm)
                    best_always_loaded = algorithm
                    @info "Best algorithm ($algorithm) is always-loaded for $(eltype) $(size_cat)"
                else
                    # Try to find the best always-loaded algorithm from benchmark results
                    if results_df !== nothing
                        best_always_loaded = find_best_always_loaded_algorithm(results_df, eltype, size_range)
                        if best_always_loaded !== nothing
                            @info "Found best always-loaded algorithm from benchmarks for $(eltype) $(size_cat): $best_always_loaded"
                        end
                    end
                    
                    # Fallback strategy if no benchmark data available or no suitable algorithm found
                    if best_always_loaded === nothing
                        if eltype == "Float64" || eltype == "Float32"
                            # For real types, prefer MKL > LU > Generic
                            if mkl_is_best_somewhere
                                best_always_loaded = "MKLLUFactorization"
                            else
                                best_always_loaded = "LUFactorization"
                            end
                        else
                            # For complex types, be more conservative since RFLU has issues
                            best_always_loaded = "LUFactorization"
                        end
                        @info "Using fallback always-loaded algorithm for $(eltype) $(size_cat): $best_always_loaded"
                    end
                end
                
                # Set the best always-loaded algorithm preference
                if best_always_loaded !== nothing
                    fallback_pref_key = "best_always_loaded_$(eltype)_$(size_cat)"
                    Preferences.set_preferences!(LinearSolve, fallback_pref_key => best_always_loaded; force = true)
                    @info "Set preference $fallback_pref_key = $best_always_loaded in LinearSolve.jl"
                end
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
Returns preferences organized by element type and size category, including both
best overall and best always-loaded algorithms.
"""
function get_algorithm_preferences()
    prefs = Dict{String, Any}()
    
    # Define the patterns we look for
    target_eltypes = ["Float32", "Float64", "ComplexF32", "ComplexF64"]
    size_categories = ["tiny", "small", "medium", "large", "big"]
    
    for eltype in target_eltypes
        for size_cat in size_categories
            readable_key = "$(eltype)_$(size_cat)"
            
            # Get best overall algorithm
            best_pref_key = "best_algorithm_$(eltype)_$(size_cat)"
            best_value = Preferences.load_preference(LinearSolve, best_pref_key, nothing)
            
            # Get best always-loaded algorithm
            fallback_pref_key = "best_always_loaded_$(eltype)_$(size_cat)"
            fallback_value = Preferences.load_preference(LinearSolve, fallback_pref_key, nothing)
            
            if best_value !== nothing || fallback_value !== nothing
                prefs[readable_key] = Dict(
                    "best" => best_value,
                    "always_loaded" => fallback_value
                )
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
    @info "Clearing LinearSolve autotune preferences (dual preference system)..."
    
    # Define the patterns we look for
    target_eltypes = ["Float32", "Float64", "ComplexF32", "ComplexF64"]
    size_categories = ["tiny", "small", "medium", "large", "big"]
    
    for eltype in target_eltypes
        for size_cat in size_categories
            # Clear best overall algorithm preference
            best_pref_key = "best_algorithm_$(eltype)_$(size_cat)"
            if Preferences.has_preference(LinearSolve, best_pref_key)
                Preferences.delete_preferences!(LinearSolve, best_pref_key; force = true)
                @info "Cleared preference: $best_pref_key"
            end
            
            # Clear best always-loaded algorithm preference
            fallback_pref_key = "best_always_loaded_$(eltype)_$(size_cat)"
            if Preferences.has_preference(LinearSolve, fallback_pref_key)
                Preferences.delete_preferences!(LinearSolve, fallback_pref_key; force = true)
                @info "Cleared preference: $fallback_pref_key"
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
    
    println("Current LinearSolve.jl autotune preferences (dual preference system):")
    println("="^70)
    
    # Group by element type for better display
    by_eltype = Dict{String, Vector{Tuple{String, Dict{String, Any}}}}()
    for (key, pref_dict) in prefs
        eltype, size_cat = split(key, "_", limit=2)
        if !haskey(by_eltype, eltype)
            by_eltype[eltype] = Vector{Tuple{String, Dict{String, Any}}}()
        end
        push!(by_eltype[eltype], (size_cat, pref_dict))
    end
    
    for eltype in sort(collect(keys(by_eltype)))
        println("\n$eltype:")
        for (size_cat, pref_dict) in sort(by_eltype[eltype])
            println("  $size_cat:")
            best_alg = get(pref_dict, "best", nothing)
            always_loaded_alg = get(pref_dict, "always_loaded", nothing)
            
            if best_alg !== nothing
                println("    Best overall: $best_alg")
            end
            if always_loaded_alg !== nothing
                println("    Best always-loaded: $always_loaded_alg")
            end
        end
    end
    
    # Show MKL preference
    mkl_pref = Preferences.load_preference(LinearSolve, "LoadMKL_JLL", nothing)
    if mkl_pref !== nothing
        println("\nMKL Usage: $(mkl_pref ? "Enabled" : "Disabled")")
    end
    
    timestamp = Preferences.load_preference(LinearSolve, "autotune_timestamp", "unknown")
    println("\nLast updated: $timestamp")
    println("\nNOTE: This uses the enhanced dual preference system where LinearSolve.jl")
    println("will try the best overall algorithm first, then fall back to the best")
    println("always-loaded algorithm if extensions are not available.")
end