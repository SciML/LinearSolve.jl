using LinearSolve, LinearAlgebra, Preferences, Printf

"""
    show_algorithm_choices(; clear_preferences=true, set_test_preferences=false)

Display what algorithm choices are actually made by the default solver for various 
matrix sizes and element types. This function helps demonstrate the current 
algorithm selection behavior and can be used to verify preference system integration.

## Arguments
- `clear_preferences::Bool = true`: Clear existing preferences before testing
- `set_test_preferences::Bool = false`: Set test preferences to demonstrate preference behavior

## Output
Shows a table of matrix sizes, their categorization, and the chosen algorithm.
"""
function show_algorithm_choices(; clear_preferences=true, set_test_preferences=false)
    println("="^80)
    println("LinearSolve.jl Default Algorithm Choice Analysis")
    println("="^80)
    
    # Clear existing preferences if requested
    if clear_preferences
        target_eltypes = ["Float32", "Float64", "ComplexF32", "ComplexF64"]
        size_categories = ["tiny", "small", "medium", "large", "big"]
        
        for eltype in target_eltypes
            for size_cat in size_categories
                for pref_type in ["best_algorithm", "best_always_loaded"]
                    pref_key = "$(pref_type)_$(eltype)_$(size_cat)"
                    if Preferences.has_preference(LinearSolve, pref_key)
                        Preferences.delete_preferences!(LinearSolve, pref_key; force = true)
                    end
                end
            end
        end
        println("✅ Cleared all existing preferences")
    end
    
    # Set test preferences to demonstrate preference behavior
    if set_test_preferences
        println("📝 Setting test preferences to demonstrate preference system...")
        
        # Set different algorithms for each size category
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_tiny" => "GenericLUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_tiny" => "GenericLUFactorization"; force = true)
        
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_small" => "RFLUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_small" => "LUFactorization"; force = true)
        
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_medium" => "AppleAccelerateLUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_medium" => "MKLLUFactorization"; force = true)
        
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_large" => "MKLLUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_large" => "LUFactorization"; force = true)
        
        Preferences.set_preferences!(LinearSolve, "best_algorithm_Float64_big" => "LUFactorization"; force = true)
        Preferences.set_preferences!(LinearSolve, "best_always_loaded_Float64_big" => "GenericLUFactorization"; force = true)
        
        println("   Set different preferences for each size category")
    end
    
    # Test sizes spanning all categories including critical boundaries
    test_cases = [
        # Size, Description
        (5, "Tiny (should always override to GenericLU)"),
        (8, "Tiny (should always override to GenericLU)"),
        (10, "Tiny boundary (should always override to GenericLU)"),
        (15, "Tiny category (≤20)"),
        (20, "Tiny boundary (=20)"),
        (21, "Small category start (=21)"),
        (50, "Small category middle"),
        (80, "Small category"),
        (100, "Small boundary (=100)"),
        (101, "Medium category start (=101)"),
        (150, "Medium category"),
        (200, "Medium category"),
        (300, "Medium boundary (=300)"),
        (301, "Large category start (=301)"),
        (500, "Large category"),
        (1000, "Large boundary (=1000)"),
        (1001, "Big category start (=1001)"),
        (2000, "Big category")
    ]
    
    println("\n📊 Algorithm Choice Analysis for Float64 matrices:")
    println("-"^80)
    println("Size      Description                         Expected Category         Chosen Algorithm")
    println("-"^80)
    
    for (size, description) in test_cases
        # Determine expected category based on LinearSolveAutotune boundaries
        expected_category = if size <= 20
            "tiny"
        elseif size <= 100
            "small"
        elseif size <= 300
            "medium"
        elseif size <= 1000
            "large"
        else
            "big"
        end
        
        # Create test problem
        A = rand(Float64, size, size) + I(size)
        b = rand(Float64, size)
        
        # Get algorithm choice
        chosen_alg = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
        
        # Format output with padding
        size_str = lpad("$(size)×$(size)", 8)
        desc_str = rpad(description, 35)
        cat_str = rpad(expected_category, 25)
        alg_str = string(chosen_alg.alg)
        
        println("$(size_str) $(desc_str) $(cat_str) $(alg_str)")
    end
    
    # Test different element types
    println("\n📊 Algorithm Choice Analysis for Different Element Types:")
    println("-"^80)
    println("Size            Element Type    Expected Category         Chosen Algorithm")
    println("-"^80)
    
    test_eltypes = [Float32, Float64, ComplexF32, ComplexF64]
    test_size = 200  # Medium category
    
    for eltype in test_eltypes
        A = rand(eltype, test_size, test_size) + I(test_size)
        b = rand(eltype, test_size)
        
        chosen_alg = LinearSolve.defaultalg(A, b, LinearSolve.OperatorAssumptions(true))
        
        size_str = lpad("$(test_size)×$(test_size)", 15)
        type_str = rpad(string(eltype), 15)
        cat_str = rpad("medium", 25)
        alg_str = string(chosen_alg.alg)
        
        println("$(size_str) $(type_str) $(cat_str) $(alg_str)")
    end
    
    # Show current preferences if any are set
    println("\n📋 Current Preferences:")
    println("-"^80)
    
    any_prefs_set = false
    for eltype in ["Float64"]  # Just show Float64 for brevity
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
    
    # Show system information
    println("\n🖥️  System Information:")
    println("-"^80)
    println("MKL available: ", LinearSolve.usemkl)
    println("Apple Accelerate available: ", LinearSolve.appleaccelerate_isavailable())
    println("RecursiveFactorization enabled: ", LinearSolve.userecursivefactorization(nothing))
    
    println("\n💡 Notes:")
    println("- Matrices ≤10 elements always use GenericLUFactorization (tiny override)")
    println("- Size categories: tiny(≤20), small(21-100), medium(101-300), large(301-1000), big(>1000)")
    println("- When preferences are set, the default solver should use the preferred algorithm")
    println("- Current choices show heuristic-based selection when no preferences are active")
    
    if set_test_preferences
        println("\n🧹 Cleaning up test preferences...")
        # Clear test preferences
        for eltype in target_eltypes
            for size_cat in size_categories
                for pref_type in ["best_algorithm", "best_always_loaded"]
                    pref_key = "$(pref_type)_$(eltype)_$(size_cat)"
                    if Preferences.has_preference(LinearSolve, pref_key)
                        Preferences.delete_preferences!(LinearSolve, pref_key; force = true)
                    end
                end
            end
        end
        println("✅ Test preferences cleared")
    end
    
    println("="^80)
end


# Run the analysis
println("🚀 Running Default Algorithm Choice Analysis...")
show_algorithm_choices()

println("\n\n🔬 Now testing WITH preferences set...")
show_algorithm_choices(clear_preferences=false, set_test_preferences=true)