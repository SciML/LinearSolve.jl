using Preferences

"""
    show_algorithm_choices()

Display what algorithm choices are actually made by the default solver for 
representative matrix sizes. Shows current preferences and system information.
"""
function show_algorithm_choices()
    println("="^60)
    println("LinearSolve.jl Algorithm Choice Analysis")
    println("="^60)
    
    # Show current preferences
    println("📋 Current Preferences:")
    println("-"^60)
    
    any_prefs_set = false
    for eltype in ["Float64"]  # Focus on Float64
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
    
    # Show algorithm choices for one representative size per category
    println("\n📊 Default Algorithm Choices (Float64):")
    println("-"^60)
    println("Size       Category    Chosen Algorithm")
    println("-"^60)
    
    # One representative size per category
    test_cases = [
        (8, "tiny"),      # ≤10 override
        (50, "small"),    # 21-100
        (200, "medium"),  # 101-300
        (500, "large"),   # 301-1000
        (1500, "big")     # >1000
    ]
    
    for (size, expected_category) in test_cases
        # Create test problem
        A = rand(Float64, size, size) + I(size)
        b = rand(Float64, size)
        
        # Get algorithm choice
        chosen_alg = defaultalg(A, b, OperatorAssumptions(true))
        
        size_str = lpad("$(size)×$(size)", 10)
        cat_str = rpad(expected_category, 11)
        alg_str = string(chosen_alg.alg)
        
        println("$(size_str) $(cat_str) $(alg_str)")
    end
    
    # Show different element types for medium size
    println("\n📊 Element Type Choices (200×200):")
    println("-"^60)
    println("Element Type    Chosen Algorithm")
    println("-"^60)
    
    test_eltypes = [Float32, Float64, ComplexF32, ComplexF64]
    test_size = 200  # Medium category
    
    for eltype in test_eltypes
        A = rand(eltype, test_size, test_size) + I(test_size)
        b = rand(eltype, test_size)
        
        chosen_alg = defaultalg(A, b, OperatorAssumptions(true))
        
        type_str = rpad(string(eltype), 15)
        alg_str = string(chosen_alg.alg)
        
        println("$(type_str) $(alg_str)")
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