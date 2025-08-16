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