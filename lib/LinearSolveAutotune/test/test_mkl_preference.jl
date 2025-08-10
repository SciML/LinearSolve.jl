using LinearSolveAutotune
using LinearSolve
using Test

@testset "MKL Preference Management" begin
    # Test that MKL preference is set before loading LinearSolve
    # This has already happened in LinearSolveAutotune.jl module initialization
    
    # Create some mock categories to test preference setting
    categories_with_mkl = Dict{String, String}(
        "Float64_tiny (5-20)" => "MKLLUFactorization",
        "Float64_small (20-100)" => "RFLUFactorization",
        "Float64_medium (100-300)" => "MKLLUFactorization",
        "Float32_tiny (5-20)" => "LUFactorization"
    )
    
    categories_without_mkl = Dict{String, String}(
        "Float64_tiny (5-20)" => "RFLUFactorization",
        "Float64_small (20-100)" => "RFLUFactorization",
        "Float64_medium (100-300)" => "LUFactorization",
        "Float32_tiny (5-20)" => "SimpleLUFactorization"
    )
    
    # Test setting preferences with MKL as best
    @info "Testing preference setting with MKL as best algorithm..."
    LinearSolveAutotune.set_algorithm_preferences(categories_with_mkl)
    
    # The MKL preference should be set to true
    # Note: We can't directly test the preference value without restarting Julia
    # but we can verify the function runs without error
    
    @info "Testing preference setting without MKL as best algorithm..."
    LinearSolveAutotune.set_algorithm_preferences(categories_without_mkl)
    
    # Clear preferences
    @info "Testing preference clearing..."
    LinearSolveAutotune.clear_algorithm_preferences()
    
    # Show current preferences
    @info "Testing preference display..."
    LinearSolveAutotune.show_current_preferences()
    
    @test true  # If we got here without errors, the test passes
end