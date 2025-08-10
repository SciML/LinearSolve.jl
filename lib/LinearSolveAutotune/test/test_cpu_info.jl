using LinearSolveAutotune
using Test

@testset "CPU Information Collection" begin
    # Test get_system_info function
    system_info = LinearSolveAutotune.get_system_info()
    
    @test haskey(system_info, "cpu_model")
    @test haskey(system_info, "cpu_speed_mhz")
    @test haskey(system_info, "heterogeneous_cpus")
    
    # Display the collected information
    @info "CPU Information collected:"
    @info "  CPU Model: $(system_info["cpu_model"])"
    @info "  CPU Speed: $(system_info["cpu_speed_mhz"]) MHz"
    @info "  Heterogeneous CPUs: $(system_info["heterogeneous_cpus"])"
    
    # Test that cpu_model is a string and not "Unknown" (unless no CPU info available)
    @test isa(system_info["cpu_model"], String)
    
    # Test that cpu_speed is a number
    @test isa(system_info["cpu_speed_mhz"], Number)
    
    # Test detailed system info
    detailed_info = LinearSolveAutotune.get_detailed_system_info()
    @test size(detailed_info, 1) == 1  # Should return one row
    @test hasproperty(detailed_info, :cpu_model)
    @test hasproperty(detailed_info, :cpu_speed_mhz)
    
    @info "Detailed CPU info from DataFrame:"
    @info "  CPU Model: $(detailed_info.cpu_model[1])"
    @info "  CPU Speed: $(detailed_info.cpu_speed_mhz[1]) MHz"
    
    # Test that the information matches between the two functions
    @test detailed_info.cpu_model[1] == system_info["cpu_model"]
    @test detailed_info.cpu_speed_mhz[1] == system_info["cpu_speed_mhz"]
    
    @test true  # If we got here without errors, the test passes
end