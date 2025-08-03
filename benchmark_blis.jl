#!/usr/bin/env julia

"""
LinearSolve.jl LU Factorization Benchmark with BLIS Integration

This benchmark compares the performance of various LU factorization algorithms
including the new BLIS implementation, with conditional support for platform-specific
optimizations like Apple Accelerate and Intel MKL.

Based on: https://docs.sciml.ai/SciMLBenchmarksOutput/stable/LinearSolve/LUFactorization/
"""

using LinearSolve
using LinearAlgebra
using BenchmarkTools
using Random
using Plots
using RecursiveFactorization
using Libdl
using Preferences

# Import the algorithm types we need
using LinearSolve: BLISLUFactorization, BLISFlameLUFactorization, LUFactorization, RFLUFactorization, 
                   MKLLUFactorization, AppleAccelerateLUFactorization, 
                   FastLUFactorization

# Detection logic following LinearSolve.jl patterns

# MKL detection function
function check_mkl_available()
    if !(Sys.ARCH === :x86_64 || Sys.ARCH === :i686)
        return false
    end
    
    # Default to loading MKL unless we detect EPYC
    should_load_mkl = !occursin("EPYC", Sys.cpu_info()[1].model)
    if should_load_mkl
        try
            return LinearSolve.usemkl
        catch
            return false
        end
    else
        return false
    end
end

# Apple Accelerate detection function  
function check_apple_accelerate_available()
    if !Sys.isapple()
        return false
    end
    
    libacc = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
    libacc_hdl = Libdl.dlopen_e(libacc)
    if libacc_hdl == C_NULL
        return false
    end
    if Libdl.dlsym_e(libacc_hdl, "dgetrf_") == C_NULL
        return false
    end
    return LinearSolve.appleaccelerate_isavailable()
end

# BLIS detection (follows same structure as MKL/Apple Accelerate)
# Unlike MKL/Apple Accelerate, BLIS has no platform restrictions
const blis_available = let
    try
        using blis_jll, LAPACK_jll
        # Check if BLIS extension can be loaded
        Base.get_extension(LinearSolve, :LinearSolveBLISExt) !== nothing
    catch
        false
    end
end

# BLISFlame detection - requires blis_jll, libflame_jll, and LAPACK_jll
const blis_flame_available = let
    try
        using blis_jll, libflame_jll, LAPACK_jll
        # Check if BLISFlame extension can be loaded
        Base.get_extension(LinearSolve, :LinearSolveBLISFlameExt) !== nothing
    catch
        false
    end
end

# FastLapackInterface detection (follows same structure)
const fastlapack_available = let
    try
        using FastLapackInterface
        true
    catch
        false
    end
end

# Initialize platform-specific detections
const usemkl = check_mkl_available()
const apple_accelerate_available = check_apple_accelerate_available()

# Log detection results
if blis_available
    @info "BLIS dependencies loaded and extension available"
else
    @info "BLIS dependencies not available"
end

if blis_flame_available
    @info "BLISFlame dependencies loaded and extension available"
else
    @info "BLISFlame dependencies not available"
end

if fastlapack_available
    @info "FastLapackInterface loaded"
else
    @info "FastLapackInterface not available"
end

# Set the number of BLAS threads for consistent benchmarking
LinearAlgebra.BLAS.set_num_threads(1)

# Seed for reproducibility
Random.seed!(1234)

# Performance calculation for LU factorization
function luflop(m, n)
    # Computational complexity for LU factorization with partial pivoting
    # This is an approximation of the floating-point operations
    if m == n
        return (2 * n^3) / 3 + n^2 / 2 + n / 6
    else
        # For non-square matrices
        return m * n^2 - n^3 / 3
    end
end

# Function to check if a package extension is available
function has_extension(pkg, ext_name)
    return Base.get_extension(pkg, ext_name) !== nothing
end

# Function to check if MKL is available
function has_mkl()
    return usemkl
end

# Function to check if Apple Accelerate is available
function has_apple_accelerate()
    return apple_accelerate_available && LinearSolve.appleaccelerate_isavailable()
end

# Function to check if BLIS extension is available
function has_blis()
    return blis_available && has_extension(LinearSolve, :LinearSolveBLISExt)
end

# Function to check if BLISFlame extension is available
function has_blis_flame()
    return blis_flame_available && has_extension(LinearSolve, :LinearSolveBLISFlameExt)
end

# Function to check if FastLapackInterface is available
function has_fastlapack()
    return fastlapack_available
end

# Build algorithm list based on available implementations
function build_algorithm_list()
    algs = []
    alg_names = []
    
    # Standard algorithms always available
    push!(algs, LUFactorization())
    push!(alg_names, "LU (OpenBLAS)")
    
    push!(algs, RFLUFactorization())
    push!(alg_names, "RecursiveFactorization")
    
    # Fast LAPACK Interface
    if has_fastlapack()
        push!(algs, FastLUFactorization())
        push!(alg_names, "FastLU")
    else
        @info "FastLapackInterface not available, skipping FastLUFactorization"
    end
    
    # BLIS implementation
    if has_blis()
        push!(algs, BLISLUFactorization())
        push!(alg_names, "BLIS")
        @info "BLIS extension loaded successfully"
    else
        @warn "BLIS extension not available. Try: using blis_jll, LAPACK_jll"
    end
    
    # BLISFlame implementation (BLIS + libflame, falls back to BLIS + reference LAPACK)
    if has_blis_flame()
        push!(algs, BLISFlameLUFactorization())
        push!(alg_names, "BLISFlame")
        @info "BLISFlame extension loaded successfully"
    else
        @warn "BLISFlame extension not available. Try: using blis_jll, libflame_jll, LAPACK_jll"
    end
    
    # Intel MKL
    if has_mkl()
        push!(algs, MKLLUFactorization())
        push!(alg_names, "MKL")
        @info "Intel MKL support detected"
    else
        @info "Intel MKL not available"
    end
    
    # Apple Accelerate (macOS only)
    if has_apple_accelerate()
        push!(algs, AppleAccelerateLUFactorization())
        push!(alg_names, "Apple Accelerate")
        @info "Apple Accelerate support detected"
    elseif Sys.isapple()
        @info "Apple Accelerate not available on this macOS system"
    end
    
    return algs, alg_names
end

# Benchmark function
function benchmark_lu_factorizations(sizes = [4, 8, 16, 32, 64, 128, 256])
    algs, alg_names = build_algorithm_list()
    
    @info "Benchmarking $(length(algs)) algorithms: $(join(alg_names, ", "))"
    
    # Results storage
    results = Dict()
    for name in alg_names
        results[name] = Float64[]
    end
    
    for n in sizes
        @info "Benchmarking matrix size: $(n)×$(n)"
        
        # Generate test problem
        A = rand(n, n)
        b = rand(n)
        prob = LinearProblem(A, b)
        
        # Benchmark each algorithm
        for (alg, name) in zip(algs, alg_names)
            try
                # Warmup
                solve(prob, alg)
                
                # Actual benchmark
                bench = @benchmark solve($prob, $alg) samples=5 evals=1
                time_sec = minimum(bench.times) / 1e9  # Convert to seconds
                flops = luflop(n, n)
                gflops = flops / time_sec / 1e9
                
                push!(results[name], gflops)
                @info "$name: $(round(gflops, digits=2)) GFLOPs"
                
            catch e
                @warn "Failed to benchmark $name for size $n: $e"
                push!(results[name], 0.0)
            end
        end
        
        println()  # Add spacing between sizes
    end
    
    return sizes, results, alg_names
end

# Plotting function
function plot_benchmark_results(sizes, results, alg_names)
    p = plot(title="LU Factorization Performance Comparison",
             xlabel="Matrix Size (n×n)", 
             ylabel="Performance (GFLOPs)",
             legend=:topright,
             dpi=300)
    
    # Color palette for different algorithms
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink]
    
    for (i, name) in enumerate(alg_names)
        if haskey(results, name) && !isempty(results[name])
            # Filter out zero values (failed benchmarks)
            valid_indices = results[name] .> 0
            if any(valid_indices)
                plot!(p, sizes[valid_indices], results[name][valid_indices], 
                      label=name, 
                      marker=:circle, 
                      linewidth=2,
                      color=colors[mod1(i, length(colors))])
            end
        end
    end
    
    return p
end

# Main execution
function main()
    println("="^60)
    println("LinearSolve.jl LU Factorization Benchmark with BLIS")
    println("="^60)
    println()
    
    # System information
    println("System Information:")
    println("  Julia Version: $(VERSION)")
    println("  OS: $(Sys.KERNEL) $(Sys.ARCH)")
    println("  CPU Threads: $(Threads.nthreads())")
    println("  BLAS Threads: $(LinearAlgebra.BLAS.get_num_threads())")
    println("  BLAS Config: $(LinearAlgebra.BLAS.get_config())")
    println()
    
    # Check available implementations
    println("Available Implementations:")
    println("  BLIS: $(has_blis())")
    println("  MKL: $(has_mkl())")
    println("  Apple Accelerate: $(has_apple_accelerate())")
    println()
    
    # Run benchmarks
    sizes, results, alg_names = benchmark_lu_factorizations()
    
    # Create and save plot
    p = plot_benchmark_results(sizes, results, alg_names)
    savefig(p, "lu_factorization_benchmark.png")
    @info "Benchmark plot saved as 'lu_factorization_benchmark.png'"
    
    # Display results table
    println("\nResults Summary (GFLOPs):")
    println("-"^60)
    print("Size")
    for name in alg_names
        print("\t$(name)")
    end
    println()
    
    for (i, n) in enumerate(sizes)
        print("$n")
        for name in alg_names
            if haskey(results, name) && length(results[name]) >= i
                print("\t$(round(results[name][i], digits=2))")
            else
                print("\t--")
            end
        end
        println()
    end
    
    return p
end

# Execute if run as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end