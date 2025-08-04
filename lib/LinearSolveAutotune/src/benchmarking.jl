# Core benchmarking functionality

"""
    test_algorithm_compatibility(alg, eltype::Type, test_size::Int=4)

Test if an algorithm is compatible with a given element type.
Returns true if compatible, false otherwise.
Uses more strict rules for BLAS-dependent algorithms with non-standard types.
"""
function test_algorithm_compatibility(alg, eltype::Type, test_size::Int = 4)
    # Get algorithm name for type-specific compatibility rules
    alg_name = string(typeof(alg).name.name)
    
    # Define strict compatibility rules for BLAS-dependent algorithms
    if eltype in [BigFloat, BigInt, Rational{Int}, Complex{BigFloat}]
        # For arbitrary precision types, only allow pure Julia algorithms
        if alg_name in ["LUFactorization", "MKLLUFactorization", "AppleAccelerateLUFactorization"]
            return false  # These rely on BLAS and shouldn't work with arbitrary precision
        end
        # SimpleLUFactorization, GenericLUFactorization, and RFLUFactorization should work (pure Julia)
    elseif eltype in [Float16, Complex{Float16}]
        # Float16 might not be supported by all BLAS
        if alg_name in ["MKLLUFactorization", "AppleAccelerateLUFactorization"]
            return false  # These might not support Float16
        end
    end
    
    # For standard types or algorithms that passed the strict check, test functionality
    try
        # Create a small test problem with the specified element type
        rng = MersenneTwister(123)
        A = rand(rng, eltype, test_size, test_size)
        b = rand(rng, eltype, test_size)
        u0 = rand(rng, eltype, test_size)
        
        prob = LinearProblem(A, b; u0 = u0)
        
        # Try to solve - if it works, the algorithm is compatible
        sol = solve(prob, alg)
        
        # Additional check: verify the solution is actually of the expected type
        if !isa(sol.u, AbstractVector{eltype})
            @debug "Algorithm $alg_name returned wrong element type for $eltype"
            return false
        end
        
        return true
        
    catch e
        # Algorithm failed - not compatible with this element type
        @debug "Algorithm $alg_name failed for $eltype: $e"
        return false
    end
end

"""
    filter_compatible_algorithms(algorithms, alg_names, eltype::Type)

Filter algorithms to only those compatible with the given element type.
Returns filtered algorithms and names.
"""
function filter_compatible_algorithms(algorithms, alg_names, eltype::Type)
    compatible_algs = []
    compatible_names = String[]
    
    @info "Testing algorithm compatibility with $(eltype)..."
    
    for (alg, name) in zip(algorithms, alg_names)
        if test_algorithm_compatibility(alg, eltype)
            push!(compatible_algs, alg)
            push!(compatible_names, name)
            @debug "✓ $name compatible with $eltype"
        else
            @debug "✗ $name not compatible with $eltype"
        end
    end
    
    @info "Found $(length(compatible_algs))/$(length(algorithms)) algorithms compatible with $eltype"
    
    return compatible_algs, compatible_names
end

"""
    benchmark_algorithms(sizes, algorithms, alg_names, eltypes; 
                        samples=5, seconds=0.5, large_matrices=false)

Benchmark the given algorithms across different matrix sizes and element types.
Returns a DataFrame with results including element type information.
"""
function benchmark_algorithms(sizes, algorithms, alg_names, eltypes;
        samples = 5, seconds = 0.5, large_matrices = false)

    # Set benchmark parameters
    old_params = BenchmarkTools.DEFAULT_PARAMETERS
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = seconds
    BenchmarkTools.DEFAULT_PARAMETERS.samples = samples

    # Initialize results DataFrame
    results_data = []

    try
        for eltype in eltypes
            @info "Benchmarking with element type: $eltype"
            
            # Filter algorithms for this element type
            compatible_algs, compatible_names = filter_compatible_algorithms(algorithms, alg_names, eltype)
            
            if isempty(compatible_algs)
                @warn "No algorithms compatible with $eltype, skipping..."
                continue
            end
            
            for n in sizes
                @info "Benchmarking $n × $n matrices with $eltype..."

                # Create test problem with specified element type
                rng = MersenneTwister(123)  # Consistent seed for reproducibility
                A = rand(rng, eltype, n, n)
                b = rand(rng, eltype, n)
                u0 = rand(rng, eltype, n)

                for (alg, name) in zip(compatible_algs, compatible_names)
                    gflops = 0.0
                    success = true
                    error_msg = ""

                    try
                        # Create the linear problem for this test
                        prob = LinearProblem(copy(A), copy(b);
                            u0 = copy(u0),
                            alias = LinearAliasSpecifier(alias_A = true, alias_b = true))

                        # Warmup run
                        solve(prob, alg)

                        # Actual benchmark
                        bench = @benchmark solve($prob, $alg) setup=(prob = LinearProblem(
                            copy($A), copy($b);
                            u0 = copy($u0),
                            alias = LinearAliasSpecifier(alias_A = true, alias_b = true)))

                        # Calculate GFLOPs
                        min_time_sec = minimum(bench.times) / 1e9
                        flops = luflop(n, n)
                        gflops = flops / min_time_sec / 1e9

                    catch e
                        success = false
                        error_msg = string(e)
                        @warn "Algorithm $name failed for size $n with $eltype: $error_msg"
                    end

                    # Store result with element type information
                    push!(results_data,
                        (
                            size = n,
                            algorithm = name,
                            eltype = string(eltype),
                            gflops = gflops,
                            success = success,
                            error = error_msg
                        ))
                end
            end
        end

    finally
        # Restore original benchmark parameters
        BenchmarkTools.DEFAULT_PARAMETERS = old_params
    end

    return DataFrame(results_data)
end

"""
    get_benchmark_sizes(large_matrices::Bool=false)

Get the matrix sizes to benchmark based on the large_matrices flag.
"""
function get_benchmark_sizes(large_matrices::Bool = false)
    if large_matrices
        # For GPU benchmarking, include much larger sizes up to 10000
        return vcat(4:8:128, 150:50:500, 600:100:1000,
            1200:200:2000, 2500:500:5000, 6000:1000:10000)
    else
        # Default sizes similar to existing benchmarks
        return 4:8:500
    end
end

"""
    categorize_results(df::DataFrame)

Categorize the benchmark results into size ranges and find the best algorithm for each range and element type.
"""
function categorize_results(df::DataFrame)
    # Filter successful results
    successful_df = filter(row -> row.success, df)

    if nrow(successful_df) == 0
        @warn "No successful benchmark results found!"
        return Dict{String, String}()
    end

    categories = Dict{String, String}()

    # Define size ranges
    ranges = [
        ("0-128", 1:128),
        ("128-256", 129:256),
        ("256-512", 257:512),
        ("512+", 513:10000)
    ]

    # Get unique element types
    eltypes = unique(successful_df.eltype)

    for eltype in eltypes
        @info "Categorizing results for element type: $eltype"
        
        # Filter results for this element type
        eltype_df = filter(row -> row.eltype == eltype, successful_df)
        
        if nrow(eltype_df) == 0
            continue
        end

        for (range_name, range) in ranges
            # Get results for this size range and element type
            range_df = filter(row -> row.size in range, eltype_df)

            if nrow(range_df) == 0
                continue
            end

            # Calculate average GFLOPs for each algorithm in this range
            avg_results = combine(groupby(range_df, :algorithm), :gflops => mean => :avg_gflops)

            # Find the best algorithm
            if nrow(avg_results) > 0
                best_idx = argmax(avg_results.avg_gflops)
                best_alg = avg_results.algorithm[best_idx]
                category_key = "$(eltype)_$(range_name)"
                categories[category_key] = best_alg
                @info "Best algorithm for $eltype size range $range_name: $best_alg ($(round(avg_results.avg_gflops[best_idx], digits=2)) GFLOPs avg)"
            end
        end
    end

    return categories
end
