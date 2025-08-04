# Core benchmarking functionality

"""
    benchmark_algorithms(sizes, algorithms, alg_names; 
                        samples=5, seconds=0.5, large_matrices=false)

Benchmark the given algorithms across different matrix sizes.
Returns a DataFrame with results.
"""
function benchmark_algorithms(sizes, algorithms, alg_names;
        samples = 5, seconds = 0.5, large_matrices = false)

    # Set benchmark parameters
    old_params = BenchmarkTools.DEFAULT_PARAMETERS
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = seconds
    BenchmarkTools.DEFAULT_PARAMETERS.samples = samples

    # Initialize results DataFrame
    results_data = []

    try
        for n in sizes
            @info "Benchmarking $n × $n matrices..."

            # Create test problem
            rng = MersenneTwister(123)  # Consistent seed for reproducibility
            A = rand(rng, n, n)
            b = rand(rng, n)
            u0 = rand(rng, n)

            for (alg, name) in zip(algorithms, alg_names)
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
                    @warn "Algorithm $name failed for size $n: $error_msg"
                end

                # Store result
                push!(results_data,
                    (
                        size = n,
                        algorithm = name,
                        gflops = gflops,
                        success = success,
                        error = error_msg
                    ))
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
        # For GPU benchmarking, include much larger sizes
        return vcat(4:8:128, 150:50:500, 600:100:1000, 1200:200:2000)
    else
        # Default sizes similar to existing benchmarks
        return 4:8:500
    end
end

"""
    categorize_results(df::DataFrame)

Categorize the benchmark results into size ranges and find the best algorithm for each range.
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

    for (range_name, range) in ranges
        # Get results for this size range
        range_df = filter(row -> row.size in range, successful_df)

        if nrow(range_df) == 0
            continue
        end

        # Calculate average GFLOPs for each algorithm in this range
        avg_results = combine(groupby(range_df, :algorithm), :gflops => mean => :avg_gflops)

        # Find the best algorithm
        if nrow(avg_results) > 0
            best_idx = argmax(avg_results.avg_gflops)
            best_alg = avg_results.algorithm[best_idx]
            categories[range_name] = best_alg
            @info "Best algorithm for size range $range_name: $best_alg ($(round(avg_results.avg_gflops[best_idx], digits=2)) GFLOPs avg)"
        end
    end

    return categories
end
