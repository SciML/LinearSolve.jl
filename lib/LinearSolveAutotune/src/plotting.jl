# Plotting functionality for benchmark results

"""
    create_benchmark_plot(df::DataFrame; title="LinearSolve.jl LU Factorization Benchmark")

Create a plot showing GFLOPs vs matrix size for different algorithms.
"""
function create_benchmark_plot(df::DataFrame; title = "LinearSolve.jl LU Factorization Benchmark")
    # Filter successful results
    successful_df = filter(row -> row.success, df)

    if nrow(successful_df) == 0
        @warn "No successful results to plot!"
        return nothing
    end

    # Get unique algorithms and sizes
    algorithms = unique(successful_df.algorithm)
    sizes = sort(unique(successful_df.size))

    # Create the plot
    p = plot(title = title,
        xlabel = "Matrix Size (N×N)",
        ylabel = "Performance (GFLOPs)",
        legend = :outertopright,
        dpi = 300)

    # Plot each algorithm
    for alg in algorithms
        alg_df = filter(row -> row.algorithm == alg, successful_df)
        if nrow(alg_df) > 0
            # Sort by size for proper line plotting
            sort!(alg_df, :size)
            plot!(p, alg_df.size, alg_df.gflops,
                label = alg,
                marker = :circle,
                linewidth = 2,
                markersize = 4)
        end
    end

    return p
end

"""
    save_benchmark_plot(p, filename_base="autotune_benchmark")

Save the benchmark plot in both PNG and PDF formats.
"""
function save_benchmark_plot(p, filename_base = "autotune_benchmark")
    if p === nothing
        @warn "Cannot save plot: plot is nothing"
        return nothing
    end

    png_file = "$(filename_base).png"
    pdf_file = "$(filename_base).pdf"

    try
        savefig(p, png_file)
        savefig(p, pdf_file)
        @info "Plots saved as $png_file and $pdf_file"
        return (png_file, pdf_file)
    catch e
        @warn "Failed to save plots: $e"
        return nothing
    end
end
