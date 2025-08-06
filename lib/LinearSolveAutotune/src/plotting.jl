# Plotting functionality for benchmark results

"""
    create_benchmark_plots(df::DataFrame; title_base="LinearSolve.jl LU Factorization Benchmark")

Create separate plots for each element type showing GFLOPs vs matrix size for different algorithms.
Returns a dictionary of plots keyed by element type.
"""
function create_benchmark_plots(df::DataFrame; title_base = "LinearSolve.jl LU Factorization Benchmark")
    # Filter successful results
    successful_df = filter(row -> row.success, df)

    if nrow(successful_df) == 0
        @warn "No successful results to plot!"
        return Dict{String, Any}()
    end

    plots_dict = Dict{String, Any}()
    
    # Get unique element types
    eltypes = unique(successful_df.eltype)
    
    for eltype in eltypes
        @info "Creating plot for element type: $eltype"
        
        # Filter results for this element type
        eltype_df = filter(row -> row.eltype == eltype, successful_df)
        
        if nrow(eltype_df) == 0
            continue
        end

        # Get unique algorithms and sizes for this element type
        algorithms = unique(eltype_df.algorithm)
        sizes = sort(unique(eltype_df.size))

        # Create the plot for this element type
        title = "$title_base ($eltype)"
        p = plot(title = title,
            xlabel = "Matrix Size (N×N)",
            ylabel = "Performance (GFLOPs)",
            legend = :topleft,
            dpi = 300)

        # Plot each algorithm for this element type
        for alg in algorithms
            alg_df = filter(row -> row.algorithm == alg, eltype_df)
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
        
        plots_dict[eltype] = p
    end

    return plots_dict
end

"""
    create_benchmark_plot(df::DataFrame; title="LinearSolve.jl LU Factorization Benchmark")

Create a single plot showing GFLOPs vs matrix size for different algorithms.
Maintains backward compatibility - uses first element type if multiple exist.
"""
function create_benchmark_plot(df::DataFrame; title = "LinearSolve.jl LU Factorization Benchmark")
    # For backward compatibility, create plots for all element types and return the first one
    plots_dict = create_benchmark_plots(df; title_base = title)
    
    if isempty(plots_dict)
        return nothing
    end
    
    # Return the first plot for backward compatibility
    return first(values(plots_dict))
end

"""
    save_benchmark_plots(plots_dict::Dict, filename_base="autotune_benchmark")

Save multiple benchmark plots (one per element type) in both PNG and PDF formats.
Returns a dictionary of saved filenames keyed by element type.
"""
function save_benchmark_plots(plots_dict::Dict, filename_base = "autotune_benchmark")
    if isempty(plots_dict)
        @warn "Cannot save plots: plots dictionary is empty"
        return Dict{String, Tuple{String, String}}()
    end

    saved_files = Dict{String, Tuple{String, String}}()
    
    for (eltype, plot_obj) in plots_dict
        if plot_obj === nothing
            @warn "Cannot save plot for $eltype: plot is nothing"
            continue
        end

        # Create filenames with element type suffix
        eltype_safe = replace(string(eltype), "{" => "", "}" => "", "," => "_")
        png_file = "$(filename_base)_$(eltype_safe).png"
        pdf_file = "$(filename_base)_$(eltype_safe).pdf"

        try
            savefig(plot_obj, png_file)
            savefig(plot_obj, pdf_file)
            @info "Plots for $eltype saved as $png_file and $pdf_file"
            saved_files[eltype] = (png_file, pdf_file)
        catch e
            @warn "Failed to save plots for $eltype: $e"
        end
    end
    
    return saved_files
end

"""
    save_benchmark_plot(p, filename_base="autotune_benchmark")

Save a single benchmark plot in both PNG and PDF formats.
Maintains backward compatibility.
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
