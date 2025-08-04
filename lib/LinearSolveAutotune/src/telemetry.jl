# Telemetry functionality for sharing benchmark results

"""
    format_results_for_github(df::DataFrame, system_info::Dict, categories::Dict{String, String})

Format benchmark results as a markdown table suitable for GitHub issues.
"""
function format_results_for_github(df::DataFrame, system_info::Dict, categories::Dict{
        String, String})
    # Filter successful results
    successful_df = filter(row -> row.success, df)

    if nrow(successful_df) == 0
        return "No successful benchmark results to report."
    end

    markdown_content = """
## LinearSolve.jl Autotune Benchmark Results

### System Information
$(format_system_info_markdown(system_info))

### Performance Summary by Size Range
$(format_categories_markdown(categories))

### Detailed Results
$(format_detailed_results_markdown(successful_df))

---
*Generated automatically by LinearSolveAutotune.jl*
"""

    return markdown_content
end

"""
    format_system_info_markdown(system_info::Dict)

Format system information as markdown.
"""
function format_system_info_markdown(system_info::Dict)
    lines = String[]
    push!(lines, "- **Julia Version**: $(system_info["julia_version"])")
    push!(lines, "- **OS**: $(system_info["os"])")
    push!(lines, "- **Architecture**: $(system_info["arch"])")
    push!(lines, "- **CPU**: $(system_info["cpu_name"])")
    push!(lines, "- **Cores**: $(system_info["num_cores"])")
    push!(lines, "- **Threads**: $(system_info["num_threads"])")
    push!(lines, "- **BLAS**: $(system_info["blas_vendor"])")
    push!(lines, "- **MKL Available**: $(system_info["mkl_available"])")
    push!(lines, "- **Apple Accelerate Available**: $(system_info["apple_accelerate_available"])")
    push!(lines, "- **CUDA Available**: $(system_info["has_cuda"])")
    push!(lines, "- **Metal Available**: $(system_info["has_metal"])")

    return join(lines, "\n")
end

"""
    format_categories_markdown(categories::Dict{String, String})

Format the categorized results as markdown.
"""
function format_categories_markdown(categories::Dict{String, String})
    if isempty(categories)
        return "No category recommendations available."
    end

    lines = String[]
    push!(lines, "| Size Range | Best Algorithm |")
    push!(lines, "|------------|----------------|")

    for (range, algorithm) in sort(categories)
        push!(lines, "| $range | $algorithm |")
    end

    return join(lines, "\n")
end

"""
    format_detailed_results_markdown(df::DataFrame)

Format detailed benchmark results as a markdown table.
"""
function format_detailed_results_markdown(df::DataFrame)
    # Create a summary table with average performance per algorithm
    summary = combine(groupby(df, :algorithm), :gflops => mean => :avg_gflops, :gflops => std => :std_gflops)
    sort!(summary, :avg_gflops, rev = true)

    lines = String[]
    push!(lines, "| Algorithm | Avg GFLOPs | Std Dev |")
    push!(lines, "|-----------|------------|---------|")

    for row in eachrow(summary)
        avg_str = @sprintf("%.2f", row.avg_gflops)
        std_str = @sprintf("%.2f", row.std_gflops)
        push!(lines, "| $(row.algorithm) | $avg_str | $std_str |")
    end

    return join(lines, "\n")
end

"""
    upload_to_github(content::String, plot_files::Union{Nothing, Tuple}; 
                     repo="SciML/LinearSolve.jl", issue_number=669)

Upload benchmark results to GitHub issue as a comment.
"""
function upload_to_github(content::String, plot_files::Union{Nothing, Tuple};
        repo = "SciML/LinearSolve.jl", issue_number = 669)
    @info "Preparing to upload results to GitHub issue #$issue_number in $repo"

    try
        # Create GitHub authentication (requires GITHUB_TOKEN environment variable)
        auth = GitHub.authenticate(ENV["GITHUB_TOKEN"])

        # Get the repository
        repo_obj = GitHub.repo(repo)

        # Create the comment content
        comment_body = content

        # If we have plot files, we would need to upload them as attachments
        # GitHub API doesn't directly support image uploads in comments, so we'll just reference them
        if plot_files !== nothing
            png_file, pdf_file = plot_files
            comment_body *= "\n\n**Note**: Benchmark plots have been generated locally as `$png_file` and `$pdf_file`."
        end

        # Post the comment
        GitHub.create_comment(repo_obj, issue_number, comment_body, auth = auth)

        @info "Successfully posted benchmark results to GitHub issue #$issue_number"

    catch e
        @warn "Failed to upload to GitHub: $e"
        @info "Make sure you have set the GITHUB_TOKEN environment variable with appropriate permissions."

        # Save locally as fallback
        fallback_file = "autotune_results_$(replace(string(Dates.now()), ":" => "-")).md"
        open(fallback_file, "w") do f
            write(f, content)
        end
        @info "Results saved locally to $fallback_file"
    end
end
