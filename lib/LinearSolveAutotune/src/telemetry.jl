# Telemetry functionality for sharing benchmark results

"""
    get_gh_command()

Get the gh command, preferring the system-installed version if available,
falling back to the JLL-provided version.
"""
function get_gh_command()
    # First check if gh is installed on the system
    if !isnothing(Sys.which("gh"))
        return `gh`
    else
        # Use the JLL-provided gh
        return `$(gh_cli_jll.gh())`
    end
end

"""
    setup_github_authentication(; auto_login::Bool = true)

Set up GitHub authentication for telemetry uploads.
If auto_login is true and no authentication is found, will prompt to run gh auth login.
Returns an authentication method indicator if successful, nothing if setup fails.
"""
function setup_github_authentication(; auto_login::Bool = true)
    # 1. Check for `gh` CLI (system or JLL)
    gh_cmd = get_gh_command()

    # First check if already authenticated
    try
        # Suppress output of gh auth status check
        if success(pipeline(`$gh_cmd auth status`; stdout = devnull, stderr = devnull))
            # Check if logged in to github.com
            auth_status_output = read(`$gh_cmd auth status`, String)
            if contains(auth_status_output, "Logged in to github.com")
                println("✅ Found active `gh` CLI session. Will use it for upload.")
                return (:gh_cli, "GitHub CLI")
            end
        end
    catch e
        @debug "gh CLI auth status check failed: $e"
    end

    # 2. Check for GITHUB_TOKEN environment variable
    if haskey(ENV, "GITHUB_TOKEN") && !isempty(ENV["GITHUB_TOKEN"])
        auth = test_github_authentication(String(ENV["GITHUB_TOKEN"]))
        if auth !== nothing
            println("✅ Found GITHUB_TOKEN environment variable.")
            return (:token, auth)
        end
    end

    # 3. If auto_login is enabled, offer to authenticate
    if auto_login
        println("\n🔐 GitHub authentication not found.")
        println("   To share results with the community, authentication is required.")
        println("\nWould you like to authenticate with GitHub now? (y/n)")
        print("> ")
        response = readline()

        if lowercase(strip(response)) in ["y", "yes"]
            println("\n📝 Starting GitHub authentication...")
            println("   This will open your browser to authenticate with GitHub.")
            println("   Please follow the prompts to complete authentication.\n")

            try
                # Run gh auth login interactively (using system gh or JLL)
                run(`$gh_cmd auth login`)

                # Check if authentication succeeded
                if success(pipeline(`$gh_cmd auth status`; stdout = devnull, stderr = devnull))
                    auth_status_output = read(`$gh_cmd auth status`, String)
                    if contains(auth_status_output, "Logged in to github.com")
                        println("\n✅ Authentication successful! You can now share results.")
                        return (:gh_cli, "GitHub CLI")
                    end
                end
            catch e
                println("\n❌ Authentication failed: $e")
                println("   You can try again later or use a GitHub token instead.")
            end
        else
            println("\n📝 Skipping authentication. You can authenticate later by:")
            println("   1. Running: gh auth login")
            println("   2. Or setting: ENV[\"GITHUB_TOKEN\"] = \"your_token\"")
        end
    end

    # 4. No authentication available - return nothing
    return (nothing, nothing)
end

"""
    test_github_authentication(token::AbstractString)

Test GitHub authentication with a provided token.
Returns authentication object if successful, nothing otherwise.
"""
function test_github_authentication(token::AbstractString)
    println("🔍 Testing GitHub authentication...")
    try
        auth_result = GitHub.authenticate(token)
        # A simple API call to verify the token works
        GitHub.user(auth = auth_result)
        println("✅ Authentication successful!")
        flush(stdout)
        return auth_result
    catch e
        println("❌ Authentication failed. Please verify your token has 'issues:write' permission.")
        # Do not show full error to avoid leaking info
        return nothing
    end
end

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

### Performance Summary by Size Range
$(format_categories_markdown(categories))

### Detailed Results
$(format_detailed_results_markdown(successful_df))

### System Information
$(format_system_info_markdown(system_info))

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
    push!(lines, "- **Julia Version**: $(get(system_info, "julia_version", "unknown"))")
    # Handle both "os" and "os_version" keys, with os_name for display
    os_display = get(system_info, "os_name", "unknown")
    os_kernel = get(system_info, "os_version", get(system_info, "os", "unknown"))
    push!(lines, "- **OS**: $os_display ($os_kernel)")
    # Handle both "arch" and "architecture" keys
    push!(lines,
        "- **Architecture**: $(get(system_info, "architecture", get(system_info, "arch", "unknown")))")
    push!(lines, "- **CPU**: $(get(system_info, "cpu_name", "unknown"))")
    # Handle both "num_cores" and "cpu_cores" keys
    push!(lines, "- **Cores**: $(get(system_info, "cpu_cores", get(system_info, "num_cores", "unknown")))")
    # Handle both "num_threads" and "julia_threads" keys
    push!(lines,
        "- **Threads**: $(get(system_info, "julia_threads", get(system_info, "num_threads", "unknown")))")
    push!(lines, "- **BLAS**: $(get(system_info, "blas_vendor", "unknown"))")
    push!(lines, "- **MKL Available**: $(get(system_info, "mkl_available", false))")
    push!(lines,
        "- **Apple Accelerate Available**: $(get(system_info, "apple_accelerate_available", false))")
    # Handle both "has_cuda" and "cuda_available" keys
    push!(lines,
        "- **CUDA Available**: $(get(system_info, "cuda_available", get(system_info, "has_cuda", false)))")
    # Handle both "has_metal" and "metal_available" keys
    push!(lines,
        "- **Metal Available**: $(get(system_info, "metal_available", get(system_info, "has_metal", false)))")

    # Add package versions section
    if haskey(system_info, "package_versions")
        push!(lines, "")
        push!(lines, "### Package Versions")
        pkg_versions = system_info["package_versions"]

        # Sort packages for consistent display
        sorted_packages = sort(collect(keys(pkg_versions)))

        for pkg_name in sorted_packages
            version = pkg_versions[pkg_name]
            push!(lines, "- **$pkg_name**: $version")
        end
    end

    return join(lines, "\n")
end

"""
    format_categories_markdown(categories::Dict{String, String})

Format the categorized results as markdown, organized by element type.
"""
function format_categories_markdown(categories::Dict{String, String})
    if isempty(categories)
        return "No category recommendations available."
    end

    lines = String[]

    # Group categories by element type
    eltype_categories = Dict{String, Dict{String, String}}()

    for (key, algorithm) in categories
        # Parse key like "Float64_tiny (5-20)" -> eltype="Float64", range="tiny (5-20)"
        if contains(key, "_")
            eltype, range = split(key, "_", limit = 2)
            if !haskey(eltype_categories, eltype)
                eltype_categories[eltype] = Dict{String, String}()
            end
            eltype_categories[eltype][range] = algorithm
        else
            # Fallback for backward compatibility
            if !haskey(eltype_categories, "Mixed")
                eltype_categories["Mixed"] = Dict{String, String}()
            end
            eltype_categories["Mixed"][key] = algorithm
        end
    end

    # Define the proper order for size ranges
    size_order = ["tiny (5-20)", "small (20-100)", "medium (100-300)",
        "large (300-1000)", "big (10000+)"]

    # Custom sort function for ranges
    function sort_ranges(ranges_dict)
        sorted_pairs = []
        for size in size_order
            if haskey(ranges_dict, size)
                push!(sorted_pairs, (size, ranges_dict[size]))
            end
        end
        # Add any other ranges not in our predefined order (for backward compatibility)
        for (range, algo) in ranges_dict
            if !(range in size_order)
                push!(sorted_pairs, (range, algo))
            end
        end
        return sorted_pairs
    end

    # Format each element type
    for (eltype, ranges) in sort(eltype_categories)
        push!(lines, "#### Recommendations for $eltype")
        push!(lines, "")
        push!(lines, "| Size Range | Best Algorithm |")
        push!(lines, "|------------|----------------|")

        for (range, algorithm) in sort_ranges(ranges)
            push!(lines, "| $range | $algorithm |")
        end
        push!(lines, "")
    end

    return join(lines, "\n")
end

"""
    format_detailed_results_markdown(df::DataFrame)

Format detailed benchmark results as markdown tables, organized by element type.
Includes both summary statistics and raw performance data in collapsible sections.
"""
function format_detailed_results_markdown(df::DataFrame)
    lines = String[]

    # Get unique element types
    eltypes = unique(df.eltype)

    for eltype in eltypes
        push!(lines, "#### Results for $eltype")
        push!(lines, "")

        # Filter results for this element type
        eltype_df = filter(row -> row.eltype == eltype, df)

        if nrow(eltype_df) == 0
            push!(lines, "No results for this element type.")
            push!(lines, "")
            continue
        end

        # Create a summary table with average performance per algorithm for this element type
        summary = combine(groupby(eltype_df, :algorithm),
            :gflops => mean => :avg_gflops,
            :gflops => std => :std_gflops,
            nrow => :num_tests)
        sort!(summary, :avg_gflops, rev = true)

        push!(lines, "##### Summary Statistics")
        push!(lines, "")
        push!(lines, "| Algorithm | Avg GFLOPs | Std Dev | Tests |")
        push!(lines, "|-----------|------------|---------|-------|")

        for row in eachrow(summary)
            avg_str = @sprintf("%.2f", row.avg_gflops)
            std_str = @sprintf("%.2f", row.std_gflops)
            push!(lines, "| $(row.algorithm) | $avg_str | $std_str | $(row.num_tests) |")
        end

        push!(lines, "")

        # Add raw performance data in collapsible details blocks for each algorithm
        push!(lines, "<details>")
        push!(lines, "<summary>Raw Performance Data</summary>")
        push!(lines, "")

        # Get unique algorithms for this element type
        algorithms = unique(eltype_df.algorithm)

        for algorithm in sort(algorithms)
            # Filter data for this algorithm
            algo_df = filter(row -> row.algorithm == algorithm, eltype_df)

            # Sort by size for better readability
            sort!(algo_df, :size)

            push!(lines, "##### $algorithm")
            push!(lines, "")
            push!(lines, "| Matrix Size | GFLOPs | Status |")
            push!(lines, "|-------------|--------|--------|")

            for row in eachrow(algo_df)
                gflops_str = row.success ? @sprintf("%.3f", row.gflops) : "N/A"
                status = row.success ? "✅ Success" : "❌ Failed"
                push!(lines, "| $(row.size) | $gflops_str | $status |")
            end

            push!(lines, "")
        end

        push!(lines, "</details>")
        push!(lines, "")
    end

    return join(lines, "\n")
end

"""
    upload_to_github(content::String, plot_files, auth_info::Tuple,
                     results_df::DataFrame, system_info::Dict, categories::Dict)

Create a GitHub issue with benchmark results for community data collection.
Note: plot_files parameter is kept for compatibility but not used.
"""
function upload_to_github(content::String, plot_files, auth_info::Tuple,
        results_df::DataFrame, system_info::Dict, categories::Dict)
    auth_method, auth_data = auth_info

    if auth_method === nothing
        @info "⚠️  No GitHub authentication available. Saving results locally instead of uploading."
        # Save locally as fallback
        fallback_file = "autotune_results_$(replace(string(Dates.now()), ":" => "-")).md"
        open(fallback_file, "w") do f
            write(f, content)
        end
        @info "📁 Results saved locally to $fallback_file"
        return
    end

    @info "📤 Preparing to upload benchmark results..."

    try
        target_repo = "SciML/LinearSolve.jl"
        issue_number = 669  # The existing issue for collecting autotune results

        # Construct comment body
        cpu_name = get(system_info, "cpu_name", "unknown")
        os_name = get(system_info, "os", "unknown")
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")

        comment_body = """
        ## Benchmark Results: $cpu_name on $os_name ($timestamp)

        $content

        ---

        ### System Summary
        - **CPU:** $cpu_name
        - **OS:** $os_name  
        - **Timestamp:** $timestamp

        🤖 *Generated automatically by LinearSolve.jl autotune system*
        """

        @info "📝 Adding comment to issue #669..."

        issue_url = nothing
        if auth_method == :gh_cli
            issue_url = comment_on_issue_gh(target_repo, issue_number, comment_body)
        elseif auth_method == :token
            issue_url = comment_on_issue_api(target_repo, issue_number, comment_body, auth_data)
        end

        if issue_url !== nothing
            @info "✅ Successfully added benchmark results to issue: $issue_url"
            @info "🔗 Your benchmark data has been shared with the LinearSolve.jl community!"
            @info "💡 View all community benchmark data: https://github.com/SciML/LinearSolve.jl/issues/669"
        else
            error("Failed to add comment to GitHub issue")
        end

    catch e
        @warn "❌ Failed to add comment to GitHub issue: $e"
        @info "💡 This could be due to network issues, repository permissions, or API limits."

        # Save locally as fallback
        timestamp = replace(string(Dates.now()), ":" => "-")
        fallback_file = "autotune_results_$(timestamp).md"
        open(fallback_file, "w") do f
            write(f, content)
        end
        @info "📁 Results saved locally to $fallback_file as backup"
    end
end

"""
    upload_plots_to_gist(plot_files::Union{Nothing, Tuple, Dict}, auth, eltype_str::String)

Upload plot files to a GitHub Gist by creating a gist and then cloning/updating it with binary files.
"""
function upload_plots_to_gist(plot_files::Union{Nothing, Tuple, Dict}, auth, eltype_str::String)
    if plot_files === nothing
        return nothing, Dict{String, String}()
    end

    try
        # Handle different plot_files formats
        files_to_upload = if isa(plot_files, Tuple)
            # Legacy format: (png_file, pdf_file)
            Dict("benchmark_plot.png" => plot_files[1], "benchmark_plot.pdf" => plot_files[2])
        elseif isa(plot_files, Dict)
            plot_files
        else
            return nothing, Dict{String, String}()
        end

        # Filter existing files
        existing_files = Dict(k => v for (k, v) in files_to_upload if isfile(v))
        if isempty(existing_files)
            return nothing, Dict{String, String}()
        end

        # Create README content
        readme_content = """
# LinearSolve.jl Benchmark Plots

**Element Type:** $eltype_str  
**Generated:** $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))

## Files

"""
        for (name, _) in existing_files
            readme_content *= "- `$name`\n"
        end

        readme_content *= """

## Viewing the Plots

The PNG images can be viewed directly in the browser. Click on any `.png` file above to view it.

---
*Generated automatically by LinearSolve.jl autotune system*
"""

        # Create initial gist with README
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        gist_desc = "LinearSolve.jl Benchmark Plots - $eltype_str - $timestamp"

        gist_files = Dict{String, Any}()
        gist_files["README.md"] = Dict("content" => readme_content)

        params = Dict(
            "description" => gist_desc,
            "public" => true,
            "files" => gist_files
        )

        # Create the gist
        gist = GitHub.create_gist(; params = params, auth = auth)
        gist_url = gist.html_url
        gist_id = split(gist_url, "/")[end]
        username = split(gist_url, "/")[end - 1]

        # Now clone the gist and add the binary files
        temp_dir = mktempdir()
        try
            # Clone using HTTPS with token authentication
            clone_url = "https://$(auth.token)@gist.github.com/$gist_id.git"
            run(`git clone $clone_url $temp_dir`)

            # Copy all plot files to the gist directory
            for (name, filepath) in existing_files
                target_path = joinpath(temp_dir, name)
                cp(filepath, target_path; force = true)
            end

            # Configure git user for the commit
            cd(temp_dir) do
                # Set a generic user for the commit
                run(`git config user.email "linearsolve-autotune@example.com"`)
                run(`git config user.name "LinearSolve Autotune"`)

                # Stage, commit and push the changes
                run(`git add .`)
                run(`git commit -m "Add benchmark plots"`)
                run(`git push`)
            end

            @info "✅ Successfully uploaded plots to gist: $gist_url"

            # Construct raw URLs for the uploaded files
            raw_urls = Dict{String, String}()
            for (name, _) in existing_files
                raw_urls[name] = "https://gist.githubusercontent.com/$username/$gist_id/raw/$name"
            end

            return gist_url, raw_urls

        finally
            # Clean up temporary directory
            rm(temp_dir; recursive = true, force = true)
        end

    catch e
        @warn "Failed to upload plots to gist via API: $e"
        # Fall back to HTML with embedded images
        return upload_plots_to_gist_fallback(existing_files, auth, eltype_str)
    end
end

"""
    upload_plots_to_gist_fallback(files, auth, eltype_str)

Fallback method that creates an HTML file with embedded base64 images.
"""
function upload_plots_to_gist_fallback(files::Dict, auth, eltype_str::String)
    try
        # Create an HTML file with embedded images
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LinearSolve.jl Benchmark Plots - $eltype_str</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .plot { margin: 20px 0; text-align: center; }
                img { max-width: 100%; height: auto; border: 1px solid #ddd; padding: 10px; }
                h2 { color: #333; }
            </style>
        </head>
        <body>
            <h1>LinearSolve.jl Benchmark Plots</h1>
            <h2>Element Type: $eltype_str</h2>
        """

        # Read files and embed as base64
        for (name, filepath) in files
            if isfile(filepath) && endswith(filepath, ".png")
                # Read as binary and encode to base64
                binary_content = read(filepath)
                base64_content = base64encode(binary_content)
                data_uri = "data:image/png;base64,$base64_content"

                # Add to HTML
                html_content *= """
                <div class="plot">
                    <h3>$(basename(filepath))</h3>
                    <img src="$data_uri" alt="$name">
                </div>
                """
            end
        end

        html_content *= """
        </body>
        </html>
        """

        # Create gist with HTML file
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        gist_desc = "LinearSolve.jl Benchmark Plots - $eltype_str - $timestamp"

        gist_files = Dict{String, Any}()
        gist_files["plots.html"] = Dict("content" => html_content)

        params = Dict(
            "description" => gist_desc,
            "public" => true,
            "files" => gist_files
        )

        gist = GitHub.create_gist(; params = params, auth = auth)

        @info "✅ Uploaded plots to gist (HTML fallback): $(gist.html_url)"
        return gist.html_url, Dict{String, String}()

    catch e
        @warn "Failed to upload plots to gist (fallback): $e"
        return nothing, Dict{String, String}()
    end
end

"""
    upload_plots_to_gist_gh(plot_files::Union{Nothing, Tuple, Dict}, eltype_str::String)

Upload plot files to a GitHub Gist using gh CLI by cloning, adding files, and pushing.
"""
function upload_plots_to_gist_gh(plot_files::Union{Nothing, Tuple, Dict}, eltype_str::String)
    if plot_files === nothing
        return nothing, Dict{String, String}()
    end

    try
        gh_cmd = get_gh_command()
        # Handle different plot_files formats
        files_to_upload = if isa(plot_files, Tuple)
            # Legacy format: (png_file, pdf_file)
            Dict("benchmark_plot.png" => plot_files[1], "benchmark_plot.pdf" => plot_files[2])
        elseif isa(plot_files, Dict)
            plot_files
        else
            return nothing, Dict{String, String}()
        end

        # Filter existing files
        existing_files = Dict(k => v for (k, v) in files_to_upload if isfile(v))
        if isempty(existing_files)
            return nothing, Dict{String, String}()
        end

        # Create initial gist with a README
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        gist_desc = "LinearSolve.jl Benchmark Plots - $eltype_str - $timestamp"

        # Create README content
        readme_content = """
# LinearSolve.jl Benchmark Plots

**Element Type:** $eltype_str  
**Generated:** $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))

## Files

"""
        for (name, _) in existing_files
            readme_content *= "- `$name`\n"
        end

        readme_content *= """

## Viewing the Plots

The PNG images can be viewed directly in the browser. Click on any `.png` file above to view it.

---
*Generated automatically by LinearSolve.jl autotune system*
"""

        # Create temporary file for README
        readme_file = tempname() * "_README.md"
        open(readme_file, "w") do f
            write(f, readme_content)
        end

        # Create initial gist with README
        out = Pipe()
        err = Pipe()
        run(pipeline(`$gh_cmd gist create -d $gist_desc -p $readme_file`, stdout = out, stderr = err))
        close(out.in)
        close(err.in)

        gist_url = strip(read(out, String))
        err_str = read(err, String)

        if !startswith(gist_url, "https://gist.github.com/")
            error("gh gist create did not return a valid URL. Output: $gist_url. Error: $err_str")
        end

        # Extract gist ID from URL
        gist_id = split(gist_url, "/")[end]

        # Clone the gist
        temp_dir = mktempdir()
        try
            # Clone the gist
            run(`$gh_cmd gist clone $gist_id $temp_dir`)

            # Copy all plot files to the gist directory
            for (name, filepath) in existing_files
                target_path = joinpath(temp_dir, name)
                cp(filepath, target_path; force = true)
            end

            # Stage, commit and push the changes
            cd(temp_dir) do
                run(`git add .`)
                run(`git commit -m "Add benchmark plots"`)
                run(`git push`)
            end

            @info "✅ Successfully uploaded plots to gist: $gist_url"

            # Get username for constructing raw URLs
            username_out = Pipe()
            run(pipeline(`$gh_cmd api user --jq .login`, stdout = username_out))
            close(username_out.in)
            username = strip(read(username_out, String))

            # Construct raw URLs for the uploaded files
            raw_urls = Dict{String, String}()
            for (name, _) in existing_files
                raw_urls[name] = "https://gist.githubusercontent.com/$username/$gist_id/raw/$name"
            end

            return gist_url, raw_urls

        finally
            # Clean up temporary directory
            rm(temp_dir; recursive = true, force = true)
            rm(readme_file; force = true)
        end

    catch e
        @warn "Failed to upload plots to gist via gh CLI: $e"
        return nothing, Dict{String, String}()
    end
end

"""
    comment_on_issue_api(target_repo, issue_number, body, auth)

Add a comment to an existing GitHub issue using the GitHub API.
"""
function comment_on_issue_api(target_repo, issue_number, body, auth)
    try
        repo_obj = GitHub.repo(target_repo; auth = auth)
        issue = GitHub.issue(repo_obj, issue_number; auth = auth)
        comment = GitHub.create_comment(repo_obj, issue, body; auth = auth)
        @info "✅ Added comment to issue #$(issue_number) via API"
        return "https://github.com/$(target_repo)/issues/$(issue_number)#issuecomment-$(comment.id)"
    catch e
        @warn "Failed to add comment via API: $e"
        return nothing
    end
end

"""
    comment_on_issue_gh(target_repo, issue_number, body)

Add a comment to an existing GitHub issue using the `gh` CLI.
"""
function comment_on_issue_gh(target_repo, issue_number, body)
    err_str = ""
    out_str = ""
    try
        gh_cmd = get_gh_command()
        # Use a temporary file for the body to avoid command line length limits
        mktemp() do path, io
            write(io, body)
            flush(io)

            # Construct and run the gh command
            cmd = `$gh_cmd issue comment $issue_number --repo $target_repo --body-file $path`

            out = Pipe()
            err = Pipe()
            run(pipeline(cmd, stdout = out, stderr = err))
            close(out)
            close(err)
            out_str = read(out, String)
            err_str = read(err, String)

            @info "✅ Added comment to issue #$(issue_number) via `gh` CLI"
            return "https://github.com/$(target_repo)/issues/$(issue_number)"
        end
    catch e
        @warn "Failed to add comment via `gh` CLI: $e" out_str err_str
        return nothing
    end
end

"""
    create_benchmark_issue_api(target_repo, title, body, auth)

Create a GitHub issue using the GitHub.jl API.
"""
function create_benchmark_issue_api(target_repo, title, body, auth)
    try
        repo_obj = GitHub.repo(target_repo; auth = auth)
        params = Dict("title" => title, "body" => body, "labels" => ["benchmark-data"])
        issue_result = GitHub.create_issue(repo_obj; params = params, auth = auth)
        @info "✅ Created benchmark results issue #$(issue_result.number) via API"
        return issue_result.html_url
    catch e
        @warn "Failed to create benchmark issue via API: $e"
        return nothing
    end
end

"""
    create_benchmark_issue_gh(target_repo, title, body)

Create a GitHub issue using the `gh` CLI.
"""
function create_benchmark_issue_gh(target_repo, title, body)
    err_str = ""
    out_str = ""
    try
        gh_cmd = get_gh_command()
        # Use a temporary file for the body to avoid command line length limits
        mktemp() do path, io
            write(io, body)
            flush(io)

            # Construct and run the gh command
            cmd = `$gh_cmd issue create --repo $target_repo --title $title --body-file $path --label benchmark-data`

            out = Pipe()
            err = Pipe()
            run(pipeline(cmd, stdout = out, stderr = err))
            closewrite(out)
            closewrite(err)
            out_str = read(out, String)
            err_str = read(err, String)
            # Capture output to get the issue URL
            issue_url = strip(out_str)

            if !startswith(issue_url, "https://github.com/")
                error("gh CLI command did not return a valid URL. Output: $issue_url. Error: $err_str")
            end

            @info "✅ Created benchmark results issue via `gh` CLI"
            return issue_url
        end
    catch e
        @warn "Failed to create benchmark issue via `gh` CLI: $e" out_str err_str
        return nothing
    end
end
