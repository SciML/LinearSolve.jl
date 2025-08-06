# Telemetry functionality for sharing benchmark results

"""
    setup_github_authentication()

Set up GitHub authentication for telemetry uploads.
Returns an authentication method indicator if successful, nothing if setup fails.
"""
function setup_github_authentication()
    # 1. Check for `gh` CLI
    if !isnothing(Sys.which("gh"))
        try
            # Suppress output of gh auth status check
            if success(pipeline(`gh auth status`; stdout=devnull, stderr=devnull))
                # Check if logged in to github.com
                auth_status_output = read(`gh auth status`, String)
                if contains(auth_status_output, "Logged in to github.com")
                    println("✅ Found active `gh` CLI session. Will use it for upload.")
                    return (:gh_cli, "GitHub CLI")
                end
            end
        catch e
            @warn "An error occurred while checking `gh` CLI status. Falling back to token auth. Error: $e"
        end
    end

    # 2. Check for GITHUB_TOKEN environment variable
    if haskey(ENV, "GITHUB_TOKEN") && !isempty(ENV["GITHUB_TOKEN"])
        auth = test_github_authentication(String(ENV["GITHUB_TOKEN"]))
        if auth !== nothing
            return (:token, auth)
        end
    end

    # 3. No environment variable or gh cli - provide setup instructions and get token
    max_input_attempts = 3

    for input_attempt in 1:max_input_attempts
        println()
        println("🚀 Help Improve LinearSolve.jl for Everyone!")
        println("="^50)
        println("Your benchmark results help the community by improving automatic")
        println("algorithm selection across different hardware configurations.")
        println()
        println("💡 Easiest method: install GitHub CLI (`gh`) and run `gh auth login`.")
        println("   Alternatively, create a token with 'issues:write' scope.")
        println()
        println("📋 Quick GitHub Token Setup (if not using `gh`):")
        println()
        println("1️⃣  Open: https://github.com/settings/tokens/new?scopes=issues:write&description=LinearSolve.jl%20Telemetry")
        println("2️⃣  Click 'Generate token' and copy it")
        println()
        println("🔑 Paste your GitHub token here:")
        println("   (If it shows julia> prompt, just paste the token there and press Enter)")
        print("Token: ")
        flush(stdout)

        # Get token input
        token = ""
        try
            sleep(0.1)
            token = String(strip(readline()))
        catch e
            println("❌ Input error: $e. Please try again.")
            continue
        end

        if !isempty(token)
            clean_token = strip(replace(token, r"[\r\n\t ]+" => ""))
            if length(clean_token) < 10
                println("❌ Token seems too short. Please check and try again.")
                continue
            end

            ENV["GITHUB_TOKEN"] = clean_token
            auth_result = test_github_authentication(clean_token)
            if auth_result !== nothing
                return (:token, auth_result)
            end
            delete!(ENV, "GITHUB_TOKEN")
        end

        if input_attempt < max_input_attempts
            println("\n🤝 Please try again - it only takes 30 seconds and greatly helps the community.")
        end
    end

    println("\n📊 Continuing without telemetry. Results will be saved locally.")
    println("💡 You can set GITHUB_TOKEN or log in with `gh auth login` and restart Julia later.")

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
        # Parse key like "Float64_0-128" -> eltype="Float64", range="0-128"
        if contains(key, "_")
            eltype, range = split(key, "_", limit=2)
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
    
    # Format each element type
    for (eltype, ranges) in sort(eltype_categories)
        push!(lines, "#### Recommendations for $eltype")
        push!(lines, "")
        push!(lines, "| Size Range | Best Algorithm |")
        push!(lines, "|------------|----------------|")

        for (range, algorithm) in sort(ranges)
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
    upload_to_github(content::String, plot_files::Union{Nothing, Tuple, Dict}, auth_info::Tuple,
                     results_df::DataFrame, system_info::Dict, categories::Dict)

Create a GitHub issue with benchmark results for community data collection.
"""
function upload_to_github(content::String, plot_files::Union{Nothing, Tuple, Dict}, auth_info::Tuple,
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
        
        # First, upload plots to a gist if available
        gist_url = nothing
        raw_urls = Dict{String, String}()
        plot_links = ""
        
        if plot_files !== nothing
            @info "📊 Uploading plots to GitHub Gist..."
            
            # Get element type for labeling
            eltype_str = if !isempty(results_df)
                unique_eltypes = unique(results_df.eltype)
                join(unique_eltypes, ", ")
            else
                "Mixed"
            end
            
            if auth_method == :gh_cli
                gist_url, raw_urls = upload_plots_to_gist_gh(plot_files, eltype_str)
            elseif auth_method == :token
                gist_url, raw_urls = upload_plots_to_gist(plot_files, auth_data, eltype_str)
            end
            
            if gist_url !== nothing
                # Add plot links section to the content
                plot_links = """
                
                ### 📊 Benchmark Plots
                
                View all plots in the gist: [Benchmark Plots Gist]($gist_url)
                
                """
                
                # Embed PNG images directly in the markdown if we have raw URLs
                for (name, url) in raw_urls
                    if endswith(name, ".png")
                        plot_links *= """
                        #### $name
                        ![$(name)]($url)
                        
                        """
                    end
                end
                
                plot_links *= "---\n"
            end
        end
        
        # Construct comment body
        cpu_name = get(system_info, "cpu_name", "unknown")
        os_name = get(system_info, "os", "unknown")
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")
        
        comment_body = """
        ## Benchmark Results: $cpu_name on $os_name ($timestamp)
        $plot_links
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
            if gist_url !== nothing
                @info "📊 Plots available at: $gist_url"
            end
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
        gist = GitHub.create_gist(; params=params, auth=auth)
        gist_url = gist.html_url
        gist_id = split(gist_url, "/")[end]
        username = split(gist_url, "/")[end-1]
        
        # Now clone the gist and add the binary files
        temp_dir = mktempdir()
        try
            # Clone using HTTPS with token authentication
            clone_url = "https://$(auth.token)@gist.github.com/$gist_id.git"
            run(`git clone $clone_url $temp_dir`)
            
            # Copy all plot files to the gist directory
            for (name, filepath) in existing_files
                target_path = joinpath(temp_dir, name)
                cp(filepath, target_path; force=true)
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
            rm(temp_dir; recursive=true, force=true)
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
        
        gist = GitHub.create_gist(; params=params, auth=auth)
        
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
        run(pipeline(`gh gist create -d $gist_desc -p $readme_file`, stdout=out, stderr=err))
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
            run(`gh gist clone $gist_id $temp_dir`)
            
            # Copy all plot files to the gist directory
            for (name, filepath) in existing_files
                target_path = joinpath(temp_dir, name)
                cp(filepath, target_path; force=true)
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
            run(pipeline(`gh api user --jq .login`, stdout=username_out))
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
            rm(temp_dir; recursive=true, force=true)
            rm(readme_file; force=true)
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
        repo_obj = GitHub.repo(target_repo; auth=auth)
        issue = GitHub.issue(repo_obj, issue_number; auth=auth)
        comment = GitHub.create_comment(repo_obj, issue, body; auth=auth)
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
        # Use a temporary file for the body to avoid command line length limits
        mktemp() do path, io
            write(io, body)
            flush(io)
            
            # Construct and run the gh command
            cmd = `gh issue comment $issue_number --repo $target_repo --body-file $path`
            
            out = Pipe()
            err = Pipe()
            run(pipeline(cmd, stdout=out, stderr=err))
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
        repo_obj = GitHub.repo(target_repo; auth=auth)
        params = Dict("title" => title, "body" => body, "labels" => ["benchmark-data"])
        issue_result = GitHub.create_issue(repo_obj; params=params, auth=auth)
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
        # Use a temporary file for the body to avoid command line length limits
        mktemp() do path, io
            write(io, body)
            flush(io)
            
            # Construct and run the gh command
            cmd = `gh issue create --repo $target_repo --title $title --body-file $path --label benchmark-data`
            
            out = Pipe()
            err = Pipe()
            run(pipeline(cmd, stdout=out, stderr=err))
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