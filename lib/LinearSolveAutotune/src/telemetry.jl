# Telemetry functionality for sharing benchmark results

"""
    setup_github_authentication()

Set up GitHub authentication for telemetry uploads.
Returns authentication object if successful, nothing if setup needed.
"""
function setup_github_authentication()
    # Check if GITHUB_TOKEN environment variable exists
    if haskey(ENV, "GITHUB_TOKEN") && !isempty(ENV["GITHUB_TOKEN"])
        try
            auth = GitHub.authenticate(ENV["GITHUB_TOKEN"])
            @info "✅ GitHub authentication successful - ready to share results!"
            return auth
        catch e
            @warn "❌ GITHUB_TOKEN exists but authentication failed: $e"
            @info "Please check that your GITHUB_TOKEN is valid and has appropriate permissions"
            return nothing
        end
    end
    
    # No environment variable - provide setup instructions and wait for token
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts
        println()
        println("🚀 Help Improve LinearSolve.jl for Everyone!")
        println("="^50)
        println("Your benchmark results help the community by improving automatic")
        println("algorithm selection across different hardware configurations.")
        println()
        println("📋 Quick GitHub Token Setup (takes 30 seconds):")
        println()
        println("1️⃣  Open: https://github.com/settings/tokens?type=beta")
        println("2️⃣  Click 'Generate new token'")
        println("3️⃣  Set:")
        println("    • Name: 'LinearSolve Autotune'")
        println("    • Expiration: 90 days")
        println("    • Repository access: 'Public Repositories (read-only)'")
        println("4️⃣  Click 'Generate token' and copy it")
        println()
        println("🔑 Paste your GitHub token here (or press Enter to skip):")
        print("Token: ")
        flush(stdout)  # Ensure the prompt is displayed before reading
        
        # Add a small safety delay to ensure prompt is fully displayed
        sleep(0.05)
        
        # Read the token input
        token = ""
        try
            token = strip(readline())
        catch e
            println("❌ Input error: $e")
            println("🔄 Please try again...")
            flush(stdout)
            sleep(0.1)
            continue
        end
        
        if !isempty(token)
            # Wrap everything in a protective try-catch to prevent REPL interference
            auth_success = false
            auth_result = nothing
            
            try
                println("🔍 Testing token...")
                flush(stdout)
                
                # Clean the token of any potential whitespace/newlines
                clean_token = strip(replace(token, r"\s+" => ""))
                ENV["GITHUB_TOKEN"] = clean_token
                
                # Test authentication
                auth_result = GitHub.authenticate(clean_token)
                
                println("✅ Perfect! Authentication successful - your results will help everyone!")
                flush(stdout)
                auth_success = true
                
            catch e
                println("❌ Token authentication failed: $e")
                println("💡 Make sure the token:")
                println("   • Has 'public_repo' or 'Public Repositories' access")
                println("   • Was copied completely without extra characters")
                println("   • Is not expired")
                flush(stdout)
                
                # Clean up on failure
                if haskey(ENV, "GITHUB_TOKEN")
                    delete!(ENV, "GITHUB_TOKEN")
                end
                auth_success = false
            end
            
            if auth_success && auth_result !== nothing
                return auth_result
            else
                attempts += 1
                if attempts < max_attempts
                    println("🔄 Let's try again...")
                    flush(stdout)
                    continue
                end
            end
        else
            attempts += 1
            if attempts < max_attempts
                println()
                println("⏰ Hold on! This really helps the LinearSolve.jl community.")
                println("   Your hardware's benchmark data improves algorithm selection for everyone.")
                println("   It only takes 30 seconds and makes LinearSolve.jl better for all users.")
                println()
                println("🤝 Please help the community - try setting up the token?")
                print("Response (y/n): ")
                flush(stdout)  # Ensure the prompt is displayed before reading
                response = strip(lowercase(readline()))
                if response == "n" || response == "no"
                    attempts += 1
                    if attempts < max_attempts
                        println("🙏 One more chance - the community really benefits from diverse hardware data!")
                        continue
                    end
                else
                    # Reset attempts if they want to try again
                    attempts = 0
                    continue
                end
            end
        end
    end
    
    println()
    println("📊 Okay, continuing without telemetry. Results will be saved locally.")
    println("💡 You can always run `export GITHUB_TOKEN=your_token` and restart Julia later.")
    
    return nothing
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

        push!(lines, "| Algorithm | Avg GFLOPs | Std Dev | Tests |")
        push!(lines, "|-----------|------------|---------|-------|")

        for row in eachrow(summary)
            avg_str = @sprintf("%.2f", row.avg_gflops)
            std_str = @sprintf("%.2f", row.std_gflops)
            push!(lines, "| $(row.algorithm) | $avg_str | $std_str | $(row.num_tests) |")
        end
        
        push!(lines, "")
    end

    return join(lines, "\n")
end

"""
    upload_to_github(content::String, plot_files::Union{Nothing, Tuple, Dict}, auth; 
                     repo="SciML/LinearSolve.jl", issue_number=669)

Upload benchmark results to GitHub issue as a comment.
Requires a pre-authenticated GitHub.jl auth object.
"""
function upload_to_github(content::String, plot_files::Union{Nothing, Tuple, Dict}, auth;
        repo = "SciML/LinearSolve.jl", issue_number = 669)
    
    if auth === nothing
        @info "⚠️  No GitHub authentication available. Saving results locally instead of uploading."
        # Save locally as fallback
        fallback_file = "autotune_results_$(replace(string(Dates.now()), ":" => "-")).md"
        open(fallback_file, "w") do f
            write(f, content)
        end
        @info "📁 Results saved locally to $fallback_file"
        return
    end
    
    @info "📤 Uploading results to GitHub issue #$issue_number in $repo"

    try

        # Get the repository
        repo_obj = GitHub.repo(repo)

        # Create the comment content
        comment_body = content

        # Handle different plot file formats
        if plot_files !== nothing
            if isa(plot_files, Tuple)
                # Backward compatibility: single plot
                png_file, pdf_file = plot_files
                comment_body *= "\n\n**Note**: Benchmark plots have been generated locally as `$png_file` and `$pdf_file`."
            elseif isa(plot_files, Dict)
                # Multiple plots by element type
                comment_body *= "\n\n**Note**: Benchmark plots have been generated locally:"
                for (eltype, files) in plot_files
                    png_file, pdf_file = files
                    comment_body *= "\n- $eltype: `$png_file` and `$pdf_file`"
                end
            end
        end

        # Post the comment
        GitHub.create_comment(repo_obj, issue_number, comment_body, auth = auth)

        @info "✅ Successfully posted benchmark results to GitHub issue #$issue_number"

    catch e
        @warn "❌ Failed to upload to GitHub: $e"
        @info "💡 This could be due to network issues, repository permissions, or API limits."

        # Save locally as fallback
        fallback_file = "autotune_results_$(replace(string(Dates.now()), ":" => "-")).md"
        open(fallback_file, "w") do f
            write(f, content)
        end
        @info "📁 Results saved locally to $fallback_file as backup"
    end
end
