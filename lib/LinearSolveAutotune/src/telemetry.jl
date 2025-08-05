# Telemetry functionality for sharing benchmark results

"""
    setup_github_authentication()

Set up GitHub authentication for telemetry uploads.
Returns authentication object if successful, nothing if setup needed.
"""
function setup_github_authentication()
    # Check if GITHUB_TOKEN environment variable exists
    if haskey(ENV, "GITHUB_TOKEN") && !isempty(ENV["GITHUB_TOKEN"])
        return test_github_authentication(String(ENV["GITHUB_TOKEN"]))
    end
    
    # No environment variable - provide setup instructions and get token
    max_input_attempts = 3
    
    for input_attempt in 1:max_input_attempts
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
        println("    • Permissions: Enable 'Issues: Write'")
        println("4️⃣  Click 'Generate token' and copy it")
        println()
        println("🔑 Paste your GitHub token here:")
        println("    (If it shows julia> prompt, just paste the token there and press Enter)")
        print("Token: ")
        flush(stdout)
        
        # Get token input - handle both direct input and REPL interpretation
        token = ""
        try
            sleep(0.1)  # Small delay for I/O stability
            input_line = String(strip(readline()))
            
            # If we got direct input, use it
            if !isempty(input_line)
                token = input_line
            else
                # Check if token was interpreted as Julia code and became a variable
                # Look for common GitHub token patterns in global variables
                println("🔍 Looking for token that may have been interpreted as Julia code...")
                for name in names(Main, all=true)
                    if startswith(string(name), "github_pat_") || startswith(string(name), "ghp_")
                        try
                            value = getfield(Main, name)
                            if isa(value, AbstractString) && length(value) > 20
                                println("✅ Found token variable: $(name)")
                                token = String(value)
                                break
                            end
                        catch
                            continue
                        end
                    end
                end
                
                # If still no token, try one more direct input
                if isempty(token)
                    println("💡 Please paste your token again (make sure to press Enter after):")
                    print("Token: ")
                    flush(stdout)
                    sleep(0.1)
                    token = String(strip(readline()))
                end
            end
            
        catch e
            println("❌ Input error: $e")
            println("💡 No worries - this sometimes happens with token input")
            continue
        end
        
        if !isempty(token)
            # Clean and validate token format
            clean_token = strip(replace(token, r"[\r\n\t ]+" => ""))
            if length(clean_token) < 10
                println("❌ Token seems too short. Please check and try again.")
                continue
            end
            
            # Set environment variable
            ENV["GITHUB_TOKEN"] = clean_token
            
            # Test authentication with multiple attempts (addressing the "third attempt works" issue)
            auth_result = test_github_authentication(clean_token)
            if auth_result !== nothing
                return auth_result
            end
            
            # If all authentication attempts failed, clean up and continue to next input attempt
            delete!(ENV, "GITHUB_TOKEN")
        end
        
        # Handle skip attempts
        if input_attempt < max_input_attempts
            println()
            println("⏰ This really helps the LinearSolve.jl community!")
            println("   Your hardware's benchmark data improves algorithm selection for everyone.")
            println("🤝 Please try again - it only takes 30 seconds.")
        end
    end
    
    println()
    println("📊 Continuing without telemetry. Results will be saved locally.")
    println("💡 You can set GITHUB_TOKEN environment variable and restart Julia later.")
    
    return nothing
end

"""
    test_github_authentication(token::AbstractString)

Test GitHub authentication with up to 3 attempts to handle connection warmup issues.
Returns authentication object if successful, nothing otherwise.
"""
function test_github_authentication(token::AbstractString)
    max_auth_attempts = 3
    
    println("🔍 Testing GitHub authentication...")
    println("📏 Token length: $(length(token))")
    flush(stdout)
    
    for auth_attempt in 1:max_auth_attempts
        try
            if auth_attempt == 1
                println("🌐 Establishing connection to GitHub API...")
            elseif auth_attempt == 2
                println("🔄 Retrying connection (sometimes GitHub needs warmup)...")
            else
                println("🎯 Final authentication attempt...")
            end
            flush(stdout)
            
            # Add delay between attempts to handle timing issues
            if auth_attempt > 1
                sleep(0.5)
            end
            
            # Test authentication
            auth_result = GitHub.authenticate(token)
            
            # If we get here, authentication worked
            println("✅ Authentication successful - your results will help everyone!")
            flush(stdout)
            return auth_result
            
        catch e
            println("❌ Attempt $auth_attempt failed: $(typeof(e))")
            if auth_attempt < max_auth_attempts
                println("   Retrying in a moment...")
            else
                println("   All authentication attempts failed.")
                # Show safe preview of token for debugging
                if length(token) > 8
                    token_preview = token[1:4] * "..." * token[end-3:end]
                    println("🔍 Token preview: $token_preview")
                end
                println("💡 Please verify your token has 'Issues: Write' permission and try again.")
            end
            flush(stdout)
        end
    end
    
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
    upload_to_github(content::String, plot_files::Union{Nothing, Tuple, Dict}, auth,
                     results_df::DataFrame, system_info::Dict, categories::Dict)

Create a GitHub issue with benchmark results for community data collection.
Requires a pre-authenticated GitHub.jl auth object.
"""
function upload_to_github(content::String, plot_files::Union{Nothing, Tuple, Dict}, auth,
                         results_df::DataFrame, system_info::Dict, categories::Dict)
    
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
    
    @info "📤 Creating GitHub issue with benchmark results for community data collection..."

    try
        # Create issue with benchmark data
        target_repo = "SciML/LinearSolve.jl"
        issue_result = create_benchmark_issue(target_repo, content, auth, system_info)
        
        if issue_result !== nothing
            @info "✅ Successfully created benchmark results issue: $(issue_result.html_url)"
            @info "🔗 Your benchmark data has been shared with the LinearSolve.jl community!"
            @info "💡 View all community benchmark data: https://github.com/SciML/LinearSolve.jl/issues?q=is%3Aissue+label%3Abenchmark-data"
        else
            error("Failed to create GitHub issue")
        end

    catch e
        @warn "❌ Failed to create GitHub issue: $e"
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
    create_benchmark_issue(target_repo, content, auth, system_info)

Create a GitHub issue with benchmark results using GitHub.create_issue().
"""
function create_benchmark_issue(target_repo, content, auth, system_info)
    try
        # Get repository object
        repo_obj = GitHub.repo(target_repo; auth=auth)
        
        # Create issue title and body
        cpu_name = get(system_info, "cpu_name", "unknown")
        os_name = get(system_info, "os", "unknown")
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")
        
        issue_title = "Benchmark Results: $cpu_name on $os_name ($timestamp)"
        
        issue_body = """
# LinearSolve.jl Autotune Benchmark Results

$content

---

## System Summary
- **CPU:** $cpu_name
- **OS:** $os_name  
- **Timestamp:** $timestamp

🤖 *Generated automatically by LinearSolve.jl autotune system*
"""
        
        # Create the issue with labels
        issue_result = GitHub.create_issue(
            repo_obj;
            title=issue_title,
            body=issue_body,
            labels=["benchmark-data"],
            auth=auth
        )
        
        @info "✅ Created benchmark results issue #$(issue_result.number)"
        return issue_result
        
    catch e
        @warn "Failed to create benchmark issue: $e"
        return nothing
    end
end

