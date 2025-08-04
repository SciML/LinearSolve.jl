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
                println("💡 Please verify your token has 'Issues' read permission and try again.")
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

Create a pull request to LinearSolveAutotuneResults.jl with comprehensive benchmark data.
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
    
    @info "📤 Creating pull request to LinearSolveAutotuneResults.jl repository..."

    try
        # Create unique folder name with timestamp and system identifier
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMM")
        cpu_name = get(system_info, "cpu_name", "unknown")
        os_name = get(system_info, "os", "unknown")
        
        # Create short system identifier
        cpu_short = ""
        if contains(lowercase(cpu_name), "intel")
            cpu_short = "intel"
        elseif contains(lowercase(cpu_name), "amd") 
            cpu_short = "amd"
        elseif contains(lowercase(cpu_name), "apple") || contains(lowercase(cpu_name), "m1") || contains(lowercase(cpu_name), "m2")
            cpu_short = "apple"
        else
            cpu_short = "cpu"
        end
        
        os_short = ""
        if contains(lowercase(os_name), "darwin")
            os_short = "macos"
        elseif contains(lowercase(os_name), "linux")
            os_short = "linux"
        elseif contains(lowercase(os_name), "windows")
            os_short = "windows"
        else
            os_short = "os"
        end
        
        folder_name = "$(timestamp)_$(cpu_short)-$(os_short)"
        
        # Fork the repository if needed and create branch
        target_repo = "SciML/LinearSolveAutotuneResults.jl"
        fallback_repo = "ChrisRackauckas-Claude/LinearSolveAutotuneResults.jl"
        branch_name = "autotune-results-$(folder_name)"
        
        @info "📋 Creating result folder: results/$folder_name"
        
        # Generate all the files we need to create
        files_to_create = create_result_files(folder_name, content, plot_files, results_df, system_info, categories)
        
        # Create pull request with all files
        pr_result = create_results_pr(target_repo, fallback_repo, branch_name, folder_name, files_to_create, auth)
        
        if pr_result !== nothing
            @info "✅ Successfully created pull request: $(pr_result["html_url"])"
            @info "🔗 Your benchmark results will help the LinearSolve.jl community once merged!"
            @info "💡 View all community results at: https://github.com/SciML/LinearSolveAutotuneResults.jl"
        else
            error("Failed to create pull request")
        end

    catch e
        @warn "❌ Failed to create pull request: $e"
        @info "💡 This could be due to network issues, repository permissions, or API limits."

        # Save locally as fallback
        timestamp = replace(string(Dates.now()), ":" => "-")
        fallback_folder = "autotune_results_$(timestamp)"
        create_local_result_folder(fallback_folder, content, plot_files, results_df, system_info, categories)
        @info "📁 Results saved locally to $fallback_folder/ as backup"
    end
end

"""
    create_result_files(folder_name, content, plot_files, results_df, system_info, categories)

Create all the files needed for a result folder.
"""
function create_result_files(folder_name, content, plot_files, results_df, system_info, categories)
    files = Dict{String, String}()
    
    # 1. README.md - human readable summary
    files["results/$folder_name/README.md"] = content
    
    # 2. results.csv - benchmark data
    csv_buffer = IOBuffer()
    CSV.write(csv_buffer, results_df)
    files["results/$folder_name/results.csv"] = String(take!(csv_buffer))
    
    # 3. system_info.csv - detailed system information
    system_df = get_detailed_system_info()
    csv_buffer = IOBuffer()
    CSV.write(csv_buffer, system_df)
    files["results/$folder_name/system_info.csv"] = String(take!(csv_buffer))
    
    # 4. Project.toml - capture current package environment
    project_toml = create_project_toml(system_info)
    files["results/$folder_name/Project.toml"] = project_toml
    
    # 5. PNG files - convert plot files to base64 for GitHub API
    if plot_files isa Dict
        for (eltype, (png_file, pdf_file)) in plot_files
            if isfile(png_file)
                png_content = base64encode(read(png_file))
                files["results/$folder_name/benchmark_$(eltype).png"] = png_content
            end
        end
    elseif plot_files isa Tuple
        png_file, pdf_file = plot_files
        if isfile(png_file)
            png_content = base64encode(read(png_file))
            files["results/$folder_name/benchmark.png"] = png_content
        end
    end
    
    return files
end

"""
    create_project_toml(system_info)

Create a Project.toml file capturing the current LinearSolve ecosystem versions.
"""
function create_project_toml(system_info)
    julia_version = string(VERSION)
    
    # Get package versions from the current environment
    pkg_versions = Dict{String, String}()
    
    # Core packages
    pkg_versions["LinearSolve"] = string(pkgversion(LinearSolve))
    pkg_versions["LinearSolveAutotune"] = "0.1.0"  # Current version
    
    # Optional packages if available
    try
        if isdefined(Main, :CUDA) || haskey(Base.loaded_modules, Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
            pkg_versions["CUDA"] = "5.0"  # Approximate current version
        end
    catch; end
    
    try
        if isdefined(Main, :Metal) || haskey(Base.loaded_modules, Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"))
            pkg_versions["Metal"] = "1.0"  # Approximate current version
        end
    catch; end
    
    try
        if get(system_info, "mkl_available", false)
            pkg_versions["MKL"] = "0.6"  # Approximate current version
        end
    catch; end
    
    # Build Project.toml content
    toml_content = """
[deps]
LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
LinearSolveAutotune = "67398393-80e8-4254-b7e4-1b9a36a3c5b6"
RecursiveFactorization = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
"""
    
    if haskey(pkg_versions, "CUDA")
        toml_content *= """CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"\n"""
    end
    
    if haskey(pkg_versions, "Metal")
        toml_content *= """Metal = "dde4c033-4e86-420c-a63e-0dd931031962"\n"""
    end
    
    if haskey(pkg_versions, "MKL")
        toml_content *= """MKL = "33e6dc65-8f57-5167-99aa-e5a354878fb2"\n"""
    end
    
    toml_content *= "\n[compat]\n"
    toml_content *= "julia = \"$(julia_version)\"\n"
    
    # Add version constraints
    for (pkg, version) in pkg_versions
        if pkg == "LinearSolveAutotune"
            toml_content *= "LinearSolveAutotune = \"0.1\"\n"
        elseif pkg == "LinearSolve"
            toml_content *= "LinearSolve = \"$(version)\"\n"
        end
    end
    
    return toml_content
end

"""
    create_results_pr(target_repo, branch_name, folder_name, files, auth)

Create a pull request with the benchmark results.
"""
function create_results_pr(target_repo, fallback_repo, branch_name, folder_name, files, auth)
    try
        @info "🚧 Creating pull request with benchmark results..."
        @info "📋 Target: $target_repo, Branch: $branch_name"
        @info "📋 Files to include: $(length(files)) files"
        
        # Try target repository first, fallback if it doesn't exist
        actual_target_repo = target_repo
        target_repo_obj = nothing
        repo_owner = ""
        repo_name = ""
        
        try
            @info "📋 Trying primary target: $target_repo"
            repo_parts = split(target_repo, "/")
            if length(repo_parts) != 2
                error("Invalid repository format: $target_repo")
            end
            repo_owner, repo_name = repo_parts
            target_repo_obj = GitHub.repo(target_repo, auth=auth)
            @info "✅ Primary target accessible: $target_repo"
        catch e
            @warn "Primary target $target_repo not accessible: $e"
            @info "📋 Using fallback repository: $fallback_repo"
            actual_target_repo = fallback_repo
            
            repo_parts = split(fallback_repo, "/")
            if length(repo_parts) != 2
                error("Invalid fallback repository format: $fallback_repo")
            end
            repo_owner, repo_name = repo_parts
            target_repo_obj = GitHub.repo(fallback_repo, auth=auth)
            @info "✅ Fallback target accessible: $fallback_repo"
        end
        
        # Get authenticated user to determine source repo
        user = GitHub.whoami(auth=auth)
        source_repo = user.login * "/" * repo_name
        
        # Try to get or create a fork
        fork_repo_obj = nothing
        try
            fork_repo_obj = GitHub.repo(source_repo, auth=auth)
            @info "📋 Using existing fork: $source_repo"
        catch
            @info "📋 Creating fork of $target_repo..."
            fork_repo_obj = GitHub.fork(target_repo_obj, auth=auth)
            # Wait a moment for fork to be ready
            sleep(2)
        end
        
        # Get the default branch (usually main)
        default_branch = target_repo_obj.default_branch
        
        # Get the SHA of the default branch
        main_branch_ref = GitHub.reference(fork_repo_obj, "heads/$default_branch", auth=auth)
        base_sha = main_branch_ref.object["sha"]
        
        # Create new branch
        try
            GitHub.create_ref(fork_repo_obj, "refs/heads/$branch_name", base_sha, auth=auth)
            @info "📋 Created new branch: $branch_name"
        catch e
            if contains(string(e), "Reference already exists")
                @info "📋 Branch $branch_name already exists, updating..."
                # Update existing branch to point to main
                GitHub.update_ref(fork_repo_obj, "heads/$branch_name", base_sha, auth=auth)
            else
                rethrow(e)
            end
        end
        
        # Create all files in the repository
        for (file_path, file_content) in files
            try
                # Try to get existing file to update it
                existing_file = nothing
                try
                    existing_file = GitHub.file(fork_repo_obj, file_path, ref=branch_name, auth=auth)
                catch
                    # File doesn't exist, that's fine
                end
                
                commit_message = if existing_file === nothing
                    "Add $(basename(file_path)) for $folder_name"
                else
                    "Update $(basename(file_path)) for $folder_name"
                end
                
                # Create or update the file
                if isa(file_content, String)
                    # Text content
                    GitHub.create_file(fork_repo_obj, file_path, 
                                     message=commit_message,
                                     content=file_content,
                                     branch=branch_name,
                                     sha=existing_file === nothing ? nothing : existing_file.sha,
                                     auth=auth)
                else
                    # Binary content (already base64 encoded)
                    GitHub.create_file(fork_repo_obj, file_path,
                                     message=commit_message, 
                                     content=file_content,
                                     branch=branch_name,
                                     sha=existing_file === nothing ? nothing : existing_file.sha,
                                     auth=auth)
                end
                
                @info "📋 Created/updated: $file_path"
            catch e
                @warn "Failed to create file $file_path: $e"
            end
        end
        
        # Create pull request to the actual accessible repository
        pr_title = "Add benchmark results: $folder_name"
        pr_body = """
# LinearSolve.jl Benchmark Results

Automated submission of benchmark results from the LinearSolve.jl autotune system.

## System Information
- **Repository**: $actual_target_repo
- **Folder**: `results/$folder_name`
- **Files**: $(length(files)) files including CSV data, plots, and system info
- **Timestamp**: $(Dates.now())

## Contents
- `results.csv` - Detailed benchmark performance data
- `system_info.csv` - System and hardware configuration
- `Project.toml` - Package versions used
- `README.md` - Human-readable summary
- `*.png` - Performance visualization plots

## Automated Submission
This PR was automatically created by the LinearSolve.jl autotune system.
The benchmark data will help improve algorithm selection for the community.

🤖 Generated by LinearSolve.jl autotune system
"""
        
        @info "📋 Creating PR to $actual_target_repo from $(user.login):$branch_name to $default_branch"
        pr_result = GitHub.create_pull_request(target_repo_obj, 
                                             title=pr_title,
                                             body=pr_body, 
                                             head="$(user.login):$branch_name",
                                             base=default_branch,
                                             auth=auth)
        
        @info "✅ Successfully created pull request #$(pr_result.number)"
        return pr_result
        
    catch e
        @warn "Failed to create pull request: $e"
        return nothing
    end
end

"""
    create_local_result_folder(folder_name, content, plot_files, results_df, system_info, categories)

Create a local result folder as fallback when GitHub upload fails.
"""
function create_local_result_folder(folder_name, content, plot_files, results_df, system_info, categories)
    # Create folder
    mkpath(folder_name)
    
    # Create all files locally
    files = create_result_files(folder_name, content, plot_files, results_df, system_info, categories)
    
    for (file_path, file_content) in files
        # Adjust path for local creation
        local_path = replace(file_path, "results/" => "")
        local_dir = dirname(local_path)
        
        if !isempty(local_dir) && local_dir != "."
            mkpath(joinpath(folder_name, local_dir))
        end
        
        full_path = joinpath(folder_name, local_path)
        
        if endswith(file_path, ".png")
            # Decode base64 and write binary
            png_data = base64decode(file_content)
            write(full_path, png_data)
        else
            # Write text content
            write(full_path, file_content)
        end
    end
    
    @info "📁 Created local result folder: $folder_name"
end
