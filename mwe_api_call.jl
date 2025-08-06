#!/usr/bin/env julia

# MWE: Exact reproduction of LinearSolveAutotune GitHub API call
# Usage: GITHUB_TOKEN=your_token julia --project=lib/LinearSolveAutotune mwe_api_call.jl

using GitHub
using Dates

function mwe_github_api_call()
    println("🧪 MWE: LinearSolveAutotune GitHub API Call")
    println("="^50)
    
    # Check for token (exactly as package does)
    token = get(ENV, "GITHUB_TOKEN", nothing)
    if token === nothing
        println("❌ No GITHUB_TOKEN found")
        return false
    end
    
    println("✅ Token found (length: $(length(token)))")
    
    try
        # Step 1: Authenticate (exactly as package does)
        println("📋 Step 1: GitHub.authenticate(token)")
        auth = GitHub.authenticate(token)
        println("✅ Authentication successful")
        println("   Auth type: $(typeof(auth))")
        
        # Step 2: Get repository object (exactly as package does)
        println("\n📋 Step 2: GitHub.repo(target_repo; auth=auth)")
        target_repo = "SciML/LinearSolve.jl"
        repo_obj = GitHub.repo(target_repo; auth=auth)
        println("✅ Repository access successful")
        println("   Repo: $(repo_obj.full_name)")
        
        # Step 3: Create title and body (exactly as package does)
        println("\n📋 Step 3: Create issue title and body")
        system_info = Dict(
            "cpu_name" => "Test CPU Apple M2 Max",
            "os" => "Test Darwin"
        )
        content = """## Test Benchmark Results

### System Information
- **Julia Version**: $(VERSION)
- **CPU**: $(system_info["cpu_name"])
- **OS**: $(system_info["os"])

### Sample Results
| Algorithm | GFLOPs |
|-----------|--------|
| AppleAccelerateLUFactorization | 72.38 |
| RFLUFactorization | 52.23 |
| LUFactorization | 50.84 |

**This is a test issue - please close it**

---
*Generated automatically by LinearSolveAutotune.jl*"""
        
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
        
        println("✅ Issue content created")
        println("   Title: $(issue_title)")
        println("   Body length: $(length(issue_body)) characters")
        
        # Step 4: Create issue (EXACT reproduction of package call)
        println("\n📋 Step 4: GitHub.create_issue() - EXACT package reproduction")
        println("   Calling: GitHub.create_issue(repo_obj, title=issue_title, body=issue_body, auth=auth)")
        
        issue_result = GitHub.create_issue(
            repo_obj,
            title=issue_title,
            body=issue_body,
            auth=auth
        )
        
        println("🎉 SUCCESS!")
        println("✅ Created issue #$(issue_result.number)")
        println("🔗 URL: $(issue_result.html_url)")
        println()
        println("📝 Please close this test issue: $(issue_result.html_url)")
        
        return true
        
    catch e
        println("❌ ERROR at step above:")
        println("   Error type: $(typeof(e))")
        println("   Error message: $e")
        
        # Additional diagnostic info
        println("\n🔍 Diagnostic Information:")
        println("   Julia version: $(VERSION)")
        println("   GitHub.jl version: $(pkgversion(GitHub))")
        
        # Try to show available methods
        try
            println("   Available GitHub.create_issue methods:")
            for m in methods(GitHub.create_issue)
                println("     $m")
            end
        catch
            println("   Could not list methods")
        end
        
        return false
    end
end

# Also test different syntax variations to find what works
function test_syntax_variations()
    println("\n🔧 Testing Syntax Variations")
    println("="^35)
    
    token = get(ENV, "GITHUB_TOKEN", nothing)
    if token === nothing
        println("❌ No token for syntax tests")
        return
    end
    
    try
        auth = GitHub.authenticate(token)
        repo_obj = GitHub.repo("SciML/LinearSolve.jl"; auth=auth)
        
        title = "SYNTAX TEST: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))"
        body = "Test of different GitHub.create_issue syntax variations.\n\n**Please close this test issue.**"
        
        # Test 1: Package's current syntax
        println("🧪 Syntax 1: GitHub.create_issue(repo_obj, title=title, body=body, auth=auth)")
        try
            issue1 = GitHub.create_issue(repo_obj, title=title, body=body, auth=auth)
            println("✅ Syntax 1 WORKS! Issue #$(issue1.number)")
            return
        catch e1
            println("❌ Syntax 1 failed: $(typeof(e1)) - $e1")
        end
        
        # Test 2: All keyword arguments
        println("\n🧪 Syntax 2: GitHub.create_issue(repo_obj; title=title, body=body, auth=auth)")
        try
            issue2 = GitHub.create_issue(repo_obj; title=title, body=body, auth=auth)
            println("✅ Syntax 2 WORKS! Issue #$(issue2.number)")
            return
        catch e2
            println("❌ Syntax 2 failed: $(typeof(e2)) - $e2")
        end
        
        # Test 3: Positional arguments
        println("\n🧪 Syntax 3: GitHub.create_issue(repo_obj, title, body, auth)")
        try
            issue3 = GitHub.create_issue(repo_obj, title, body, auth)
            println("✅ Syntax 3 WORKS! Issue #$(issue3.number)")
            return
        catch e3
            println("❌ Syntax 3 failed: $(typeof(e3)) - $e3")
        end
        
        # Test 4: Auth first
        println("\n🧪 Syntax 4: GitHub.create_issue(repo_obj, auth=auth, title=title, body=body)")
        try
            issue4 = GitHub.create_issue(repo_obj, auth=auth, title=title, body=body)
            println("✅ Syntax 4 WORKS! Issue #$(issue4.number)")
            return
        catch e4
            println("❌ Syntax 4 failed: $(typeof(e4)) - $e4")
        end
        
        println("\n❌ All syntax variations failed")
        
    catch e
        println("❌ Setup for syntax tests failed: $e")
    end
end

# Run the MWE
println("Starting MWE reproduction of LinearSolveAutotune API call...\n")

success = mwe_github_api_call()

if !success
    test_syntax_variations()
end

println("\n📊 MWE Results:")
if success
    println("✅ The package's API call works correctly")
    println("💡 Issue might be in authentication or permissions")
else
    println("❌ The package's API call fails")
    println("💡 Need to fix the API call syntax in the package")
end