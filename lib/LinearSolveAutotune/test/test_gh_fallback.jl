using Test
using LinearSolveAutotune
using gh_cli_jll

@testset "gh CLI fallback tests" begin
    # Test get_gh_command function
    @testset "get_gh_command" begin
        gh_cmd = LinearSolveAutotune.get_gh_command()
        @test gh_cmd isa Cmd

        # Test that the command can be executed
        @test_nowarn begin
            version = read(`$gh_cmd version`, String)
            @test !isempty(version)
            @test occursin("gh version", version)
        end
    end

    # Test JLL-provided gh directly
    @testset "JLL gh" begin
        jll_gh_cmd = `$(gh_cli_jll.gh())`
        @test jll_gh_cmd isa Cmd

        # Test that JLL gh works
        @test_nowarn begin
            version = read(`$jll_gh_cmd version`, String)
            @test !isempty(version)
            @test occursin("gh version", version)
        end
    end

    # Test authentication setup (may fail if not authenticated)
    @testset "Authentication setup" begin
        auth_result = LinearSolveAutotune.setup_github_authentication()
        @test auth_result isa Tuple
        @test length(auth_result) == 2
        # We don't require authentication to succeed, just that the function works
    end
end

println("✅ All gh fallback tests passed!")
