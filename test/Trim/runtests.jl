using SafeTestsets

@safetestset "LUFactorization implementation" begin
    using LinearAlgebra
    include("linear_lu.jl")
    sol = TestLUFactorization.solve_linear(1.0)
    @test sol.u ≈ [0.09090909090909091, 0.6363636363636364]
end

@safetestset "MKLLUFactorization implementation" begin
    using LinearAlgebra
    using LinearSolve
    # Only test if MKL is available
    if LinearSolve.usemkl
        include("linear_mkl.jl")
        sol = TestMKLLUFactorization.solve_linear(1.0)
        @test sol.u ≈ [0.09090909090909091, 0.6363636363636364]
    else
        @test_skip "MKL not available"
    end
end

@safetestset "RFLUFactorization implementation" begin
    using LinearAlgebra
    include("linear_rf.jl")
    sol = TestRFLUFactorization.solve_linear(1.0)
    @test sol.u ≈ [0.09090909090909091, 0.6363636363636364]
end

@safetestset "Run trim" begin
    # https://discourse.julialang.org/t/capture-stdout-and-stderr-in-case-a-command-fails/101772/3?u=romeov
    """
    Run a Cmd object, returning the stdout & stderr contents plus the exit code
    """
    function _execute(cmd::Cmd)
        out = Pipe()
        err = Pipe()
        process = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
        close(out.in)
        close(err.in)
        out = (
            stdout = String(read(out)), stderr = String(read(err)),
            exitcode = process.exitcode,
        )
        return out
    end

    JULIAC = normpath(
        joinpath(
            Sys.BINDIR, Base.DATAROOTDIR, "julia", "juliac",
            "juliac.jl"
        )
    )
    # Julia 1.13 removed `juliac.jl` from the distribution; juliac now lives
    # in the JuliaC package (a test dep of this project).
    JULIAC_CMD = isfile(JULIAC) ? `$(JULIAC)` :
        Cmd(["-e", "using JuliaC; JuliaC.main(ARGS)", "--"])
    @test isfile(JULIAC) || VERSION ≥ v"1.13-"

    using LinearSolve
    # Build list of tests to run, conditionally including MKL
    test_files = [
        ("main_lu.jl", true),
        ("main_rf.jl", true),
    ]
    if LinearSolve.usemkl
        push!(test_files, ("main_mkl.jl", true))
    end

    for (mainfile, expectedtopass) in test_files
        binpath = tempname()
        # JuliaC requires `--output-exe` to be a bare name, so run from the
        # output directory and pass absolute paths for project and entry file.
        # The project can't be this directory nor the active env: SciMLTesting's
        # `activate_group_env` sandboxes the env (instantiated TOMLs, no `src/`),
        # while this repo dir has the sources but no Manifest. JuliaC copies the
        # project dir for its buildscript, so assemble one that has both —
        # sources from here, instantiated Project/Manifest from the active env.
        project_dir = mktempdir()
        for f in readdir(@__DIR__)
            f in ("Project.toml", "Manifest.toml") ||
                cp(joinpath(@__DIR__, f), joinpath(project_dir, f))
        end
        active_env = dirname(Base.active_project())
        cp(joinpath(active_env, "Project.toml"), joinpath(project_dir, "Project.toml"))
        manifest = joinpath(active_env, "Manifest.toml")
        isfile(manifest) && cp(manifest, joinpath(project_dir, "Manifest.toml"))
        cmd = `$(Base.julia_cmd()) --project=$(project_dir) --depwarn=error $(JULIAC_CMD) --experimental --trim=unsafe-warn --output-exe $(basename(binpath)) $(joinpath(@__DIR__, mainfile))`

        # since we are calling Julia from Julia, we first need to clean some
        # environment variables
        clean_env = copy(ENV)
        delete!(clean_env, "JULIA_PROJECT")
        delete!(clean_env, "JULIA_LOAD_PATH")
        # We could just check for success, but then failures are hard to debug.
        # Instead we use `_execute` to also capture `stdout` and `stderr`.
        # @test success(setenv(cmd, clean_env))
        trimcall = _execute(setenv(cmd, clean_env; dir = dirname(binpath)))
        if trimcall.exitcode != 0 && expectedtopass
            @show trimcall.stdout
            @show trimcall.stderr
        end
        @test trimcall.exitcode == 0 broken = !expectedtopass
        @test isfile(binpath) broken = !expectedtopass
        @test success(`$(binpath) 1.0`) broken = !expectedtopass
    end
end
