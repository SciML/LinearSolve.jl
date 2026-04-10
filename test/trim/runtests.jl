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
    @test isfile(JULIAC)

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
        cmd = `$(Base.julia_cmd()) --project=. --depwarn=error $(JULIAC) --experimental --trim=unsafe-warn --output-exe $(binpath) $(mainfile)`

        # since we are calling Julia from Julia, we first need to clean some
        # environment variables
        clean_env = copy(ENV)
        delete!(clean_env, "JULIA_PROJECT")
        delete!(clean_env, "JULIA_LOAD_PATH")
        # We could just check for success, but then failures are hard to debug.
        # Instead we use `_execute` to also capture `stdout` and `stderr`.
        # @test success(setenv(cmd, clean_env))
        trimcall = _execute(setenv(cmd, clean_env; dir = @__DIR__))
        if trimcall.exitcode != 0 && expectedtopass
            @show trimcall.stdout
            @show trimcall.stderr
        end
        @test trimcall.exitcode == 0 broken = !expectedtopass
        @test isfile(binpath) broken = !expectedtopass
        @test success(`$(binpath) 1.0`) broken = !expectedtopass
    end
end
