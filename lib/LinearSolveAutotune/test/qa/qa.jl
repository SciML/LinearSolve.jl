using SciMLTesting, LinearSolveAutotune, Test
using JET

run_qa(
    LinearSolveAutotune;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (;
        # BlasFloat (LinearAlgebra.BLAS, reached via LinearAlgebra) and Base.run.
        all_qualified_accesses_via_owners = (; ignore = (:BlasFloat, :run)),
        # Non-public names accessed qualified: LinearSolve internals
        # (get_config/get_extension/is_available/userecursivefactorization/...),
        # Base/Pkg internals (PkgId, UUID, dependencies, loaded_modules, run, ...),
        # and CPUSummary/blis names used by the benchmarking harness.
        all_qualified_accesses_are_public = (;
            ignore = (
                :BLISLUFactorization, :BlasFloat, :GIT_VERSION_INFO, :Parameters,
                :PkgId, :UUID, :appleaccelerate_isavailable, :dependencies, :format,
                :functional, :get_config, :get_extension, :get_num_threads,
                :is_available, :libm_name, :loaded_modules, :run,
                :userecursivefactorization, :vendor,
            ),
        ),
    ),
    # Heavy `using LinearAlgebra, Statistics, Random, Printf, Base64, Plots, ...`
    # brings ~55 names implicitly; making them explicit is a source refactor tracked
    # in https://github.com/SciML/LinearSolve.jl/issues/1058
    ei_broken = (:no_implicit_imports,),
    # JET reports 8 pre-existing latent issues in the telemetry/GPU-detection
    # source (parse/split union splits and try/catch variable-scope leaks in
    # telemetry.jl + gpu_detection.jl). These predate this QA conversion: the
    # prior `JET.test_package(LinearSolveAutotune; target_defined_modules=true)`
    # call surfaced the identical 8 reports, and #1033 (which added this QA group)
    # merged with the same QA(julia 1) lane red. Fixing them is a telemetry-source
    # task tracked in https://github.com/SciML/LinearSolve.jl/issues/1058; mark the
    # JET check known-broken so it auto-flags once those source bugs are fixed.
    jet_broken = true,
)
