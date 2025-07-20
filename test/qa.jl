using LinearSolve, Aqua
using ExplicitImports

@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(LinearSolve)
    Aqua.test_ambiguities(LinearSolve, recursive = false, broken = true)
    Aqua.test_deps_compat(LinearSolve, ignore = [:MKL_jll])
    Aqua.test_piracies(LinearSolve,
        treat_as_own = [LinearProblem])
    Aqua.test_project_extras(LinearSolve)
    Aqua.test_stale_deps(LinearSolve, ignore = [:MKL_jll])
    Aqua.test_unbound_args(LinearSolve)
    Aqua.test_undefined_exports(LinearSolve)
end

@testset "Explicit Imports" begin
    # Get extension modules that might be unanalyzable
    klu_mod = try
        Base.get_extension(LinearSolve, :LinearSolveSparseArraysExt).KLU
    catch
        nothing
    end
    unanalyzable_mods = (LinearSolve.OperatorCondition, LinearSolve.DefaultAlgorithmChoice)
    if klu_mod !== nothing
        unanalyzable_mods = (unanalyzable_mods..., klu_mod)
    end
    
    @test check_no_implicit_imports(LinearSolve; skip = (Base, Core), 
        allow_unanalyzable = unanalyzable_mods) === nothing
    @test check_no_stale_explicit_imports(LinearSolve; allow_unanalyzable = unanalyzable_mods) === nothing
    @test check_all_qualified_accesses_via_owners(LinearSolve) === nothing
end
