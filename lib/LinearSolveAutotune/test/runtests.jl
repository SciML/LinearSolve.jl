using SafeTestsets

const TEST_GROUP = get(ENV, "LINEARSOLVE_TEST_GROUP", "ALL")

if TEST_GROUP == "Core" || TEST_GROUP == "ALL"
    @time @safetestset "LinearSolveAutotune Tests" include("autotune_tests.jl")
end
