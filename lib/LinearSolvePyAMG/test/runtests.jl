using Pkg
using SafeTestsets

const TEST_GROUP = get(ENV, "LINEARSOLVE_TEST_GROUP", "All")

# QA tooling (Aqua/JET) lives in an isolated sub-environment under test/qa so
# its compat bounds don't constrain the main test resolve. Develop the in-repo
# path deps so [sources] also works on Julia < 1.11 (where the Project.toml
# [sources] table is ignored), then instantiate.
function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop(
            [
                Pkg.PackageSpec(path = joinpath(@__DIR__, "..")),
                Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", ".."))
            ]
        )
    end
    return Pkg.instantiate()
end

if TEST_GROUP == "Core" || TEST_GROUP == "All"
    @time @safetestset "LinearSolvePyAMG Tests" include("pyamg_tests.jl")
end

# QA (Aqua/JET) is a dep-adding group: it runs in its own isolated sub-env
# under test/qa (excluded from the Core/All run).
if TEST_GROUP == "QA"
    activate_qa_env()
    @safetestset "Code quality (Aqua + JET)" include("qa/qa.jl")
end
