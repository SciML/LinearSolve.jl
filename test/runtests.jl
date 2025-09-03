using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")

const HAS_EXTENSIONS = isdefined(Base, :get_extension)

if GROUP == "All" || GROUP == "Core"
    @time @safetestset "Basic Tests" include("basictests.jl")
    @time @safetestset "Return codes" include("retcodes.jl")
    @time @safetestset "Re-solve" include("resolve.jl")
    @time @safetestset "Zero Initialization Tests" include("zeroinittests.jl")
    @time @safetestset "Non-Square Tests" include("nonsquare.jl")
    @time @safetestset "SparseVector b Tests" include("sparse_vector.jl")
    @time @safetestset "Default Alg Tests" include("default_algs.jl")
    @time @safetestset "Adjoint Sensitivity" include("adjoint.jl")
    @time @safetestset "ForwardDiff Overloads" include("forwarddiff_overloads.jl")
    @time @safetestset "Traits" include("traits.jl")
    @time @safetestset "BandedMatrices" include("banded.jl")
    @time @safetestset "Issue #771: AutoTune Failures" include("test_issue_771_autotune_failures.jl")
end

# Don't run Enzyme tests on prerelease
if GROUP == "All" || GROUP == "NoPre" && isempty(VERSION.prerelease)
    Pkg.activate("nopre")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @time @safetestset "Quality Assurance" include("qa.jl")
    @time @safetestset "Enzyme Derivative Rules" include("nopre/enzyme.jl")
    @time @safetestset "JET Tests" include("nopre/jet.jl")
    @time @safetestset "Static Arrays" include("nopre/static_arrays.jl")
end

if GROUP == "DefaultsLoading"
    @time @safetestset "Defaults Loading Tests" include("defaults_loading.jl")
end

if GROUP == "LinearSolveCUDA"
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @time @safetestset "CUDA" include("gpu/cuda.jl")
end

if GROUP == "LinearSolvePardiso"
    Pkg.activate("pardiso")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @time @safetestset "Pardiso" include("pardiso/pardiso.jl")
end

if Base.Sys.islinux() && (GROUP == "All" || GROUP == "LinearSolveHYPRE") && HAS_EXTENSIONS
    @time @safetestset "LinearSolveHYPRE" include("hypretests.jl")
end
