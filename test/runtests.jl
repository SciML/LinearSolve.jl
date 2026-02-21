using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")

const HAS_EXTENSIONS = true

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
    @time @safetestset "Verbosity" include("verbosity.jl")
    @time @safetestset "BandedMatrices" include("banded.jl")
    @time @safetestset "Butterfly Factorization" include("butterfly.jl")
    @time @safetestset "Mixed Precision" include("test_mixed_precision.jl")
    # ParU_jll requires Julia >= 1.12 (SuiteSparse_jll in older stdlib is incompatible)
    if VERSION >= v"1.12.0-"
        Pkg.activate("paru")
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @time @safetestset "ParU" include("paru/paru.jl")
        Pkg.activate(".")
    end
end

# Don't run Enzyme tests on prerelease or Julia >= 1.12 (Enzyme compatibility issues)
# See: https://github.com/SciML/LinearSolve.jl/issues/817
if GROUP == "NoPre" && isempty(VERSION.prerelease)
    Pkg.activate("nopre")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @time @safetestset "Quality Assurance" include("qa.jl")
    @time @safetestset "Mooncake Derivative Rules" include("nopre/mooncake.jl")
    @time @safetestset "JET Tests" include("nopre/jet.jl")
    @time @safetestset "Static Arrays" include("nopre/static_arrays.jl")
    @time @safetestset "Caching Allocation Tests" include("nopre/caching_allocation_tests.jl")
    # Disable Enzyme tests on Julia >= 1.12 due to compatibility issues
    if VERSION < v"1.12.0-"
        @time @safetestset "Enzyme Derivative Rules" include("nopre/enzyme.jl")
    end
end

if GROUP == "DefaultsLoading"
    @time @safetestset "Defaults Loading Tests" include("defaults_loading.jl")
end

if GROUP == "LinearSolveAutotune"
    Pkg.activate(joinpath(dirname(@__DIR__), "lib", GROUP))
    Pkg.test(
        GROUP,
        julia_args = ["--check-bounds=auto", "--compiled-modules=yes", "--depwarn=yes"],
        force_latest_compatible_version = false,
        allow_reresolve = true
    )
end

if GROUP == "Preferences"
    @time @safetestset "Dual Preference System Integration" include("preferences.jl")
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

if Base.Sys.islinux() && GROUP == "LinearSolvePETSc" && HAS_EXTENSIONS
    Pkg.activate("petsc")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @time @safetestset "LinearSolvePETSc" include("petsctests.jl")
end

if GROUP == "Trim" && VERSION >= v"1.12.0"
    Pkg.activate("trim")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @time @safetestset "Trim Tests" include("trim/runtests.jl")
end
