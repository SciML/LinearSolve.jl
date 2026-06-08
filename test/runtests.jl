using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")

const HAS_EXTENSIONS = true

# Detect sublibrary test groups.
# GROUP can be a bare sublibrary name (Core test group) or
# "{sublibrary}_{TEST_GROUP}" for any custom group (e.g., QA, etc.).
# Sublibraries declare their groups in test/test_groups.toml.
#
# Note on the dependency direction: LinearSolve is an "inverted-leaf" monorepo.
# The root LinearSolve package does NOT depend on its lib/* sublibraries;
# instead each sublibrary (LinearSolveAutotune, LinearSolvePyAMG) depends on the
# root LinearSolve via its own [sources] entry. That is why the root Project.toml
# has no [sources] section pointing at lib/* (the sublibs are not root deps).
lib_dir = joinpath(dirname(@__DIR__), "lib")

# Check if GROUP matches a sublibrary, possibly with a _SUFFIX for the test group.
# Scan underscores right-to-left to find the longest matching sublibrary prefix.
function _detect_sublibrary_group(group, lib_dir)
    isdir(joinpath(lib_dir, group)) && return (group, "Core")
    for i in length(group):-1:1
        if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
            return (group[1:(i - 1)], group[(i + 1):end])
        end
    end
    return (group, "Core")
end
const base_group, test_group = _detect_sublibrary_group(GROUP, lib_dir)

if isdir(joinpath(lib_dir, base_group))
    Pkg.activate(joinpath(lib_dir, base_group))
    # On Julia < 1.11, the [sources] section in Project.toml is not supported.
    # Manually Pkg.develop local path dependencies so CI tests the PR branch code.
    # We resolve transitively: each developed dependency's own [sources] are also
    # developed.
    if VERSION < v"1.11.0-DEV.0"
        developed = Set{String}()
        # Never develop the active project: when sublibraries cyclically
        # reference each other via [sources], the transitive walk below would
        # otherwise try to `Pkg.develop` the active project itself, which Pkg
        # refuses.
        push!(developed, normpath(joinpath(lib_dir, base_group)))
        specs = Pkg.PackageSpec[]
        queue = [joinpath(lib_dir, base_group)]
        while !isempty(queue)
            pkg_dir = popfirst!(queue)
            toml_path = joinpath(pkg_dir, "Project.toml")
            isfile(toml_path) || continue
            toml = Pkg.TOML.parsefile(toml_path)
            if haskey(toml, "sources")
                for (dep_name, source_spec) in toml["sources"]
                    if source_spec isa Dict && haskey(source_spec, "path")
                        dep_path = normpath(joinpath(pkg_dir, source_spec["path"]))
                        if isdir(dep_path) && !(dep_path in developed)
                            push!(developed, dep_path)
                            @info "Queuing local source dependency" dep_name dep_path
                            push!(specs, Pkg.PackageSpec(path = dep_path))
                            push!(queue, dep_path)
                        end
                    end
                end
            end
        end
        isempty(specs) || Pkg.develop(specs)
    end
    withenv("LINEARSOLVE_TEST_GROUP" => test_group) do
        Pkg.test(base_group, julia_args = ["--check-bounds=auto", "--compiled-modules=yes", "--depwarn=yes"], force_latest_compatible_version = false, allow_reresolve = true)
    end
else
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests" include("basictests.jl")
        @time @safetestset "Return codes" include("retcodes.jl")
        @time @safetestset "Re-solve" include("resolve.jl")
        @time @safetestset "Zero Initialization Tests" include("zeroinittests.jl")
        @time @safetestset "Non-Square Tests" include("nonsquare.jl")
        @time @safetestset "SparseVector b Tests" include("sparse_vector.jl")
        @time @safetestset "Nonstructural Zeros" include("nonstructural_zeros.jl")
        @time @safetestset "Default Alg Tests" include("default_algs.jl")
        @time @safetestset "Adjoint Sensitivity" include("adjoint.jl")
        @time @safetestset "ForwardDiff Overloads" include("forwarddiff_overloads.jl")
        @time @safetestset "Traits" include("traits.jl")
        @time @safetestset "Verbosity" include("verbosity.jl")
        @time @safetestset "BandedMatrices" include("banded.jl")
        @time @safetestset "Butterfly Factorization" include("butterfly.jl")
        @time @safetestset "Mixed Precision" include("test_mixed_precision.jl")
        @time @safetestset "Resize" include("resize.jl")
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

    if GROUP == "All" || GROUP == "LinearSolveSTRUMPACK"
        @time @safetestset "LinearSolveSTRUMPACK" include("strumpack/strumpack.jl")
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

    if Base.Sys.islinux() && GROUP == "LinearSolveMUMPS"
        Pkg.activate("mumps")
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @time @safetestset "MUMPS" include("mumps/mumps.jl")
    end

    if !Base.Sys.iswindows() && GROUP == "LinearSolveGinkgo"
        Pkg.activate("ginkgo")
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @time @safetestset "Ginkgo" include("ginkgo/ginkgo.jl")
    end

    if !Base.Sys.iswindows() && GROUP == "LinearSolveElemental"
        Pkg.activate("elemental")
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @time @safetestset "Elemental" include("elemental/elemental.jl")
    end

    if Base.Sys.islinux() && (GROUP == "All" || GROUP == "LinearSolveHYPRE") && HAS_EXTENSIONS
        @time @safetestset "LinearSolveHYPRE" include("hypretests.jl")
    end

    if Base.Sys.islinux() && (GROUP == "All" || GROUP == "LinearSolvePartitionedSolvers") &&
            HAS_EXTENSIONS
        @time @safetestset "LinearSolvePartitionedSolvers" include("partitionedsolverstests.jl")
        @time @safetestset "LinearSolvePartitionedSolversMPI" include(
            "partitionedsolverstests_mpi.jl"
        )
    end

    if Base.Sys.islinux() && GROUP == "LinearSolvePETSc" && HAS_EXTENSIONS
        Pkg.activate("petsc")
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @time @safetestset "LinearSolvePETSc" include("petsctests.jl")
        @time @safetestset "LinearSolvePETScMPI" include("petsctests_mpi.jl")
    end

    if GROUP == "Trim" && VERSION >= v"1.12.0"
        Pkg.activate("trim")
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @time @safetestset "Trim Tests" include("trim/runtests.jl")
    end
end
