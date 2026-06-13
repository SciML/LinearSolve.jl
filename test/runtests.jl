using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")

const HAS_EXTENSIONS = true

# Activate a dep-adding root test group's sub-environment (test/<group>). Each
# such group carries its extra dependencies in test/<group>/Project.toml and is
# excluded from the `All` run (which executes only the base-environment groups).
# Pkg.develop the root LinearSolve so the group runs against the PR branch code
# (the [sources] entry covers this on Julia >= 1.11; Pkg.develop keeps it working
# on the 1.10 floor where [sources] is ignored).
function activate_group_env(group)
    Pkg.activate(joinpath(@__DIR__, group))
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

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
    # Base-environment groups run in the package's main test env and are part of
    # `All`. Each test file lives in test/<group>/.
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Basic Tests" include("core/basictests.jl")
        @time @safetestset "Return codes" include("core/retcodes.jl")
        @time @safetestset "Re-solve" include("core/resolve.jl")
        @time @safetestset "Zero Initialization Tests" include("core/zeroinittests.jl")
        @time @safetestset "Non-Square Tests" include("core/nonsquare.jl")
        @time @safetestset "SparseVector b Tests" include("core/sparse_vector.jl")
        @time @safetestset "Nonstructural Zeros" include("core/nonstructural_zeros.jl")
        @time @safetestset "Default Alg Tests" include("core/default_algs.jl")
        @time @safetestset "Adjoint Sensitivity" include("core/adjoint.jl")
        @time @safetestset "ForwardDiff Overloads" include("core/forwarddiff_overloads.jl")
        @time @safetestset "Traits" include("core/traits.jl")
        @time @safetestset "Verbosity" include("core/verbosity.jl")
        @time @safetestset "BandedMatrices" include("core/banded.jl")
        @time @safetestset "Butterfly Factorization" include("core/butterfly.jl")
        @time @safetestset "Mixed Precision" include("core/test_mixed_precision.jl")
        @time @safetestset "Resize" include("core/resize.jl")
        @time @safetestset "SpecializingFactorizations" include("core/specializing_factorizations.jl")
    end

    # STRUMPACK runs in the base env: STRUMPACK_jll is a base test dep (the Core
    # suite also probes the STRUMPACK extension), so this group adds no deps.
    if GROUP == "All" || GROUP == "LinearSolveSTRUMPACK"
        @time @safetestset "LinearSolveSTRUMPACK" include("strumpack/strumpack.jl")
    end

    if GROUP == "DefaultsLoading"
        @time @safetestset "Defaults Loading Tests" include("defaultsloading/defaults_loading.jl")
    end

    if GROUP == "Preferences"
        @time @safetestset "Dual Preference System Integration" include("preferences/preferences.jl")
    end

    # Quality Assurance (Aqua, ExplicitImports, JET) — dep-adding group whose
    # tooling deps stay out of the main test target (test/qa).
    if GROUP == "QA" && isempty(VERSION.prerelease)
        activate_group_env("qa")
        @time @safetestset "Quality Assurance" include("qa/qa.jl")
        @time @safetestset "JET Tests" include("qa/jet.jl")
    end

    # Dep-adding groups below: each activates test/<group>/Project.toml and is
    # excluded from the `All` run.

    # AD/allocation tests (Enzyme, Mooncake, AllocCheck, StaticArrays); the AD
    # stack is not compatible with prerelease Julia.
    # Don't run Enzyme tests on Julia >= 1.12 (Enzyme compatibility issues)
    # See: https://github.com/SciML/LinearSolve.jl/issues/817
    if GROUP == "AD" && isempty(VERSION.prerelease)
        activate_group_env("AD")
        @time @safetestset "Mooncake Derivative Rules" include("AD/mooncake.jl")
        @time @safetestset "Static Arrays" include("AD/static_arrays.jl")
        @time @safetestset "Caching Allocation Tests" include("AD/caching_allocation_tests.jl")
        # Disable Enzyme tests on Julia >= 1.12 due to compatibility issues
        if VERSION < v"1.12.0-"
            @time @safetestset "Enzyme Derivative Rules" include("AD/enzyme.jl")
        end
    end

    if GROUP == "LinearSolvePureUMFPACK"
        @time @safetestset "PureUMFPACK" include("pureumfpack.jl")
    end

    # ParU_jll requires Julia >= 1.12 (SuiteSparse_jll in older stdlib is incompatible)
    if GROUP == "LinearSolveParU" && VERSION >= v"1.12.0-"
        activate_group_env("paru")
        @time @safetestset "ParU" include("paru/paru.jl")
    end

    # GPU is a dep-adding group on a self-hosted CUDA runner (folds the former
    # bespoke GPU.yml workflow). LinearSolveCUDA kept as an alias.
    if GROUP == "GPU" || GROUP == "LinearSolveCUDA"
        activate_group_env("gpu")
        @time @safetestset "CUDA" include("gpu/cuda.jl")
    end

    if GROUP == "LinearSolvePardiso"
        activate_group_env("pardiso")
        @time @safetestset "Pardiso" include("pardiso/pardiso.jl")
    end

    if GROUP == "LinearSolveHSL"
        activate_group_env("hsl")
        @time @safetestset "HSL" include("hsl/hsl.jl")
    end

    if Base.Sys.islinux() && GROUP == "LinearSolveMUMPS"
        activate_group_env("mumps")
        @time @safetestset "MUMPS" include("mumps/mumps.jl")
    end

    if !Base.Sys.iswindows() && GROUP == "LinearSolveGinkgo"
        activate_group_env("ginkgo")
        @time @safetestset "Ginkgo" include("ginkgo/ginkgo.jl")
    end

    if !Base.Sys.iswindows() && GROUP == "LinearSolveElemental"
        activate_group_env("elemental")
        @time @safetestset "Elemental" include("elemental/elemental.jl")
    end

    if Base.Sys.islinux() && GROUP == "LinearSolveHYPRE" && HAS_EXTENSIONS
        activate_group_env("hypre")
        @time @safetestset "LinearSolveHYPRE" include("hypre/hypretests.jl")
    end

    if Base.Sys.islinux() && GROUP == "LinearSolvePartitionedSolvers" && HAS_EXTENSIONS
        activate_group_env("partitionedsolvers")
        @time @safetestset "LinearSolvePartitionedSolvers" include(
            "partitionedsolvers/partitionedsolverstests.jl"
        )
        @time @safetestset "LinearSolvePartitionedSolversMPI" include(
            "partitionedsolvers/partitionedsolverstests_mpi.jl"
        )
    end

    if Base.Sys.islinux() && GROUP == "LinearSolvePETSc" && HAS_EXTENSIONS
        activate_group_env("petsc")
        @time @safetestset "LinearSolvePETSc" include("petsc/petsctests.jl")
        @time @safetestset "LinearSolvePETScMPI" include("petsc/petsctests_mpi.jl")
    end

    if GROUP == "Trim" && VERSION >= v"1.12.0"
        activate_group_env("trim")
        @time @safetestset "Trim Tests" include("trim/runtests.jl")
    end
end
