using Pkg
using SafeTestsets
using SciMLTesting
const LONGER_TESTS = false

const GROUP = current_group(default = "All")

const HAS_EXTENSIONS = true

# Note on the dependency direction: LinearSolve is an "inverted-leaf" monorepo.
# The root LinearSolve package does NOT depend on its lib/* sublibraries;
# instead each sublibrary (LinearSolveAutotune, LinearSolvePyAMG) depends on the
# root LinearSolve via its own [sources] entry. That is why the root Project.toml
# has no [sources] section pointing at lib/* (the sublibs are not root deps).
lib_dir = joinpath(dirname(@__DIR__), "lib")

# GROUP can be a bare sublibrary name (Core test group) or
# "{sublibrary}_{TEST_GROUP}" for any custom group (e.g., QA, etc.).
const base_group, test_group = detect_sublibrary_group(GROUP, lib_dir)

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
    run_tests(;
        default = "All",
        core = function ()
            @time @safetestset "Basic Tests" include("Core/basictests.jl")
            @time @safetestset "Batched RHS" include("Core/batch.jl")
            @time @safetestset "GESV Factorization" include("Core/gesv.jl")
            @time @safetestset "LU Refactorization Reuse" include("Core/lu_refactorization.jl")
            @time @safetestset "Return codes" include("Core/retcodes.jl")
            @time @safetestset "Re-solve" include("Core/resolve.jl")
            @time @safetestset "Zero Initialization Tests" include("Core/zeroinittests.jl")
            @time @safetestset "Non-Square Tests" include("Core/nonsquare.jl")
            @time @safetestset "SparseVector b Tests" include("Core/sparse_vector.jl")
            @time @safetestset "Nonstructural Zeros" include("Core/nonstructural_zeros.jl")
            @time @safetestset "Default Alg Tests" include("Core/default_algs.jl")
            @time @safetestset "FixedSizeArrays" include("Core/fixedsizearrays.jl")
            @time @safetestset "ComponentArrays" include("Core/componentarrays.jl")
            @time @safetestset "Adjoint Sensitivity" include("Core/adjoint.jl")
            @time @safetestset "ForwardDiff Overloads" include("Core/forwarddiff_overloads.jl")
            @time @safetestset "Traits" include("Core/traits.jl")
            @time @safetestset "Verbosity" include("Core/verbosity.jl")
            @time @safetestset "BandedMatrices" include("Core/banded.jl")
            @time @safetestset "Butterfly Factorization" include("Core/butterfly.jl")
            @time @safetestset "Mixed Precision" include("Core/test_mixed_precision.jl")
            @time @safetestset "Resize" include("Core/resize.jl")
            return @time @safetestset "SpecializingFactorizations" include("Core/specializing_factorizations.jl")
        end,
        groups = Dict(
            # STRUMPACK runs in the base env: STRUMPACK_jll is a base test dep (the
            # Core suite also probes the STRUMPACK extension), so this group adds no
            # deps.
            "LinearSolveSTRUMPACK" => function ()
                return @time @safetestset "LinearSolveSTRUMPACK" include("LinearSolveSTRUMPACK/strumpack.jl")
            end,
            "DefaultsLoading" => function ()
                return @time @safetestset "Defaults Loading Tests" include("DefaultsLoading/defaults_loading.jl")
            end,
            "Preferences" => function ()
                return @time @safetestset "Dual Preference System Integration" include("Preferences/preferences.jl")
            end,
            "LinearSolvePureUMFPACK" => function ()
                return @time @safetestset "PureUMFPACK" include("LinearSolvePureUMFPACK/pureumfpack.jl")
            end,
            # The dep-adding groups below activate their sub-environment INSIDE the
            # platform/version guard, so on a platform/version where the group does
            # not run, the env is never activated or instantiated (matching the old
            # `if Base.Sys.islinux() && GROUP == ...` dispatch, which gated the
            # activation as well as the test body). GPU, Pardiso, and HSL have no such
            # guard in the old dispatch, so they activate unconditionally via `env =`.

            # Don't run Enzyme tests on prerelease or Julia >= 1.12 (Enzyme
            # compatibility issues). See:
            # https://github.com/SciML/LinearSolve.jl/issues/817
            "AD" => function ()
                if isempty(VERSION.prerelease)
                    activate_group_env(joinpath(@__DIR__, "AD"))
                    @time @safetestset "Mooncake Derivative Rules" include("AD/mooncake.jl")
                    @time @safetestset "Static Arrays" include("AD/static_arrays.jl")
                    @time @safetestset "Caching Allocation Tests" include("AD/caching_allocation_tests.jl")
                    # Disable Enzyme tests on Julia >= 1.12 due to compatibility issues
                    if VERSION < v"1.12.0-"
                        @time @safetestset "Enzyme Derivative Rules" include("AD/enzyme.jl")
                    end
                end
                return nothing
            end,
            # ParU_jll requires Julia >= 1.12 (SuiteSparse_jll in older stdlib is
            # incompatible)
            "LinearSolveParU" => function ()
                if VERSION >= v"1.12.0-"
                    activate_group_env(joinpath(@__DIR__, "LinearSolveParU"))
                    @time @safetestset "ParU" include("LinearSolveParU/paru.jl")
                end
                return nothing
            end,
            # GPU is a dep-adding group on a self-hosted CUDA runner (folds the
            # former bespoke GPU.yml workflow). LinearSolveCUDA kept as an alias.
            "GPU" => (;
                env = joinpath(@__DIR__, "GPU"), body = function ()
                    return @time @safetestset "CUDA" include("GPU/cuda.jl")
                end
            ),
            "LinearSolvePardiso" => (;
                env = joinpath(@__DIR__, "LinearSolvePardiso"), body = function ()
                    return @time @safetestset "Pardiso" include("LinearSolvePardiso/pardiso.jl")
                end
            ),
            "LinearSolveHSL" => (;
                env = joinpath(@__DIR__, "LinearSolveHSL"), body = function ()
                    return @time @safetestset "HSL" include("LinearSolveHSL/hsl.jl")
                end
            ),
            "LinearSolveMUMPS" => function ()
                if Base.Sys.islinux()
                    activate_group_env(joinpath(@__DIR__, "LinearSolveMUMPS"))
                    @time @safetestset "MUMPS" include("LinearSolveMUMPS/mumps.jl")
                end
                return nothing
            end,
            "LinearSolveGinkgo" => function ()
                if !Base.Sys.iswindows()
                    activate_group_env(joinpath(@__DIR__, "LinearSolveGinkgo"))
                    @time @safetestset "Ginkgo" include("LinearSolveGinkgo/ginkgo.jl")
                end
                return nothing
            end,
            "LinearSolveElemental" => function ()
                if !Base.Sys.iswindows()
                    activate_group_env(joinpath(@__DIR__, "LinearSolveElemental"))
                    @time @safetestset "Elemental" include("LinearSolveElemental/elemental.jl")
                end
                return nothing
            end,
            "LinearSolveHYPRE" => function ()
                if Base.Sys.islinux() && HAS_EXTENSIONS
                    activate_group_env(joinpath(@__DIR__, "LinearSolveHYPRE"))
                    @time @safetestset "LinearSolveHYPRE" include("LinearSolveHYPRE/hypretests.jl")
                end
                return nothing
            end,
            "LinearSolvePartitionedSolvers" => function ()
                if Base.Sys.islinux() && HAS_EXTENSIONS
                    activate_group_env(joinpath(@__DIR__, "LinearSolvePartitionedSolvers"))
                    @time @safetestset "LinearSolvePartitionedSolvers" include(
                        "LinearSolvePartitionedSolvers/partitionedsolverstests.jl"
                    )
                    @time @safetestset "LinearSolvePartitionedSolversMPI" include(
                        "LinearSolvePartitionedSolvers/partitionedsolverstests_mpi.jl"
                    )
                end
                return nothing
            end,
            "LinearSolvePETSc" => function ()
                if Base.Sys.islinux() && HAS_EXTENSIONS
                    activate_group_env(joinpath(@__DIR__, "LinearSolvePETSc"))
                    @time @safetestset "LinearSolvePETSc" include("LinearSolvePETSc/petsctests.jl")
                    @time @safetestset "LinearSolvePETScMPI" include("LinearSolvePETSc/petsctests_mpi.jl")
                end
                return nothing
            end,
            "LinearSolveSuperLUDIST" => function ()
                if Base.Sys.islinux() && HAS_EXTENSIONS
                    activate_group_env(joinpath(@__DIR__, "LinearSolveSuperLUDIST"))
                    @time @safetestset "LinearSolveSuperLUDIST" include("LinearSolveSuperLUDIST/superludist.jl")
                end
                return nothing
            end,
            "Trim" => function ()
                if VERSION >= v"1.12.0"
                    activate_group_env(joinpath(@__DIR__, "Trim"))
                    @time @safetestset "Trim Tests" include("Trim/runtests.jl")
                end
                return nothing
            end,
        ),
        # Quality Assurance (Aqua, ExplicitImports) — dep-adding group whose tooling
        # deps stay out of the main test target (test/qa). The prerelease guard gates
        # the activation too, so on a prerelease the qa env is never instantiated.
        qa = function ()
            if isempty(VERSION.prerelease)
                activate_group_env(joinpath(@__DIR__, "qa"))
                @time @safetestset "Quality Assurance" include("qa/qa.jl")
                @time @safetestset "JET Tests" include("qa/jet.jl")
            end
            return nothing
        end,
        # LinearSolveCUDA is an alias for the GPU group.
        umbrellas = Dict("LinearSolveCUDA" => ["GPU"]),
        # Curated All: the base-environment groups only (matching test_groups.toml's
        # "All runs every base-env group"). Dep-adding groups are excluded from All
        # and selectable by name.
        all = ["Core", "LinearSolveSTRUMPACK"],
        sublib_env = "LINEARSOLVE_TEST_GROUP",
        lib_dir = lib_dir,
    )
end
