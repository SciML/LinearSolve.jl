"""
    LinearVerbosity <: AbstractVerbositySpecifier

Verbosity configuration for LinearSolve.jl solvers, providing fine-grained control over
diagnostic messages, warnings, and errors during linear system solution.

# Fields

## Error Control Group
- `default_lu_fallback`: Messages when falling back to LU factorization from other methods
- `blas_errors`: Critical BLAS errors that stop computation
- `blas_invalid_args`: BLAS errors due to invalid arguments

## Performance Group
- `no_right_preconditioning`: Messages when right preconditioning is not used

## Numerical Group
- `using_IterativeSolvers`: Messages when using the IterativeSolvers.jl package
- `IterativeSolvers_iterations`: Iteration count messages from IterativeSolvers.jl
- `KrylovKit_verbosity`: Verbosity level passed to KrylovKit.jl solvers
- `KrylovJL_verbosity`: Verbosity level passed to Krylov.jl solvers
- `HYPRE_verbosity`: Verbosity level passed to HYPRE solvers
- `pardiso_verbosity`: Verbosity level passed to Pardiso solvers
- `blas_info`: Informational messages from BLAS operations
- `blas_success`: Success messages from BLAS operations
- `condition_number`: Messages related to condition number calculations
- `convergence_failure`: Messages when iterative solvers fail to converge
- `solver_failure`: Messages when solvers fail for reasons other than convergence
- `max_iters`: Messages when iterative solvers reach maximum iterations

# Constructors

    LinearVerbosity(preset::AbstractVerbosityPreset)

Create a `LinearVerbosity` using a preset configuration:
- `SciMLLogging.None()`: All messages disabled
- `SciMLLogging.Minimal()`: Only critical errors and fatal issues
- `SciMLLogging.Standard()`: Balanced verbosity (default)
- `SciMLLogging.Detailed()`: Comprehensive debugging information
- `SciMLLogging.All()`: Maximum verbosity

    LinearVerbosity(; error_control=nothing, performance=nothing, numerical=nothing, kwargs...)

Create a `LinearVerbosity` with group-level or individual field control.

# Examples

```julia
# Use a preset
verbose = LinearVerbosity(SciMLLogging.Standard())

# Set entire groups
verbose = LinearVerbosity(
    error_control = SciMLLogging.WarnLevel(),
    numerical = SciMLLogging.InfoLevel()
)

# Set individual fields
verbose = LinearVerbosity(
    default_lu_fallback = SciMLLogging.InfoLevel(),
    KrylovJL_verbosity = SciMLLogging.CustomLevel(1),
    blas_errors = SciMLLogging.ErrorLevel()
)

# Mix group and individual settings
verbose = LinearVerbosity(
    numerical = SciMLLogging.InfoLevel(),  # Set all numerical to InfoLevel
    blas_errors = SciMLLogging.ErrorLevel()  # Override specific field
)
```
"""
LinearSolve.@concrete struct LinearVerbosity <:
                                      AbstractVerbositySpecifier
    # Error control
    default_lu_fallback
    # Performance
    no_right_preconditioning
    # Numerical
    using_IterativeSolvers
    IterativeSolvers_iterations
    KrylovKit_verbosity
    KrylovJL_verbosity
    HYPRE_verbosity
    pardiso_verbosity
    blas_errors
    blas_invalid_args
    blas_info
    blas_success
    condition_number
    convergence_failure
    solver_failure
    max_iters
end

function LinearVerbosity(;
        error_control = nothing, performance = nothing, numerical = nothing, kwargs...)
    # Fast path for default construction (type-stable)
    if error_control === nothing && performance === nothing &&
       numerical === nothing && isempty(kwargs)
        return LinearVerbosity(
            Silent(),
            Silent(),
            Silent(),
            Silent(),
            CustomLevel(1), # WARN_LEVEL in KrylovKit.jl
            Silent(),
            InfoLevel(),
            Silent(),
            ErrorLevel(),
            ErrorLevel(),
            Silent(),
            Silent(),
            Silent(),
            WarnLevel(),
            WarnLevel(),
            WarnLevel())
    end

    # Validate group arguments
    if error_control !== nothing && !(error_control isa AbstractMessageLevel)
        throw(ArgumentError("error_control must be a SciMLLogging.AbstractMessageLevel, got $(typeof(error_control))"))
    end
    if performance !== nothing && !(performance isa AbstractMessageLevel)
        throw(ArgumentError("performance must be a SciMLLogging.AbstractMessageLevel, got $(typeof(performance))"))
    end
    if numerical !== nothing && !(numerical isa AbstractMessageLevel)
        throw(ArgumentError("numerical must be a SciMLLogging.AbstractMessageLevel, got $(typeof(numerical))"))
    end

    # Validate individual kwargs
    for (key, value) in kwargs
        if !(key in error_control_options || key in performance_options ||
             key in numerical_options)
            throw(ArgumentError("Unknown verbosity option: $key. Valid options are: $(tuple(error_control_options..., performance_options..., numerical_options...))"))
        end
        if !(value isa AbstractMessageLevel)
            throw(ArgumentError("$key must be a SciMLLogging.AbstractMessageLevel, got $(typeof(value))"))
        end
    end

    # Build arguments using NamedTuple for type stability
    default_args = (
        default_lu_fallback = Silent(),
        no_right_preconditioning = Silent(),
        using_IterativeSolvers = Silent(),
        IterativeSolvers_iterations = Silent(),
        KrylovKit_verbosity = CustomLevel(1), # WARN_LEVEL in KrylovKit.jl
        KrylovJL_verbosity = Silent(),
        HYPRE_verbosity = InfoLevel(),
        pardiso_verbosity = Silent(),
        blas_errors = ErrorLevel(),
        blas_invalid_args = ErrorLevel(),
        blas_info = Silent(),
        blas_success = Silent(),
        condition_number=Silent(),
        convergence_failure=WarnLevel(),
        solver_failure=WarnLevel(),
        max_iters=WarnLevel()
    )

    # Apply group-level settings
    final_args = if error_control !== nothing || performance !== nothing ||
                    numerical !== nothing
        NamedTuple{keys(default_args)}(
            _resolve_arg_value(
                key, default_args[key], error_control, performance, numerical)
        for key in keys(default_args)
        )
    else
        default_args
    end

    # Apply individual overrides
    if !isempty(kwargs)
        final_args = merge(final_args, NamedTuple(kwargs))
    end

    LinearVerbosity(values(final_args)...)
end

# Constructor for verbosity presets following the hierarchical levels:
# None < Minimal < Standard < Detailed < All
# Each level includes all messages from levels below it plus additional ones
function LinearVerbosity(verbose::AbstractVerbosityPreset)
    if verbose isa Minimal
        # Minimal: Only fatal errors and critical warnings (BLAS errors/invalid args)
        LinearVerbosity(
            default_lu_fallback = Silent(),
            no_right_preconditioning = Silent(),
            using_IterativeSolvers = Silent(),
            IterativeSolvers_iterations = Silent(),
            KrylovKit_verbosity = Silent(),
            KrylovJL_verbosity = Silent(),
            HYPRE_verbosity = Silent(),
            pardiso_verbosity = Silent(),
            blas_errors = ErrorLevel(),
            blas_invalid_args = ErrorLevel(),
            blas_info = Silent(),
            blas_success = Silent(),
            condition_number = Silent(),
            convergence_failure = Silent(),
            solver_failure = Silent(),
            max_iters = Silent()
        )
    elseif verbose isa Standard
        # Standard: Everything from Minimal + non-fatal warnings
        LinearVerbosity()
    elseif verbose isa Detailed
        # Detailed: Everything from Standard + debugging/solver behavior
        LinearVerbosity(
            default_lu_fallback = WarnLevel(),
            no_right_preconditioning = InfoLevel(),
            using_IterativeSolvers = InfoLevel(),
            IterativeSolvers_iterations = Silent(),
            KrylovKit_verbosity = CustomLevel(2), # STARTSTOP_LEVEL in KrylovKit.jl
            KrylovJL_verbosity = CustomLevel(1), # verbose = true in Krylov.jl
            HYPRE_verbosity = InfoLevel(),
            pardiso_verbosity = CustomLevel(1), # verbose = true in Pardiso.jl
            blas_errors = ErrorLevel(),
            blas_invalid_args = ErrorLevel(),
            blas_info = InfoLevel(),
            blas_success = InfoLevel(),
            condition_number = Silent(),
            convergence_failure = WarnLevel(),
            solver_failure = WarnLevel(),
            max_iters = WarnLevel()
        )
    elseif verbose isa All
        # All: Maximum verbosity - every possible logging message at InfoLevel
        LinearVerbosity(
            default_lu_fallback = WarnLevel(),
            no_right_preconditioning = InfoLevel(),
            using_IterativeSolvers = InfoLevel(),
            IterativeSolvers_iterations = InfoLevel(),
            KrylovKit_verbosity = CustomLevel(3), # EACHITERATION_LEVEL in KrylovKit.jl
            KrylovJL_verbosity = CustomLevel(1),
            HYPRE_verbosity = InfoLevel(),
            pardiso_verbosity = CustomLevel(1), # verbsoe = true in Pardiso.jl
            blas_errors = ErrorLevel(),
            blas_invalid_args = ErrorLevel(),
            blas_info = InfoLevel(),
            blas_success = InfoLevel(),
            condition_number = InfoLevel(),
            convergence_failure = WarnLevel(),
            solver_failure = WarnLevel(),
            max_iters = WarnLevel()
        )
    end
end

@inline function LinearVerbosity(verbose::None)
    LinearVerbosity(
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent(),
    Silent())
end

# Helper function to resolve argument values based on group membership
@inline function _resolve_arg_value(key::Symbol, default_val, error_control, performance, numerical)
    if key in error_control_options && error_control !== nothing
        return error_control
    elseif key in performance_options && performance !== nothing
        return performance
    elseif key in numerical_options && numerical !== nothing
        return numerical
    else
        return default_val
    end
end

# Group classifications
const error_control_options = (:default_lu_fallback, :blas_errors, :blas_invalid_args)
const performance_options = (:no_right_preconditioning,)
const numerical_options = (:using_IterativeSolvers, :IterativeSolvers_iterations,
                       :KrylovKit_verbosity, :KrylovJL_verbosity, :HYPRE_verbosity, :pardiso_verbosity,
                       :blas_info, :blas_success, :condition_number, :convergence_failure, :solver_failure, :max_iters)

function option_group(option::Symbol)
    if option in error_control_options
        return :error_control
    elseif option in performance_options
        return :performance
    elseif option in numerical_options
        return :numerical
    else
        error("Unknown verbosity option: $option")
    end
end

# Get all options in a group
function group_options(verbosity::LinearVerbosity, group::Symbol)
    if group === :error_control
        return NamedTuple{error_control_options}(getproperty(verbosity, opt) for opt in error_control_options)
    elseif group === :performance
        return NamedTuple{performance_options}(getproperty(verbosity, opt) for opt in performance_options)
    elseif group === :numerical
        return NamedTuple{numerical_options}(getproperty(verbosity, opt) for opt in numerical_options)
    else
        error("Unknown group: $group")
    end
end