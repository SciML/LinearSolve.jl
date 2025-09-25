mutable struct LinearVerbosity{Enabled} <: AbstractVerbositySpecifier{Enabled}
    # Error control
    default_lu_fallback::MessageLevel
    # Performance
    no_right_preconditioning::MessageLevel
    # Numerical
    using_iterative_solvers::MessageLevel
    using_IterativeSolvers::MessageLevel
    IterativeSolvers_iterations::MessageLevel
    KrylovKit_verbosity::MessageLevel
    KrylovJL_verbosity::MessageLevel
    HYPRE_verbosity::MessageLevel
    pardiso_verbosity::MessageLevel
    blas_errors::MessageLevel
    blas_invalid_args::MessageLevel
    blas_info::MessageLevel
    blas_success::MessageLevel
    condition_number::MessageLevel

    function LinearVerbosity{true}(;
        # Error control defaults
        default_lu_fallback = WarnLevel(),
        # Performance defaults
        no_right_preconditioning = WarnLevel(),
        # Numerical defaults
        using_iterative_solvers = WarnLevel(),
        using_IterativeSolvers = WarnLevel(),
        IterativeSolvers_iterations = WarnLevel(),
        KrylovKit_verbosity = WarnLevel(),
        KrylovJL_verbosity = Silent(),
        HYPRE_verbosity = InfoLevel(),
        pardiso_verbosity = Silent(),
        blas_errors = WarnLevel(),
        blas_invalid_args = WarnLevel(),
        blas_info = Silent(),
        blas_success = Silent(),
        condition_number = Silent())

        new{true}(default_lu_fallback, no_right_preconditioning,
                     using_iterative_solvers, using_IterativeSolvers,
                     IterativeSolvers_iterations, KrylovKit_verbosity,
                     KrylovJL_verbosity, HYPRE_verbosity, pardiso_verbosity,
                     blas_errors, blas_invalid_args, blas_info, blas_success, condition_number)
    end

    function LinearVerbosity{false}()
        new{false}(Silent(), Silent(),
        Silent(), Silent(),
        Silent(), Silent(),
        Silent(), Silent(), Silent(),
        Silent(), Silent(), Silent(), Silent(), Silent())
    end
end

LinearVerbosity(enabled::Bool) = enabled ? LinearVerbosity{true}() : LinearVerbosity{false}()

function LinearVerbosity(verbose::SciMLLogging.VerbosityPreset)
    if verbose isa SciMLLogging.None
        LinearVerbosity{false}()
    elseif verbose isa SciMLLogging.All
        LinearVerbosity{true}(
            default_lu_fallback = InfoLevel(),
            no_right_preconditioning = InfoLevel(),
            using_iterative_solvers = InfoLevel(),
            using_IterativeSolvers = InfoLevel(),
            IterativeSolvers_iterations = InfoLevel(),
            KrylovKit_verbosity = InfoLevel(),
            KrylovJL_verbosity = InfoLevel(),
            HYPRE_verbosity = InfoLevel(),
            pardiso_verbosity = InfoLevel(),
            blas_errors = InfoLevel(),
            blas_invalid_args = InfoLevel(),
            blas_info = InfoLevel(),
            blas_success = InfoLevel(),
            condition_number = InfoLevel()
        )
    elseif verbose isa SciMLLogging.Minimal
        LinearVerbosity{true}(
            default_lu_fallback = ErrorLevel(),
            no_right_preconditioning = Silent(),
            using_iterative_solvers = Silent(),
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
            condition_number = Silent()
        )
    elseif verbose isa SciMLLogging.Standard
        LinearVerbosity{true}()  # Use default settings
    elseif verbose isa SciMLLogging.Detailed
        LinearVerbosity{true}(
            default_lu_fallback = InfoLevel(),
            no_right_preconditioning = InfoLevel(),
            using_iterative_solvers = InfoLevel(),
            using_IterativeSolvers = InfoLevel(),
            IterativeSolvers_iterations = InfoLevel(),
            KrylovKit_verbosity = WarnLevel(),
            KrylovJL_verbosity = WarnLevel(),
            HYPRE_verbosity = InfoLevel(),
            pardiso_verbosity = WarnLevel(),
            blas_errors = WarnLevel(),
            blas_invalid_args = WarnLevel(),
            blas_info = InfoLevel(),
            blas_success = InfoLevel(),
            condition_number = InfoLevel()
        )
    else
        LinearVerbosity{true}()  # Default fallback
    end
end

@inline function LinearVerbosity(verbose::SciMLLogging.None)
    LinearVerbosity{false}()
end

function LinearVerbosity(; error_control=nothing, performance=nothing, numerical=nothing, kwargs...)
    # Validate group arguments
    if error_control !== nothing && !(error_control isa MessageLevel)
        throw(ArgumentError("error_control must be a SciMLLogging.MessageLevel, got $(typeof(error_control))"))
    end
    if performance !== nothing && !(performance isa MessageLevel)
        throw(ArgumentError("performance must be a SciMLLogging.MessageLevel, got $(typeof(performance))"))
    end
    if numerical !== nothing && !(numerical isa MessageLevel)
        throw(ArgumentError("numerical must be a SciMLLogging.MessageLevel, got $(typeof(numerical))"))
    end

    # Validate individual kwargs
    for (key, value) in kwargs
        if !(key in error_control_options || key in performance_options || key in numerical_options)
            throw(ArgumentError("Unknown verbosity option: $key. Valid options are: $(tuple(error_control_options..., performance_options..., numerical_options...))"))
        end
        if !(value isa MessageLevel)
            throw(ArgumentError("$key must be a SciMLLogging.MessageLevel, got $(typeof(value))"))
        end
    end

    # Build arguments using NamedTuple for type stability
    default_args = (
        default_lu_fallback = WarnLevel(),
        no_right_preconditioning = WarnLevel(),
        using_iterative_solvers = WarnLevel(),
        using_IterativeSolvers = WarnLevel(),
        IterativeSolvers_iterations = WarnLevel(),
        KrylovKit_verbosity = WarnLevel(),
        KrylovJL_verbosity = Silent(),
        HYPRE_verbosity = InfoLevel(),
        pardiso_verbosity = Silent(),
        blas_errors = WarnLevel(),
        blas_invalid_args = WarnLevel(),
        blas_info = Silent(),
        blas_success = Silent(),
        condition_number = Silent()
    )

    # Apply group-level settings
    final_args = if error_control !== nothing || performance !== nothing || numerical !== nothing
        NamedTuple{keys(default_args)}(
            _resolve_arg_value(key, default_args[key], error_control, performance, numerical)
            for key in keys(default_args)
        )
    else
        default_args
    end

    # Apply individual overrides
    if !isempty(kwargs)
        final_args = merge(final_args, NamedTuple(kwargs))
    end

    LinearVerbosity{true}(; final_args...)
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
const numerical_options = (:using_iterative_solvers, :using_IterativeSolvers, :IterativeSolvers_iterations,
                       :KrylovKit_verbosity, :KrylovJL_verbosity, :HYPRE_verbosity, :pardiso_verbosity,
                       :blas_info, :blas_success, :condition_number)

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

function Base.setproperty!(verbosity::LinearVerbosity, name::Symbol, value)
    # Check if this is a group name
    if name === :error_control
        if value isa MessageLevel
            for opt in error_control_options
                setfield!(verbosity, opt, value)
            end
        else
            error("error_control must be set to a SciMLLogging.MessageLevel")
        end
    elseif name === :performance
        if value isa MessageLevel
            for opt in performance_options
                setfield!(verbosity, opt, value)
            end
        else
            error("performance must be set to a SciMLLogging.MessageLevel")
        end
    elseif name === :numerical
        if value isa MessageLevel
            for opt in numerical_options
                setfield!(verbosity, opt, value)
            end
        else
            error("numerical must be set to a SciMLLogging.MessageLevel")
        end
    else
        # Check if this is an individual option
        if name in error_control_options || name in performance_options || name in numerical_options
            if value isa MessageLevel
                setfield!(verbosity, name, value)
            else
                error("$name must be set to a SciMLLogging.MessageLevel")
            end
        else
            # Fall back to default behavior for unknown properties
            setfield!(verbosity, name, value)
        end
    end
end

function Base.getproperty(verbosity::LinearVerbosity, name::Symbol)
    # Check if this is a group name
    if name === :error_control
        return group_options(verbosity, :error_control)
    elseif name === :performance
        return group_options(verbosity, :performance)
    elseif name === :numerical
        return group_options(verbosity, :numerical)
    else
        # Fall back to default field access
        return getfield(verbosity, name)
    end
end

function Base.show(io::IO, verbosity::LinearVerbosity{Enabled}) where Enabled
    if Enabled
        println(io, "LinearVerbosity{true}:")

        # Show error control group
        println(io, "  Error Control:")
        for opt in error_control_options
            level = getfield(verbosity, opt)
            level_name = typeof(level).name.name
            println(io, "    $opt: $level_name")
        end

        # Show performance group
        println(io, "  Performance:")
        for opt in performance_options
            level = getfield(verbosity, opt)
            level_name = typeof(level).name.name
            println(io, "    $opt: $level_name")
        end

        # Show numerical group
        println(io, "  Numerical:")
        for opt in numerical_options
            level = getfield(verbosity, opt)
            level_name = typeof(level).name.name
            println(io, "    $opt: $level_name")
        end
    else
        print(io, "LinearVerbosity{false} (all logging disabled)")
    end
end
