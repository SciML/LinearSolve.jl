mutable struct LinearVerbosity{Enabled} <: AbstractVerbositySpecifier{Enabled}
    # Error control
    default_lu_fallback::SciMLLogging.LogLevel
    # Performance
    no_right_preconditioning::SciMLLogging.LogLevel
    # Numerical
    using_iterative_solvers::SciMLLogging.LogLevel
    using_IterativeSolvers::SciMLLogging.LogLevel
    IterativeSolvers_iterations::SciMLLogging.LogLevel
    KrylovKit_verbosity::SciMLLogging.LogLevel
    KrylovJL_verbosity::SciMLLogging.LogLevel
    HYPRE_verbosity::SciMLLogging.LogLevel
    pardiso_verbosity::SciMLLogging.LogLevel
    blas_errors::SciMLLogging.LogLevel
    blas_invalid_args::SciMLLogging.LogLevel
    blas_info::SciMLLogging.LogLevel
    blas_success::SciMLLogging.LogLevel
    condition_number::SciMLLogging.LogLevel

    function LinearVerbosity{true}(;
        # Error control defaults
        default_lu_fallback = SciMLLogging.Warn(),
        # Performance defaults
        no_right_preconditioning = SciMLLogging.Warn(),
        # Numerical defaults
        using_iterative_solvers = SciMLLogging.Warn(),
        using_IterativeSolvers = SciMLLogging.Warn(),
        IterativeSolvers_iterations = SciMLLogging.Warn(),
        KrylovKit_verbosity = SciMLLogging.Warn(),
        KrylovJL_verbosity = SciMLLogging.Silent(),
        HYPRE_verbosity = SciMLLogging.Info(),
        pardiso_verbosity = SciMLLogging.Silent(),
        blas_errors = SciMLLogging.Warn(),
        blas_invalid_args = SciMLLogging.Warn(),
        blas_info = SciMLLogging.Silent(),
        blas_success = SciMLLogging.Silent(),
        condition_number = SciMLLogging.Silent())

        new{true}(default_lu_fallback, no_right_preconditioning,
                     using_iterative_solvers, using_IterativeSolvers,
                     IterativeSolvers_iterations, KrylovKit_verbosity,
                     KrylovJL_verbosity, HYPRE_verbosity, pardiso_verbosity,
                     blas_errors, blas_invalid_args, blas_info, blas_success, condition_number)
    end

    function LinearVerbosity{false}()
        new{false}(SciMLLogging.Silent(), SciMLLogging.Silent(),
        SciMLLogging.Silent(), SciMLLogging.Silent(),
        SciMLLogging.Silent(), SciMLLogging.Silent(),
        SciMLLogging.Silent(), SciMLLogging.Silent(), SciMLLogging.Silent(),
        SciMLLogging.Silent(), SciMLLogging.Silent(), SciMLLogging.Silent(), SciMLLogging.Silent(), SciMLLogging.Silent())
    end
end

LinearVerbosity(enabled::Bool) = enabled ? LinearVerbosity{true}() : LinearVerbosity{false}()

function LinearVerbosity(verbose::SciMLLogging.VerbosityPreset)
    if verbose isa SciMLLogging.None
        LinearVerbosity{false}()
    elseif verbose isa SciMLLogging.All
        LinearVerbosity{true}(
            default_lu_fallback = SciMLLogging.Info(),
            no_right_preconditioning = SciMLLogging.Info(),
            using_iterative_solvers = SciMLLogging.Info(),
            using_IterativeSolvers = SciMLLogging.Info(),
            IterativeSolvers_iterations = SciMLLogging.Info(),
            KrylovKit_verbosity = SciMLLogging.Info(),
            KrylovJL_verbosity = SciMLLogging.Info(),
            HYPRE_verbosity = SciMLLogging.Info(),
            pardiso_verbosity = SciMLLogging.Info(),
            blas_errors = SciMLLogging.Info(),
            blas_invalid_args = SciMLLogging.Info(),
            blas_info = SciMLLogging.Info(),
            blas_success = SciMLLogging.Info(),
            condition_number = SciMLLogging.Info()
        )
    elseif verbose isa SciMLLogging.Minimal
        LinearVerbosity{true}(
            default_lu_fallback = SciMLLogging.Error(),
            no_right_preconditioning = SciMLLogging.Silent(),
            using_iterative_solvers = SciMLLogging.Silent(),
            using_IterativeSolvers = SciMLLogging.Silent(),
            IterativeSolvers_iterations = SciMLLogging.Silent(),
            KrylovKit_verbosity = SciMLLogging.Silent(),
            KrylovJL_verbosity = SciMLLogging.Silent(),
            HYPRE_verbosity = SciMLLogging.Silent(),
            pardiso_verbosity = SciMLLogging.Silent(),
            blas_errors = SciMLLogging.Error(),
            blas_invalid_args = SciMLLogging.Error(),
            blas_info = SciMLLogging.Silent(),
            blas_success = SciMLLogging.Silent(),
            condition_number = SciMLLogging.Silent()
        )
    elseif verbose isa SciMLLogging.Standard
        LinearVerbosity{true}()  # Use default settings
    elseif verbose isa SciMLLogging.Detailed
        LinearVerbosity{true}(
            default_lu_fallback = SciMLLogging.Info(),
            no_right_preconditioning = SciMLLogging.Info(),
            using_iterative_solvers = SciMLLogging.Info(),
            using_IterativeSolvers = SciMLLogging.Info(),
            IterativeSolvers_iterations = SciMLLogging.Info(),
            KrylovKit_verbosity = SciMLLogging.Warn(),
            KrylovJL_verbosity = SciMLLogging.Warn(),
            HYPRE_verbosity = SciMLLogging.Info(),
            pardiso_verbosity = SciMLLogging.Warn(),
            blas_errors = SciMLLogging.Warn(),
            blas_invalid_args = SciMLLogging.Warn(),
            blas_info = SciMLLogging.Info(),
            blas_success = SciMLLogging.Info(),
            condition_number = SciMLLogging.Info()
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
    if error_control !== nothing && !(error_control isa SciMLLogging.LogLevel)
        throw(ArgumentError("error_control must be a SciMLLogging.LogLevel, got $(typeof(error_control))"))
    end
    if performance !== nothing && !(performance isa SciMLLogging.LogLevel)
        throw(ArgumentError("performance must be a SciMLLogging.LogLevel, got $(typeof(performance))"))
    end
    if numerical !== nothing && !(numerical isa SciMLLogging.LogLevel)
        throw(ArgumentError("numerical must be a SciMLLogging.LogLevel, got $(typeof(numerical))"))
    end

    # Validate individual kwargs
    for (key, value) in kwargs
        if !(key in error_control_options || key in performance_options || key in numerical_options)
            throw(ArgumentError("Unknown verbosity option: $key. Valid options are: $(tuple(error_control_options..., performance_options..., numerical_options...))"))
        end
        if !(value isa SciMLLogging.LogLevel)
            throw(ArgumentError("$key must be a SciMLLogging.LogLevel, got $(typeof(value))"))
        end
    end

    # Build arguments using NamedTuple for type stability
    default_args = (
        default_lu_fallback = SciMLLogging.Warn(),
        no_right_preconditioning = SciMLLogging.Warn(),
        using_iterative_solvers = SciMLLogging.Warn(),
        using_IterativeSolvers = SciMLLogging.Warn(),
        IterativeSolvers_iterations = SciMLLogging.Warn(),
        KrylovKit_verbosity = SciMLLogging.Warn(),
        KrylovJL_verbosity = SciMLLogging.Silent(),
        HYPRE_verbosity = SciMLLogging.Info(),
        pardiso_verbosity = SciMLLogging.Silent(),
        blas_errors = SciMLLogging.Warn(),
        blas_invalid_args = SciMLLogging.Warn(),
        blas_info = SciMLLogging.Silent(),
        blas_success = SciMLLogging.Silent(),
        condition_number = SciMLLogging.Silent()
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
        if value isa SciMLLogging.LogLevel
            for opt in error_control_options
                setfield!(verbosity, opt, value)
            end
        else
            error("error_control must be set to a SciMLLogging.LogLevel")
        end
    elseif name === :performance
        if value isa SciMLLogging.LogLevel
            for opt in performance_options
                setfield!(verbosity, opt, value)
            end
        else
            error("performance must be set to a SciMLLogging.LogLevel")
        end
    elseif name === :numerical
        if value isa SciMLLogging.LogLevel
            for opt in numerical_options
                setfield!(verbosity, opt, value)
            end
        else
            error("numerical must be set to a SciMLLogging.LogLevel")
        end
    else
        # Check if this is an individual option
        if name in error_control_options || name in performance_options || name in numerical_options
            if value isa SciMLLogging.LogLevel
                setfield!(verbosity, name, value)
            else
                error("$name must be set to a SciMLLogging.LogLevel")
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
