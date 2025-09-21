mutable struct LinearVerbosity{Enabled} <: AbstractVerbositySpecifier{Enabled}
    # Error control
    default_lu_fallback::Verbosity.LogLevel
    # Performance
    no_right_preconditioning::Verbosity.LogLevel
    # Numerical
    using_iterative_solvers::Verbosity.LogLevel
    using_IterativeSolvers::Verbosity.LogLevel
    IterativeSolvers_iterations::Verbosity.LogLevel
    KrylovKit_verbosity::Verbosity.LogLevel
    KrylovJL_verbosity::Verbosity.LogLevel
    HYPRE_verbosity::Verbosity.LogLevel
    pardiso_verbosity::Verbosity.LogLevel

    function LinearVerbosity{true}(;
        # Error control defaults
        default_lu_fallback = Verbosity.Warn(),
        # Performance defaults
        no_right_preconditioning = Verbosity.Warn(),
        # Numerical defaults
        using_iterative_solvers = Verbosity.Warn(),
        using_IterativeSolvers = Verbosity.Warn(),
        IterativeSolvers_iterations = Verbosity.Warn(),
        KrylovKit_verbosity = Verbosity.Warn(),
        KrylovJL_verbosity = Verbosity.Silent(),
        HYPRE_verbosity = Verbosity.Info(),
        pardiso_verbosity = Verbosity.Silent())

        new{true}(default_lu_fallback, no_right_preconditioning,
                     using_iterative_solvers, using_IterativeSolvers,
                     IterativeSolvers_iterations, KrylovKit_verbosity,
                     KrylovJL_verbosity, HYPRE_verbosity, pardiso_verbosity)
    end

    function LinearVerbosity{false}()
        new{false}(Verbosity.Silent(), Verbosity.Silent(), 
        Verbosity.Silent(), Verbosity.Silent(), 
        Verbosity.Silent(), Verbosity.Silent(), 
        Verbosity.Silent(), Verbosity.Silent(), Verbosity.Silent())
    end
end

LinearVerbosity(enabled::Bool) = enabled ? LinearVerbosity{true}() : LinearVerbosity{false}()

function LinearVerbosity(verbose::Verbosity.VerbosityPreset)
    if verbose isa Verbosity.None
        LinearVerbosity{false}()
    elseif verbose isa Verbosity.All
        LinearVerbosity{true}(
            default_lu_fallback = Verbosity.Info(),
            no_right_preconditioning = Verbosity.Info(),
            using_iterative_solvers = Verbosity.Info(),
            using_IterativeSolvers = Verbosity.Info(),
            IterativeSolvers_iterations = Verbosity.Info(),
            KrylovKit_verbosity = Verbosity.Info(),
            KrylovJL_verbosity = Verbosity.Info(),
            HYPRE_verbosity = Verbosity.Info(),
            pardiso_verbosity = Verbosity.Info()
        )
    elseif verbose isa Verbosity.Minimal
        LinearVerbosity{true}(
            default_lu_fallback = Verbosity.Error(),
            no_right_preconditioning = Verbosity.Silent(),
            using_iterative_solvers = Verbosity.Silent(),
            using_IterativeSolvers = Verbosity.Silent(),
            IterativeSolvers_iterations = Verbosity.Silent(),
            KrylovKit_verbosity = Verbosity.Silent(),
            KrylovJL_verbosity = Verbosity.Silent(),
            HYPRE_verbosity = Verbosity.Silent(),
            pardiso_verbosity = Verbosity.Silent()
        )
    elseif verbose isa Verbosity.Standard
        LinearVerbosity{true}()  # Use default settings
    elseif verbose isa Verbosity.Detailed
        LinearVerbosity{true}(
            default_lu_fallback = Verbosity.Info(),
            no_right_preconditioning = Verbosity.Info(),
            using_iterative_solvers = Verbosity.Info(),
            using_IterativeSolvers = Verbosity.Info(),
            IterativeSolvers_iterations = Verbosity.Info(),
            KrylovKit_verbosity = Verbosity.Warn(),
            KrylovJL_verbosity = Verbosity.Warn(),
            HYPRE_verbosity = Verbosity.Info(),
            pardiso_verbosity = Verbosity.Warn()
        )
    else
        LinearVerbosity{true}()  # Default fallback
    end
end

@inline function LinearVerbosity(verbose::Verbosity.None)
    LinearVerbosity{false}()
end

function LinearVerbosity(; error_control=nothing, performance=nothing, numerical=nothing, kwargs...)
    # Validate group arguments
    if error_control !== nothing && !(error_control isa Verbosity.LogLevel)
        throw(ArgumentError("error_control must be a Verbosity.LogLevel, got $(typeof(error_control))"))
    end
    if performance !== nothing && !(performance isa Verbosity.LogLevel)
        throw(ArgumentError("performance must be a Verbosity.LogLevel, got $(typeof(performance))"))
    end
    if numerical !== nothing && !(numerical isa Verbosity.LogLevel)
        throw(ArgumentError("numerical must be a Verbosity.LogLevel, got $(typeof(numerical))"))
    end

    # Validate individual kwargs
    for (key, value) in kwargs
        if !(key in error_control_options || key in performance_options || key in numerical_options)
            throw(ArgumentError("Unknown verbosity option: $key. Valid options are: $(tuple(error_control_options..., performance_options..., numerical_options...))"))
        end
        if !(value isa Verbosity.LogLevel)
            throw(ArgumentError("$key must be a Verbosity.LogLevel, got $(typeof(value))"))
        end
    end

    # Build arguments using NamedTuple for type stability
    default_args = (
        default_lu_fallback = Verbosity.Warn(),
        no_right_preconditioning = Verbosity.Warn(),
        using_iterative_solvers = Verbosity.Warn(),
        using_IterativeSolvers = Verbosity.Warn(),
        IterativeSolvers_iterations = Verbosity.Warn(),
        KrylovKit_verbosity = Verbosity.Warn(),
        KrylovJL_verbosity = Verbosity.Silent(),
        HYPRE_verbosity = Verbosity.Info(),
        pardiso_verbosity = Verbosity.Silent()
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
const error_control_options = (:default_lu_fallback,)
const performance_options = (:no_right_preconditioning,)
const numerical_options = (:using_iterative_solvers, :using_IterativeSolvers, :IterativeSolvers_iterations,
                       :KrylovKit_verbosity, :KrylovJL_verbosity, :HYPRE_verbosity, :pardiso_verbosity)

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
        if value isa Verbosity.LogLevel
            for opt in error_control_options
                setfield!(verbosity, opt, value)
            end
        else
            error("error_control must be set to a Verbosity.LogLevel")
        end
    elseif name === :performance
        if value isa Verbosity.LogLevel
            for opt in performance_options
                setfield!(verbosity, opt, value)
            end
        else
            error("performance must be set to a Verbosity.LogLevel")
        end
    elseif name === :numerical
        if value isa Verbosity.LogLevel
            for opt in numerical_options
                setfield!(verbosity, opt, value)
            end
        else
            error("numerical must be set to a Verbosity.LogLevel")
        end
    else
        # Check if this is an individual option
        if name in error_control_options || name in performance_options || name in numerical_options
            if value isa Verbosity.LogLevel
                setfield!(verbosity, name, value)
            else
                error("$name must be set to a Verbosity.LogLevel")
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

