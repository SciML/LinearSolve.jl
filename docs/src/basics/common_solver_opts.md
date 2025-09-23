# Common Solver Options (Keyword Arguments for Solve)

While many algorithms have specific arguments within their constructor,
the keyword arguments for `solve` are common across all the algorithms
in order to give composability. These are also the options taken at `init` time.
The following are the options these algorithms take, along with their defaults.

## General Controls

  - `alias::LinearAliasSpecifier`: Holds the fields `alias_A` and `alias_b` which specify
    whether to alias the matrices `A` and `b` respectively. When these fields are `true`,
    `A` and `b` can be written to and changed by the solver algorithm. When fields are `nothing`
    the default behavior is used, which is to default to `true` when the algorithm is known
    not to modify the matrices, and false otherwise.
  - `verbose`: Whether to print extra information. Defaults to `false`.
  - `assumptions`: Sets the assumptions of the operator in order to effect the default
    choice algorithm. See the [Operator Assumptions page for more details](@ref assumptions).

## Iterative Solver Controls

Error controls are not used by all algorithms. Specifically, direct solves always
solve completely. Error controls only apply to iterative solvers.

  - `abstol`: The absolute tolerance. Defaults to `√(eps(eltype(A)))`
  - `reltol`: The relative tolerance. Defaults to `√(eps(eltype(A)))`
  - `maxiters`: The number of iterations allowed. Defaults to `length(prob.b)`
  - `Pl,Pr`: The left and right preconditioners, respectively. For more information,
    see [the Preconditioners page](@ref prec).

## Verbosity Controls

The verbosity system in LinearSolve.jl provides fine-grained control over the diagnostic messages, warnings, and errors that are displayed during the solution of linear systems.

The verbosity system is organized hierarchically into three main categories:

1. Error Control - Messages related to fallbacks and error handling
2. Performance - Messages related to performance considerations
3. Numerical - Messages related to numerical solvers and iterations

Each category can be configured independently, and individual settings can be adjusted to suit your needs.

### Verbosity Levels
The following verbosity levels are available:

#### Individual Settings
These settings are meant for individual settings within a category. These can also be used to set all of the individual settings in a group to the same value.
- SciMLLogging.None() - Suppress all messages
- SciMLLogging.Info() - Show message as log message at info level
- SciMLLogging.Warn() - Show warnings (default for most settings)
- SciMLLogging.Error() - Throw errors instead of warnings
- SciMLLogging.Level(n) - Show messages with a log level setting of n

#### Group Settings
These settings are meant for controlling a group of settings. 
- SciMLLogging.Default() - Use the default settings
- SciMLLogging.All() - Show all possible messages

### Basic Usage 

#### Global Verbosity Control

```julia 
using LinearSolve

# Suppress all messages
verbose = LinearVerbosity(SciMLLogging.None())
prob = LinearProblem(A, b)
sol = solve(prob; verbose=verbose)

# Show all messages
verbose = LinearVerbosity(SciMLLogging.All())
sol = solve(prob; verbose=verbose)

# Use default settings
verbose = LinearVerbosity(SciMLLogging.Default())
sol = solve(prob; verbose=verbose)
```

#### Group Level Control 

```julia 
# Customize by category
verbose = LinearVerbosity(
    error_control = SciMLLogging.Warn(),   # Show warnings for error control related issues
    performance = SciMLLogging.None(),     # Suppress performance messages
    numerical = SciMLLogging.Info()        # Show all numerical related log messages at info level
)

sol = solve(prob; verbose=verbose)
```

#### Fine-grained Control
The constructor for `LinearVerbosity` allows you to set verbosity for each specific message toggle, giving you fine-grained control. 
The verbosity settings for the toggles are automatically passed to the group objects. 
```julia
# Set specific message types
verbose = LinearVerbosity(
    default_lu_fallback = SciMLLogging.Info(),                     # Show info when LU fallback is used
    KrylovJL_verbosity = SciMLLogging.Warn(),                      # Show warnings from KrylovJL
    no_right_preconditioning = SciMLLogging.None(),                # Suppress right preconditioning messages
    KrylovKit_verbosity = SciMLLogging.Level(KrylovKit.WARN_LEVEL) # Set KrylovKit verbosity level using KrylovKit's own verbosity levels
)

sol = solve(prob; verbose=verbose)

```

#### Verbosity Levels
##### Error Control Settings
- default_lu_fallback: Controls messages when falling back to LU factorization (default: Warn)
##### Performance Settings
- no_right_preconditioning: Controls messages when right preconditioning is not used (default: Warn)
##### Numerical Settings
- using_IterativeSolvers: Controls messages when using the IterativeSolvers.jl package (default: Warn)
- IterativeSolvers_iterations: Controls messages about iteration counts from IterativeSolvers.jl (default: Warn)
- KrylovKit_verbosity: Controls messages from the KrylovKit.jl package (default: Warn)
- KrylovJL_verbosity: Controls verbosity of the KrylovJL.jl package (default: None)