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

The verbosity system in LinearSolve.jl provides fine-grained control over the diagnostic messages, warnings, and errors that are displayed during the solution of linear systems. To use this system, a keyword argument `verbose` is provided to `solve`. 

```@docs
LinearVerbosity
```

### Basic Usage 

#### Global Verbosity Control

```julia
using LinearSolve

# Suppress all messages
verbose = LinearVerbosity(SciMLLogging.None())
prob = LinearProblem(A, b)
sol = solve(prob; verbose=verbose)

# Show only essential messages (critical errors and fatal issues)
verbose = LinearVerbosity(SciMLLogging.Minimal())
sol = solve(prob; verbose=verbose)

# Use default settings (balanced verbosity for typical usage)
verbose = LinearVerbosity(SciMLLogging.Standard())
sol = solve(prob; verbose=verbose)

# Show comprehensive debugging information
verbose = LinearVerbosity(SciMLLogging.Detailed())
sol = solve(prob; verbose=verbose)

# Show all messages (maximum verbosity)
verbose = LinearVerbosity(SciMLLogging.All())
sol = solve(prob; verbose=verbose)
```

#### Group Level Control 

```julia 
# Customize by category
verbose = LinearVerbosity(
    error_control = SciMLLogging.Warn(),   # Show warnings for error control related issues
    performance = SciMLLogging.Silent(),     # Suppress performance messages
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
    default_lu_fallback = SciMLLogging.InfoLevel(),                     # Show info when LU fallback is used
    KrylovJL_verbosity = SciMLLogging.WarnLevel(),                      # Show warnings from KrylovJL
    no_right_preconditioning = SciMLLogging.Silent(),                # Suppress right preconditioning messages
    KrylovKit_verbosity = SciMLLogging.Level(KrylovKit.WARN_LEVEL) # Set KrylovKit verbosity level using KrylovKit's own verbosity levels
)

sol = solve(prob; verbose=verbose)

```