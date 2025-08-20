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
- Verbosity.None() - Suppress all messages
- Verbosity.Info() - Show message as log message at info level
- Verbosity.Warn() - Show warnings (default for most settings)
- Verbosity.Error() - Throw errors instead of warnings
- Verbosity.Level(n) - Show messages with a log level setting of n

#### Group Settings
These settings are meant for controlling a group of settings. 
- Verbosity.Default() - Use the default settings
- Verbosity.All() - Show all possible messages

### Basic Usage 

#### Global Verbosity Control

```julia 
using LinearSolve

# Suppress all messages
verbose = LinearVerbosity(Verbosity.None())
prob = LinearProblem(A, b)
sol = solve(prob; verbose=verbose)

# Show all messages
verbose = LinearVerbosity(Verbosity.All())
sol = solve(prob; verbose=verbose)

# Use default settings
verbose = LinearVerbosity(Verbosity.Default())
sol = solve(prob; verbose=verbose)
```

#### Group Level Control 

```julia 
# Customize by category
verbose = LinearVerbosity(
    error_control = Verbosity.Warn(),   # Show warnings for error control related issues
    performance = Verbosity.None(),     # Suppress performance messages
    numerical = Verbosity.Info()        # Show all numerical related log messages at info level
)

sol = solve(prob; verbose=verbose)
```

#### Fine-grained Control
The constructor for `LinearVerbosity` allows you to set verbosity for each specific message toggle, giving you fine-grained control. 
The verbosity settings for the toggles are automatically passed to the group objects. 
```julia
# Set specific message types
verbose = LinearVerbosity(
    default_lu_fallback = Verbosity.Info(),                     # Show info when LU fallback is used
    KrylovJL_verbosity = Verbosity.Warn(),                      # Show warnings from KrylovJL
    no_right_preconditioning = Verbosity.None(),                # Suppress right preconditioning messages
    KrylovKit_verbosity = Verbosity.Level(KrylovKit.WARN_LEVEL) # Set KrylovKit verbosity level using KrylovKit's own verbosity levels
)

sol = solve(prob; verbose=verbose)

```

#### Verbosity Levels
##### Error Control Settings
- default_lu_fallback: Controls messages when falling back to LU factorization (default: Warn)
- blas_invalid_args: Controls messages when BLAS/LAPACK receives invalid arguments (default: Error)
##### Performance Settings
- no_right_preconditioning: Controls messages when right preconditioning is not used (default: Warn)
##### Numerical Settings
- using_IterativeSolvers: Controls messages when using the IterativeSolvers.jl package (default: Warn)
- IterativeSolvers_iterations: Controls messages about iteration counts from IterativeSolvers.jl (default: Warn)
- KrylovKit_verbosity: Controls messages from the KrylovKit.jl package (default: Warn)
- KrylovJL_verbosity: Controls verbosity of the KrylovJL.jl package (default: None)
- blas_errors: Controls messages for BLAS/LAPACK errors like singular matrices (default: Warn)
- blas_info: Controls informational messages from BLAS/LAPACK operations (default: None)
- blas_success: Controls success messages from BLAS/LAPACK operations (default: None)
- condition_number: Controls computation and logging of matrix condition numbers (default: None)

### BLAS/LAPACK Return Code Interpretation

LinearSolve.jl now provides detailed interpretation of BLAS/LAPACK return codes (info parameter) to help diagnose numerical issues. When BLAS/LAPACK operations encounter problems, the logging system will provide human-readable explanations based on the specific return code and operation.

#### Common BLAS/LAPACK Return Codes

- **info = 0**: Operation completed successfully
- **info < 0**: Argument -info had an illegal value (e.g., wrong dimensions, invalid parameters)
- **info > 0**: Operation-specific issues:
  - **LU factorization (getrf)**: U(info,info) is exactly zero, matrix is singular
  - **Cholesky factorization (potrf)**: Leading minor of order info is not positive definite
  - **QR factorization (geqrf)**: Numerical issues with Householder reflector info
  - **SVD (gesdd/gesvd)**: Algorithm failed to converge, info off-diagonal elements did not converge
  - **Eigenvalue computation (syev/heev)**: info off-diagonal elements did not converge to zero
  - **Bunch-Kaufman (sytrf/hetrf)**: D(info,info) is exactly zero, matrix is singular

#### Example Usage with Enhanced BLAS Logging

```julia
using LinearSolve

# Enable detailed BLAS error logging with condition numbers
verbose = LinearVerbosity(
    blas_errors = Verbosity.Info(),     # Show detailed error interpretations
    blas_info = Verbosity.Info(),       # Show all BLAS operation details
    condition_number = Verbosity.Info() # Compute and log condition numbers
)

# Solve a potentially singular system
A = [1.0 2.0; 2.0 4.0]  # Singular matrix
b = [1.0, 2.0]
prob = LinearProblem(A, b)
sol = solve(prob, LUFactorization(); verbose=verbose)

# The logging will show:
# - BLAS/LAPACK dgetrf: Matrix is singular
# - Details: U(2,2) is exactly zero. The factorization has been completed, but U is singular
# - Return code (info): 2
# - Additional context: matrix size, condition number, memory usage
```

The enhanced logging also provides:
- Matrix properties (size, type, element type)
- Memory usage estimates
- Detailed context for debugging numerical issues
- Condition number computation controlled by the condition_number verbosity setting