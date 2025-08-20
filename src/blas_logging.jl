# BLAS and LAPACK Return Code Interpretation

using SciMLLogging: Verbosity, @match, @SciMLMessage, verbosity_to_int
using LinearAlgebra: cond

"""
    interpret_blas_code(func::Symbol, info::Integer)

Interpret BLAS/LAPACK return codes (info parameter) to provide human-readable error messages.
Returns a tuple of (category::Symbol, message::String, details::String)
"""
function interpret_blas_code(func::Symbol, info::Integer)
    if info == 0
        return (:success, "Operation completed successfully", "")
    elseif info < 0
        return (:invalid_argument, 
                "Invalid argument error", 
                "Argument $(-info) had an illegal value")
    else
        # info > 0 means different things for different functions
        return interpret_positive_info(func, info)
    end
end

function interpret_positive_info(func::Symbol, info::Integer)
    func_str = string(func)
    
    # LU factorization routines
    if occursin("getrf", func_str)
        return (:singular_matrix,
                "Matrix is singular",
                "U($info,$info) is exactly zero. The factorization has been completed, but U is singular and division by U will produce infinity.")
    
    # Cholesky factorization routines
    elseif occursin("potrf", func_str)
        return (:not_positive_definite,
                "Matrix is not positive definite",
                "The leading minor of order $info is not positive definite, and the factorization could not be completed.")
    
    # QR factorization routines
    elseif occursin("geqrf", func_str) || occursin("geqrt", func_str)
        return (:numerical_issue,
                "Numerical issue in QR factorization",
                "Householder reflector $info could not be formed properly.")
    
    # SVD routines
    elseif occursin("gesdd", func_str) || occursin("gesvd", func_str)
        return (:convergence_failure,
                "SVD did not converge",
                "The algorithm failed to compute singular values. $info off-diagonal elements of an intermediate bidiagonal form did not converge to zero.")
    
    # Symmetric/Hermitian eigenvalue routines
    elseif occursin("syev", func_str) || occursin("heev", func_str)
        return (:convergence_failure,
                "Eigenvalue computation did not converge",
                "$info off-diagonal elements of an intermediate tridiagonal form did not converge to zero.")
    
    # Bunch-Kaufman factorization
    elseif occursin("sytrf", func_str) || occursin("hetrf", func_str)
        return (:singular_matrix,
                "Matrix is singular", 
                "D($info,$info) is exactly zero. The factorization has been completed, but the block diagonal matrix D is singular.")
    
    # Solve routines (should not have positive info)
    elseif occursin("getrs", func_str) || occursin("potrs", func_str) || 
           occursin("sytrs", func_str) || occursin("hetrs", func_str)
        return (:unexpected_error,
                "Unexpected positive return code from solve routine",
                "Solve routine $func returned info=$info which should not happen.")
    
    # General eigenvalue problem
    elseif occursin("ggev", func_str) || occursin("gges", func_str)
        if info <= size
            return (:convergence_failure,
                    "QZ iteration failed",
                    "The QZ iteration failed to compute all eigenvalues. Elements 1:$(info-1) converged.")
        else
            return (:unexpected_error,
                    "Unexpected error in generalized eigenvalue problem",
                    "Info value $info is unexpected for $func.")
        end
    
    # LDLT factorization
    elseif occursin("ldlt", func_str)
        return (:singular_matrix,
                "Matrix is singular",
                "The $(info)-th pivot is zero. The factorization has been completed but division will produce infinity.")
    
    # Default case
    else
        return (:unknown_error,
                "Unknown positive return code",
                "Function $func returned info=$info. Consult LAPACK documentation for details.")
    end
end

"""
    log_blas_info(func::Symbol, info::Integer, verbose::LinearVerbosity; 
                  extra_context::Dict{Symbol,Any} = Dict())

Log BLAS/LAPACK return code information with appropriate verbosity level.
"""
function log_blas_info(func::Symbol, info::Integer, verbose::LinearVerbosity;
                       extra_context::Dict{Symbol,Any} = Dict())
    category, message, details = interpret_blas_code(func, info)
    
    # Determine appropriate verbosity field based on category
    verbosity_field = if category in [:singular_matrix, :not_positive_definite, :convergence_failure]
        verbose.numerical.blas_errors
    elseif category == :invalid_argument
        verbose.error_control.blas_invalid_args
    else
        verbose.numerical.blas_info
    end
    
    # Build structured message components
    msg_main = "BLAS/LAPACK $func: $message"
    msg_details = !isempty(details) ? details : nothing
    msg_info = info
    
    # Build complete message with all details
    full_msg = if !isempty(extra_context) || msg_details !== nothing
        parts = String[msg_main]
        if msg_details !== nothing
            push!(parts, "Details: $msg_details")
        end
        push!(parts, "Return code (info): $msg_info")
        if !isempty(extra_context)
            for (key, value) in extra_context
                push!(parts, "$key: $value")
            end
        end
        join(parts, "\n  ")
    else
        "$msg_main (info=$msg_info)"
    end
    
    # Use proper @SciMLMessage syntax
    @SciMLMessage(full_msg, verbosity_field, :blas_return_code, :numerical)
end

"""
    check_and_log_lapack_result(func::Symbol, result, verbose::LinearVerbosity;
                                extra_context::Dict{Symbol,Any} = Dict())

Check LAPACK operation result and log appropriately based on verbosity settings.
Returns true if operation was successful, false otherwise.
"""
function check_and_log_lapack_result(func::Symbol, result, verbose::LinearVerbosity;
                                     extra_context::Dict{Symbol,Any} = Dict())
    # Extract info code from result
    info = if isa(result, Tuple) && length(result) >= 3
        # Standard LAPACK return format: (A, ipiv, info, ...)
        result[3]
    elseif isa(result, LinearAlgebra.Factorization) && hasfield(typeof(result), :info)
        result.info
    else
        0  # Assume success if we can't find info
    end
    
    if info != 0
        log_blas_info(func, info, verbose; extra_context=extra_context)
    elseif verbosity_to_int(verbose.numerical.blas_success) > 0
        success_msg = "BLAS/LAPACK $func completed successfully"
        @SciMLMessage(success_msg, verbose.numerical.blas_success, :blas_success, :numerical)
    end
    
    return info == 0
end

# Extended information for specific BLAS operations
"""
    get_blas_operation_info(func::Symbol, A, b, verbose::LinearVerbosity)

Get additional information about a BLAS operation for enhanced logging.
Condition number is computed based on the condition_number verbosity setting.
"""
function get_blas_operation_info(func::Symbol, A, b, verbose::LinearVerbosity)
    info = Dict{Symbol,Any}()
    
    # Matrix properties
    info[:matrix_size] = size(A)
    info[:matrix_type] = typeof(A)
    info[:element_type] = eltype(A)
    
    # Condition number (based on verbosity setting)
    should_compute_cond = verbosity_to_int(verbose.numerical.condition_number) > 0
    if should_compute_cond && size(A, 1) == size(A, 2)
        try
            cond_num = cond(A)
            info[:condition_number] = cond_num
            
            # Log the condition number if enabled  
            cond_msg = "Matrix condition number: $(round(cond_num, sigdigits=4)) for $(size(A, 1))×$(size(A, 2)) matrix in $func"
            @SciMLMessage(cond_msg, verbose.numerical.condition_number, :condition_number, :numerical)
        catch
            # Skip if condition number computation fails
        end
    end
    
    # RHS properties if provided
    if b !== nothing
        info[:rhs_size] = size(b)
        info[:rhs_type] = typeof(b)
    end
    
    # Memory usage estimate
    mem_bytes = prod(size(A)) * sizeof(eltype(A))
    info[:memory_usage_MB] = round(mem_bytes / 1024^2, digits=2)
    
    return info
end

