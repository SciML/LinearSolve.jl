
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
    blas_info_msg(func::Symbol, info::Integer, verbose::LinearVerbosity;
                  extra_context::Dict{Symbol,Any} = Dict())

Log BLAS/LAPACK return code information with appropriate verbosity level.
"""
function blas_info_msg(func::Symbol, info::Integer;
        extra_context::Dict{Symbol, Any} = Dict())
    category, message, details = interpret_blas_code(func, info)

    verbosity_field = if category in [:singular_matrix, :not_positive_definite, :convergence_failure]
        :blas_errors
    elseif category == :invalid_argument
        :blas_invalid_args
    else
        :blas_info
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

    verbosity_field, full_msg
end


function get_blas_operation_info(func::Symbol, A, b; condition = false)
    info = Dict{Symbol, Any}()

    # Matrix properties
    info[:matrix_size] = size(A)
    info[:matrix_type] = typeof(A)
    info[:element_type] = eltype(A)

    # Condition number (based on verbosity setting)
    if condition && size(A, 1) == size(A, 2)
        try
            cond_num = cond(A)
            info[:condition_number] = cond_num

            # Log the condition number if enabled  
            cond_msg = "Matrix condition number: $(round(cond_num, sigdigits=4)) for $(size(A, 1))×$(size(A, 2)) matrix in $func"

        catch
            # Skip if condition number computation fails
            info[:condition_number] = nothing
        end
    end

    # RHS properties if provided
    if b !== nothing
        info[:rhs_size] = size(b)
        info[:rhs_type] = typeof(b)
    end

    # Memory usage estimate
    mem_bytes = prod(size(A)) * sizeof(eltype(A))
    info[:memory_usage_MB] = round(mem_bytes / 1024^2, digits = 2)

    return info
end