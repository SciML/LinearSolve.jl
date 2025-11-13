"""
Type-stable container for BLAS operation information.

Uses sentinel values for optional fields to maintain type stability:

  - condition_number: -Inf means not computed
  - rhs_length: 0 means not applicable
  - rhs_type: "" means not applicable
"""
struct BlasOperationInfo
    matrix_size::Tuple{Int, Int}
    matrix_type::String
    element_type::String
    condition_number::Float64  # -Inf means not computed
    rhs_length::Int  # 0 means not applicable
    rhs_type::String  # "" means not applicable
    memory_usage_MB::Float64
end

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
        return (:convergence_failure,
            "Generalized eigenvalue computation failed",
            "The algorithm failed to compute eigenvalues (info=$info). This may indicate QZ iteration failure or other numerical issues.")

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
Format BlasOperationInfo fields into human-readable strings.

Type-stable implementation using concrete struct fields instead of Dict iteration.
"""
function _format_blas_context(op_info::BlasOperationInfo)
    parts = String[]

    # Always-present fields
    push!(parts, "Matrix size: $(op_info.matrix_size)")
    push!(parts, "Matrix type: $(op_info.matrix_type)")
    push!(parts, "Element type: $(op_info.element_type)")
    push!(parts, "Memory usage: $(op_info.memory_usage_MB) MB")

    # Optional fields - check for sentinel values
    if !isinf(op_info.condition_number)
        push!(parts, "Condition number: $(round(op_info.condition_number, sigdigits=4))")
    end

    if op_info.rhs_length > 0
        push!(parts, "RHS length: $(op_info.rhs_length)")
    end

    if !isempty(op_info.rhs_type)
        push!(parts, "RHS type: $(op_info.rhs_type)")
    end

    return parts
end

"""
    blas_info_msg(func::Symbol, info::Integer;
                  extra_context::BlasOperationInfo = BlasOperationInfo(
                      (0, 0), "", "", -Inf, 0, "", 0.0))

Log BLAS/LAPACK return code information with appropriate verbosity level.
"""
function blas_info_msg(func::Symbol, info::Integer;
        extra_context::BlasOperationInfo = BlasOperationInfo(
            (0, 0), "", "", -Inf, 0, "", 0.0))
    category, message, details = interpret_blas_code(func, info)

    verbosity_field = if category in [
        :singular_matrix, :not_positive_definite, :convergence_failure]
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
    # Check if extra_context has any non-sentinel values
    has_extra_context = extra_context.matrix_size != (0, 0)

    full_msg = if has_extra_context || msg_details !== nothing
        parts = String[msg_main]
        if msg_details !== nothing
            push!(parts, "Details: $msg_details")
        end
        push!(parts, "Return code (info): $msg_info")
        if has_extra_context
            # Type-stable formatting using struct fields
            append!(parts, _format_blas_context(extra_context))
        end
        join(parts, "\n  ")
    else
        "$msg_main (info=$msg_info)"
    end

    verbosity_field, full_msg
end

function get_blas_operation_info(func::Symbol, A, b; condition = false)
    # Matrix properties (always present)
    matrix_size = size(A)
    matrix_type = string(typeof(A))
    element_type = string(eltype(A))

    # Memory usage estimate (always present)
    mem_bytes = prod(matrix_size) * sizeof(eltype(A))
    memory_usage_MB = round(mem_bytes / 1024^2, digits = 2)

    # Condition number (optional - use -Inf as sentinel)
    condition_number = if condition && matrix_size[1] == matrix_size[2]
        try
            cond(A)
        catch
            -Inf
        end
    else
        -Inf
    end

    # RHS properties (optional - use 0 and "" as sentinels)
    rhs_length = b !== nothing ? length(b) : 0
    rhs_type = b !== nothing ? string(typeof(b)) : ""

    return BlasOperationInfo(
        matrix_size,
        matrix_type,
        element_type,
        condition_number,
        rhs_length,
        rhs_type,
        memory_usage_MB
    )
end
