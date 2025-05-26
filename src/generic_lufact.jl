# From LinearAlgebra.lu.jl
# Modified to be non-allocating
@static if VERSION < v"1.11"
    function generic_lufact!(A::AbstractMatrix{T}, pivot::Union{RowMaximum,NoPivot,RowNonZero} = LinearAlgebra.lupivottype(T),
                            ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...));
                            check::Bool = true, allowsingular::Bool = false) where {T}
        check && LinearAlgebra.LAPACK.chkfinite(A)
        # Extract values
        m, n = size(A)
        minmn = min(m,n)

        # Initialize variables
        info = 0
        
        @inbounds begin
            for k = 1:minmn
                # find index max
                kp = k
                if pivot === LinearAlgebra.RowMaximum() && k < m
                    amax = abs(A[k, k])
                    for i = k+1:m
                        absi = abs(A[i,k])
                        if absi > amax
                            kp = i
                            amax = absi
                        end
                    end
                elseif pivot === LinearAlgebra.RowNonZero()
                    for i = k:m
                        if !iszero(A[i,k])
                            kp = i
                            break
                        end
                    end
                end
                ipiv[k] = kp
                if !iszero(A[kp,k])
                    if k != kp
                        # Interchange
                        for i = 1:n
                            tmp = A[k,i]
                            A[k,i] = A[kp,i]
                            A[kp,i] = tmp
                        end
                    end
                    # Scale first column
                    Akkinv = inv(A[k,k])
                    for i = k+1:m
                        A[i,k] *= Akkinv
                    end
                elseif info == 0
                    info = k
                end
                # Update the rest
                for j = k+1:n
                    for i = k+1:m
                        A[i,j] -= A[i,k]*A[k,j]
                    end
                end
            end
        end
        check && LinearAlgebra.checknonsingular(info, pivot)
        return LinearAlgebra.LU{T,typeof(A),typeof(ipiv)}(A, ipiv, convert(LinearAlgebra.BlasInt, info))
    end
elseif VERSION < v"1.13"
    function generic_lufact!(A::AbstractMatrix{T}, pivot::Union{RowMaximum,NoPivot,RowNonZero} = LinearAlgebra.lupivottype(T),
                         ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...));
                         check::Bool = true, allowsingular::Bool = false) where {T}
        check && LAPACK.chkfinite(A)
        # Extract values
        m, n = size(A)
        minmn = min(m,n)

        # Initialize variables
        info = 0
        
        @inbounds begin
            for k = 1:minmn
                # find index max
                kp = k
                if pivot === LinearAlgebra.RowMaximum() && k < m
                    amax = abs(A[k, k])
                    for i = k+1:m
                        absi = abs(A[i,k])
                        if absi > amax
                            kp = i
                            amax = absi
                        end
                    end
                elseif pivot === LinearAlgebra.RowNonZero()
                    for i = k:m
                        if !iszero(A[i,k])
                            kp = i
                            break
                        end
                    end
                end
                ipiv[k] = kp
                if !iszero(A[kp,k])
                    if k != kp
                        # Interchange
                        for i = 1:n
                            tmp = A[k,i]
                            A[k,i] = A[kp,i]
                            A[kp,i] = tmp
                        end
                    end
                    # Scale first column
                    Akkinv = inv(A[k,k])
                    for i = k+1:m
                        A[i,k] *= Akkinv
                    end
                elseif info == 0
                    info = k
                end
                # Update the rest
                for j = k+1:n
                    for i = k+1:m
                        A[i,j] -= A[i,k]*A[k,j]
                    end
                end
            end
        end
        if pivot === LinearAlgebra.NoPivot()
            # Use a negative value to distinguish a failed factorization (zero in pivot
            # position during unpivoted LU) from a valid but rank-deficient factorization
            info = -info
        end
        check && LinearAlgebra._check_lu_success(info, allowsingular)
        return LinearAlgebra.LU{T,typeof(A),typeof(ipiv)}(A, ipiv, convert(LinearAlgebra.BlasInt, info))
    end
else    
     generic_lufact!(args...; kwargs...) = LinearAlgebra.generic_lufact!(args...; kwargs...)
end