module LinearSolveNNlibExt

using LinearAlgebra, LinearSolve, NNlib
import LinearSolve: SimpleGMRESCache, SimpleGMRES, OperatorAssumptions, _no_preconditioner,
    _init_cacheval, _norm2, LinearCache
import UnPack: @unpack

function SciMLBase.solve!(cache::SimpleGMRESCache{true, T}, lincache::LinearCache) where {T}
    @unpack M, N, maxiters, ϵ, Q, H, x, r, βe₁, A, b, β, abstol, blocksize = cache
    res_norm = β

    # FIXME: The performance for this is quite bad when compared to the KrylovJL_GMRES
    #        version
    for _ in 1:((maxiters ÷ M) + 1)
        for j in 1:M
            Qⱼ₊₁ = @view(Q[:, j + 1, :])
            mul!(vec(Qⱼ₊₁), A, vec(@view(Q[:, j, :])))  # Q(:,j+1) <- A Q(:, j)
            for i in 1:j
                H[i, j, :] .= vec(sum(@view(Q[:, i, :]) .* Qⱼ₊₁; dims = 1))
                Qⱼ₊₁ .-= H[i:i, j, :] .* @view(Q[:, i, :])
            end
            H[j + 1, j, :] .= vec(_norm2(Qⱼ₊₁, 1))
            Qⱼ₊₁ ./= H[j + 1, j:j, :]

            # FIXME: Figure out a way to avoid the allocation
            # Using views doesn't work very well with LinearSolve
            y = similar(b, j, 1, size(H, 3))
            for bidx in 1:size(y, 3)
                y[:, :, bidx] .= @view(H[1:(j + 1), 1:j, bidx]) \ @view(βe₁[1:(j + 1), bidx])
            end

            # Update the solution
            batched_mul!(reshape(x, blocksize, 1, :), @view(Q[:, 1:j, :]), y)
            mul!(r, A, x, T(-1), T(0))
            r .+= b
            res_norm = _norm2(reshape(r, blocksize, :), 1)

            if maximum(res_norm) < abstol
                return SciMLBase.build_linear_solution(lincache.alg, x, r, lincache;
                    retcode = ReturnCode.Success)
            end
        end

        # Restart
        Q[:, 1, :] = reshape(r, blocksize, :) ./ res_norm
        fill!(H, zero(T))
    end

    return SciMLBase.build_linear_solution(lincache.alg, x, r, lincache;
        retcode = ReturnCode.MaxIters)
end

end
