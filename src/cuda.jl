struct GPUOffloadFactorization <: AbstractFactorization end

function SciMLBase.solve(cache::LinearCache, alg::GPUOffloadFactorization; kwargs...)
    if cache.isfresh
        fact = do_factorization(alg, CUDA.CuArray(cache.A), cache.b, cache.u)
        cache = set_cacheval(cache, fact)
    end

    copyto!(cache.u,cache.b)
    y = Array(ldiv!(cache.cacheval, CUDA.CuArray(cache.u)))
    SciMLBase.build_linear_solution(alg,y,nothing,cache)
end

function do_factorization(alg::GPUOffloadFactorization, A, b, u)
    A isa Union{AbstractMatrix,AbstractDiffEqOperator} ||
        error("LU is not defined for $(typeof(A))")

    if A isa AbstractDiffEqOperator
        A = A.A
    end
    fact = qr(CUDA.CuArray(A))
    return fact
end

function LinearAlgebra.ldiv!(x::CUDA.CuArray,_qr::CUDA.CUSOLVER.CuQR,b::CUDA.CuArray)
  _x = UpperTriangular(_qr.R) \ (_qr.Q' * reshape(b,length(b),1))
  x .= vec(_x)
  CUDA.unsafe_free!(_x)
  return x
end
# make `\` work
LinearAlgebra.ldiv!(F::CUDA.CUSOLVER.CuQR, b::CUDA.CuArray) = (x = similar(b); ldiv!(x, F, b); x)

export GPUOffloadFactorization
