## Default algorithm

function SciMLBase.solve(cache::LinearCache, alg::Nothing,
                         args...; kwargs...)
    @unpack A = cache
    if A isa DiffEqArrayOperator
        A = A.A
    end

    if A isa Matrix
        if ArrayInterface.can_setindex(cache.b) && (size(A,1) <= 100 ||
                                              (isopenblas() && size(A,1) <= 500)
                                             )
            alg = GenericFactorization(;fact_alg=RecursiveFactorization.lu!)
            SciMLBase.solve(cache, alg, args...; kwargs...)
        else
            alg = LUFactorization()
            SciMLBase.solve(cache, alg, args...; kwargs...)
        end
    elseif A isa Tridiagonal
        alg = GenericFactorization(;fact_alg=lu!)
        SciMLBase.solve(cache, alg, args...; kwargs...)
    elseif A isa SymTridiagonal
        alg = GenericFactorization(;fact_alg=ldlt!)
        SciMLBase.solve(cache, alg, args...; kwargs...)
    elseif A isa SparseMatrixCSC
        alg = LUFactorization()
        SciMLBase.solve(cache, alg, args...; kwargs...)
    elseif ArrayInterface.isstructured(A)
        alg = GenericFactorization()
        SciMLBase.solve(cache, alg, args...; kwargs...)
    elseif !(A isa AbstractDiffEqOperator)
        alg = QRFactorization()
        SciMLBase.solve(cache, alg, args...; kwargs...)
    else
        alg = IterativeSolversJL_GMRES()
        SciMLBase.solve(cache, alg, args...; kwargs...)
    end
end
