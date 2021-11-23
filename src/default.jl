## Default algorithm

struct DefaultLinSolve{Ta} <: SciMLLinearSolveAlgorithm
    linalg::Ta
    ifopenblas::Union{Bool,Nothing}
    isset::Bool # true => do nothing, false => find alg
end

DefaultLinSolve() = DefaultLinSolve(nothing, nothing, true)

function isopenblas()
    @static if VERSION < v"1.7beta"
        blas = BLAS.vendor()
        blas == :openblas64 || blas == :openblas
    else
        occursin("openblas", BLAS.get_config().loaded_libs[1].libname)
    end
end

function SciMLBase.solve(cache::LinearCache, alg::DefaultLinSolve,
                         args...; kwargs...)
    @unpack A = cache

    if alg.isset
      linalg = if A isa Matrix
          if ArrayInterface.can_setindex(x) && (size(A,1) <= 100 ||
                                                (p.openblas && size(A,1) <= 500)
                                               )
              DefaultFactorization(;fact_alg=:(RecursiveFactorization.lu!))
          else
              LUFactorization()
          end
      elseif A isa Union{Tridiagonal,} # ForwardSensitivityJacobian
          DefaultFactorization(;fact_alg=lu!)
      elseif A isa Union{SymTridiagonal}
          DefaultFactorization(;fact_alg=ldlt!)
      elseif A isa SparseMatrixCSC
          LUFactorization()
      elseif ArrayInterface.isstructured(A)
          DefaultFactorization() # change fact_alg=LinearAlgebra.factorize
      elseif !(A isa AbstractDiffEqOperator)
          QRFactorization()
      else
          IterativeSolversJL_GMRES()
      end

      @set! alg.linalg = linalg
    end

    SciMLBase.solve(cache, alg.linalg, args...; kwargs...)
end
