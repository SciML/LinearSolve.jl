module LinearSolveCUDSSExt

using LinearSolve: LinearSolve, cudss_loaded
using CUDSS

LinearSolve.cudss_loaded(A::CUDSS.CUDA.CUSPARSE.CuSparseMatrixCSR) = true
LinearSolve.cudss_loaded(A::CUDSS.CUDA.CUSPARSE.CuSparseMatrixCSC) = true

end
