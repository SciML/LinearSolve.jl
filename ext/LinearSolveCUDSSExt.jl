module LinearSolveCUDSSExt

using LinearSolve: LinearSolve, cudss_loaded
using CUDSS

LinearSolve.cudss_loaded(A::CUDSS.CUDA.CUSPARSE.CuSparseMatrixCSR) = true

end
