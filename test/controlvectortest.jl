using SparseArrays, LinearSolve

A = sparse(rand(3,3));
b=rand(3);
prob = LinearProblem(A, b);

#check without control Vector
u=solve(prob,UMFPACKFactorization()).u

#check plugging in a control vector
controlv=SparseArrays.UMFPACK.get_umfpack_control(Float64,Int64)
u=solve(prob,UMFPACKFactorization(control=controlv)).u