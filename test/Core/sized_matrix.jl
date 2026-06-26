using LinearSolve, StaticArrays, RecursiveFactorization, LinearAlgebra, Test

# A `SizedMatrix` (StaticArrays) is an in-place, dense matrix whose
# `ArrayInterface.lu_instance` returns a `StaticArrays.LU` (fields `L`/`U`/`p`),
# not a `LinearAlgebra.LU`. `init_cacheval` for `GenericLUFactorization` /
# `RFLUFactorization` must not assume `lu_instance` yields a `LinearAlgebra.LU`
# and reach for its `factors`/`info` fields. Regression for
# SciML/LinearSolve.jl#1056: `init` used to throw
# `type StaticArrays.LU has no field factors`, which broke the downstream
# OrdinaryDiffEq "Sized Matrix Tests" (`Rodas4` over a `SizedMatrix` state, whose
# default solver initializes the `GenericLU`/`RFLU` cachevals).

A = SizedMatrix{9, 9}(rand(9, 9) + 9I)
b = SizedVector{9}(rand(9))

# `init` is where `init_cacheval` runs; this is the call that used to throw.
@testset "init does not assume LinearAlgebra.LU for $(typeof(alg))" for alg in (
        LinearSolve.GenericLUFactorization(), LinearSolve.RFLUFactorization(),
    )
    @test init(LinearProblem(A, b), alg) isa LinearSolve.LinearCache
end

@testset "default init/solve LinearProblem(SizedMatrix)" begin
    cache = init(LinearProblem(A, b))
    sol = solve!(cache)
    @test sol.retcode == ReturnCode.Success
end
