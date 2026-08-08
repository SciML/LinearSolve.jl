using LinearSolve, LinearAlgebra, RecursiveFactorization, Test, Random

Random.seed!(1161)

# `RecursiveFactorization.lu!(A, ipiv, Val(false), ...)` leaves a
# caller-supplied `ipiv` undefined on Julia >= 1.8, so the pivot buffer reaching
# the back-solve holds whatever the allocator left behind. Seeding it with a
# valid-but-wrong permutation makes that observable as a wrong answer instead of
# as an out-of-bounds read.
poison(n) = fill(LinearAlgebra.BlasInt(n), n)

@testset "RFLU pivot = Val(false) ignores stale pivots" begin
    for n in (5, 20, 64)
        # Diagonally dominant, so the unpivoted factorization is stable and the
        # identity permutation is the right answer.
        A = rand(n, n) + n * I
        b = rand(n)
        xref = A \ b

        cache = init(LinearProblem(copy(A), copy(b)), RFLUFactorization(pivot = Val(false)))
        cache.cacheval = (cache.cacheval[1], poison(n))
        sol = solve!(cache)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ xref rtol = 1.0e-10
        @test cache.cacheval[2] == 1:n

        cache32 = init(
            LinearProblem(copy(A), copy(b)), RF32MixedLUFactorization(pivot = Val(false))
        )
        cv = cache32.cacheval
        cache32.cacheval = (cv[1], poison(n), cv[3], cv[4], cv[5])
        sol32 = solve!(cache32)
        @test SciMLBase.successful_retcode(sol32)
        @test sol32.u ≈ xref rtol = 1.0e-4
        @test cache32.cacheval[2] == 1:n
    end
end
