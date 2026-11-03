using LinearSolve, LinearAlgebra, SparseArrays, Zygote, Random, Test

@testset "QR adjoints for sparse and wide matrices" begin
    rng = MersenneTwister(1362)
    for T in (Float64, ComplexF64), (m, n) in ((6, 6), (9, 4), (4, 9)),
            sparse_input in (false, true)
        A = randn(rng, T, m, n)
        sparse_input && (A = sparse(A))
        b = randn(rng, T, m)
        dy = randn(rng, T, n)
        algorithms = !sparse_input && m < n ? (QRFactorization(ColumnNorm()),) :
            (QRFactorization(), QRFactorization(ColumnNorm()))
        for alg in algorithms
            cache = init(LinearProblem(copy(A), copy(b)), alg)
            forward = copy(solve!(cache).u)
            F = sparse_input ? qr(A) : qr(copy(A), ColumnNorm())
            @test forward ≈ F \ b
            # Applying the forward map to a basis independently determines its adjoint,
            # including SPQR's choice of basic variables for a wide matrix.
            forward_map = F \ Matrix{T}(I, m, m)
            expected = forward_map' * dy
            @test LinearSolve._adjoint_solve(cache, dy) ≈ expected
            @test LinearSolve._adjoint_solve(cache, hcat(dy, 2dy)) ≈
                hcat(expected, 2expected)
            @test solve!(cache).u == forward
            db, = Zygote.gradient(
                y -> real(dot(dy, solve(LinearProblem(copy(A), y), alg).u)), b
            )
            @test db ≈ expected
            if m == n
                cache.b = copy(dy)
                @test solve!(cache; adjoint = true).u ≈ expected
            end
        end
    end

    for (m, n, sparse_input) in ((4, 9, false), (1001, 12, true))
        A = randn(rng, m, n)
        sparse_input && (A = sparse(A))
        b = randn(rng, m)
        dy = randn(rng, n)
        cache = init(LinearProblem(copy(A), copy(b)))
        forward = copy(solve!(cache).u)
        @test forward ≈ A \ b
        db, = Zygote.gradient(y -> dot(dy, solve(LinearProblem(copy(A), y)).u), b)
        @test db ≈ pinv(Matrix(A))' * dy
        @test LinearSolve._adjoint_solve(cache, hcat(dy, 2dy)) ≈
            pinv(Matrix(A))' * hcat(dy, 2dy)
        @test solve!(cache).u == forward
    end
end
