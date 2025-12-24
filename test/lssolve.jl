# Test LSMR complete
@testset "LSMR small problem ($mode)" for mode in (:vector, :inplace, :outplace, :mixed)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (2 * n, n))
        U, S, V = svd(A)
        invS = 1 ./ S
        S[end] = 0 # make rank deficient
        invS[end] = 0 # choose minimal norm solution
        A = U * Diagonal(S) * V'

        b = rand(T, 2 * n)
        tol = tol = 10 * n * eps(real(T))
        x, info = @constinferred lssolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            maxiter = 3, krylovdim = 1, verbosity = SILENT_LEVEL
        ) # no reorthogonalization
        r = b - A * unwrapvec(x)
        @test unwrapvec(info.residual) ≈ r
        @test info.normres ≈ norm(A' * r)
        @test info.converged == 0
        @test_logs lssolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)); maxiter = 3,
            verbosity = SILENT_LEVEL
        )
        @test_logs (:warn,) lssolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)); maxiter = 3,
            verbosity = WARN_LEVEL
        )
        @test_logs (:info,) (:warn,) lssolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            maxiter = 3, verbosity = STARTSTOP_LEVEL
        )

        alg = LSMR(; maxiter = n, tol = tol, verbosity = SILENT_LEVEL, krylovdim = n)
        # reorthogonalisation is essential here to converge in exactly n iterations
        x, info = @constinferred lssolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), alg)

        @test info.converged > 0
        @test abs(inner(V[:, end], unwrapvec(x))) < alg.tol
        @test unwrapvec(x) ≈ V * Diagonal(invS) * U' * b
        @test_logs lssolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), alg)
        alg = LSMR(; maxiter = 2 * n, tol = tol, verbosity = WARN_LEVEL)
        @test_logs lssolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), alg)
        alg = LSMR(; maxiter = 2 * n, tol = tol, verbosity = STARTSTOP_LEVEL)
        @test_logs (:info,) (:info,) lssolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            alg
        )
        alg = LSMR(; maxiter = 2 * n, tol = tol, verbosity = EACHITERATION_LEVEL)
        @test_logs min_level = Logging.Warn lssolve(
            wrapop(A, Val(mode)),
            wrapvec(b, Val(mode)),
            alg
        )

        λ = rand(real(T))
        alg = LSMR(; maxiter = n, tol = tol, verbosity = SILENT_LEVEL, krylovdim = n)
        # reorthogonalisation is essential here to converge in exactly n iterations
        x, info = @constinferred lssolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), alg,
            λ
        )

        r = b - A * unwrapvec(x)
        @test info.converged > 0
        @test A' * r ≈ λ^2 * unwrapvec(x) atol = 2 * tol

        if mode == :vector && T <: Complex
            A = rand(T, (2 * n, n)) .- one(T) / 2
            B = rand(T, (2 * n, n)) .- one(T) / 2
            f = buildrealmap(A, B)
            # the effective linear problem has twice the size, so 4n x 2n
            alg = LSMR(; maxiter = 2 * n, tol = tol, verbosity = SILENT_LEVEL, krylovdim = 2 * n)
            xr, infor = @constinferred reallssolve(f, b, alg)
            @test infor.converged > 0
            y = (A * xr + B * conj(xr))
            @test b ≈ y + infor.residual
            @test (A' * b + conj(B' * b)) ≈ (A' * y + conj(B' * y))
        end
    end
end
@testset "LSMR large problem ($mode)" for mode in (:vector, :inplace, :outplace, :mixed)
    scalartypes = mode === :vector ? (Float64, ComplexF64) : (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (2 * N, N)) .- (one(T) / 2)
        b = rand(T, 2 * N) .- (one(T) / 2)

        tol = 10 * N * eps(real(T))
        x, info = @constinferred lssolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            maxiter = N, tol = tol, verbosity = SILENT_LEVEL,
            krylovdim = 5
        )

        r = b - A * unwrapvec(x)
        @test info.converged > 0
        @test norm(A' * r) < 5 * tol # there seems to be some loss of precision in the computation of the convergence measure

        if mode == :vector && T <: Complex
            A = rand(T, (2 * N, N)) .- one(T) / 2
            B = rand(T, (2 * N, N)) .- one(T) / 2
            f = buildrealmap(A, B)
            alg = LSMR(; maxiter = N, tol = tol, verbosity = SILENT_LEVEL, krylovdim = 5)
            xr, infor = @constinferred reallssolve(f, b, alg)
            @test infor.converged > 0
            y = (A * xr + B * conj(xr))
            @test b ≈ y + infor.residual
            @test (A' * b + conj(B' * b)) ≈ (A' * y + conj(B' * y))
        end
    end
end
