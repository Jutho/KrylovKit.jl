@testset "GKL - svdsolve full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (n, n))
            alg = GKL(; orth=orth, krylovdim=2 * n, maxiter=1, tol=tolerance(T))
            S, lvecs, rvecs, info = @constinferred svdsolve(A, A[:, 1], n, :LR, alg)

            @test S ≈ svdvals(A)

            U = stack(lvecs)
            V = stack(rvecs)
            @test U' * U ≈ I
            @test V' * V ≈ I
            @test A * V ≈ U * Diagonal(S)
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (n, n))
        v = MinimalVec{IP}(rand(T, (n,)))
        alg = GKL(; krylovdim=2 * n, maxiter=1, tol=tolerance(T))
        S, lvecs, rvecs, info = @constinferred svdsolve(wrapop(A), v, n, :LR, alg)

        @test S ≈ svdvals(A)

        U = stack(unwrap, lvecs)
        V = stack(unwrap, rvecs)
        @test U' * U ≈ I
        @test V' * V ≈ I
        @test A * V ≈ U * Diagonal(S)
    end

    @testset "MixedVec" begin
        T = ComplexF64
        A = rand(T, (n, n))
        v = MinimalVec{false}(rand(T, (n,)))
        alg = GKL(; krylovdim=2 * n, maxiter=1, tol=tolerance(T))
        S, lvecs, rvecs, info = @constinferred svdsolve(wrapop2(A), v, n, :LR, alg)

        @test S ≈ svdvals(A)

        U = stack(unwrap, lvecs)
        V = stack(unwrap, rvecs)
        @test U' * U ≈ I
        @test V' * V ≈ I
        @test A * V ≈ U * Diagonal(S)
    end
end

@testset "GKL - svdsolve iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (2 * N, N))
            v = rand(T, (2 * N,))
            n₁ = div(n, 2)
            alg = GKL(; orth=orth, krylovdim=n, maxiter=10, tol=tolerance(T), eager=true)
            S, lvecs, rvecs, info = @constinferred svdsolve(A, v, n₁, :LR,
                                                            alg)

            l = info.converged
            @test S[1:l] ≈ svdvals(A)[1:l]

            U = stack(lvecs)
            V = stack(rvecs)
            @test U[:, 1:l]' * U[:, 1:l] ≈ I
            @test V[:, 1:l]' * V[:, 1:l] ≈ I

            R = stack(info.residual)
            @test A' * U ≈ V * Diagonal(S)
            @test A * V ≈ U * Diagonal(S) + R
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (2 * N, N))
        v = MinimalVec{IP}(rand(T, (2 * N,)))
        n₁ = div(n, 2)
        alg = GKL(; krylovdim=n, maxiter=10, tol=tolerance(T), eager=true)
        S, lvecs, rvecs, info = @constinferred svdsolve(wrapop(A), v, n₁, :LR,
                                                        alg)

        l = info.converged
        @test S[1:l] ≈ svdvals(A)[1:l]

        U = stack(unwrap, lvecs)
        V = stack(unwrap, rvecs)
        @test U[:, 1:l]' * U[:, 1:l] ≈ I
        @test V[:, 1:l]' * V[:, 1:l] ≈ I

        R = stack(unwrap, info.residual)
        @test A' * U ≈ V * Diagonal(S)
        @test A * V ≈ U * Diagonal(S) + R
    end

    @testset "MixedVec" begin
        T = ComplexF64
        A = rand(T, (2 * N, N))
        v = MinimalVec{false}(rand(T, (2 * N,)))
        n₁ = div(n, 2)
        alg = GKL(; krylovdim=n, maxiter=10, tol=tolerance(T), eager=true)
        S, lvecs, rvecs, info = @constinferred svdsolve(wrapop2(A), v, n₁, :LR,
                                                        alg)

        l = info.converged
        @test S[1:l] ≈ svdvals(A)[1:l]

        U = stack(unwrap, lvecs)
        V = stack(unwrap, rvecs)
        @test U[:, 1:l]' * U[:, 1:l] ≈ I
        @test V[:, 1:l]' * V[:, 1:l] ≈ I

        R = stack(unwrap, info.residual)
        @test A' * U ≈ V * Diagonal(S)
        @test A * V ≈ U * Diagonal(S) + R
    end
end
