@testset "GKL - svdsolve full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (n,n))
            alg = GKL(orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            S, lvecs, rvecs, info = @inferred svdsolve(A, A[:,1], n, :LR, alg)

            @test S ≈ svdvals(A)

            U = hcat(lvecs...)
            V = hcat(rvecs...)
            @test U'*U ≈ I
            @test V'*V ≈ I
            @test A*V ≈ U*Diagonal(S)
        end
    end
end

@testset "GKL - svdsolve iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (2*N,N))
            v = rand(T, (2*N,))
            n₁ = div(n, 2)
            alg = GKL(orth = orth, krylovdim = n, maxiter = 10, tol = 10*n*eps(real(T)))
            S, lvecs, rvecs, info = @inferred svdsolve(A, v, n₁, :LR, alg)

            l = info.converged
            @test S[1:l] ≈ svdvals(A)[1:l]

            U = hcat(lvecs...)
            V = hcat(rvecs...)
            @test U[:,1:l]'*U[:,1:l] ≈ I
            @test V[:,1:l]'*V[:,1:l] ≈ I

            R = hcat(info.residual...)
            @test A' * U ≈ V * Diagonal(S)
            @test A * V ≈ U * Diagonal(S) + R
        end
    end
end
