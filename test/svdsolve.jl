@testset "Lanczos - svdsolve full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (2*n,n))
            S, lvecs, rvecs, info = @inferred svdsolve(A, n; krylovdim = 3*n, maxiter = 1, tol = 10*n*eps(real(T)))

            @test info.converged == n
            @test S ≈ svdvals(A)

            U = hcat(lvecs...)
            V = hcat(rvecs...)
            @test U'*U ≈ I
            @test V'*V ≈ I
            @test A*V ≈ U*Diagonal(S)
        end
    end
end

@testset "Lanczos - svdsolve iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (N,2*N))
            v = rand(T, (2*N,))
            w = rand(T, (N,))
            n₁ = div(n, 2)
            alg = Lanczos(orth; krylovdim = n, maxiter = 10, tol = 10*n*eps(real(T)))
            S, lvecs, rvecs, info = @inferred svdsolve(A, v, w, n₁, alg)

            l = info.converged
            @test S[1:l] ≈ svdvals(A)[1:l]

            U = hcat(lvecs...)
            V = hcat(rvecs...)
            @test U[:,1:l]'*U[:,1:l] ≈ I
            @test V[:,1:l]'*V[:,1:l] ≈ I

            R1 = sqrt(2*one(T))*hcat(map(first, info.residual)...)
            R2 = sqrt(2*one(T))*hcat(map(last, info.residual)...)
            @test A' * U ≈ V * Diagonal(S) + R1
            @test A * V ≈ U * Diagonal(S) + R2
        end
    end
end
