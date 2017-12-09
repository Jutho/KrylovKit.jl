# Test GMRES complete
@testset "GMRES full factorization" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        A = rand(T,(n,n)).-one(T)/2
        b = rand(T,n)
        alg = GMRES(krylovdim = n, maxiter = 2, reltol = 2*n*eps(real(T)))
        x, hist = @inferred linsolve(A, b, alg)
        @test hist.converged > 0
        @test b ≈ A*x

        A = rand(T,(n,n))
        α₀ = rand(T)
        α₁ = -rand(T)
        x, hist = @inferred linsolve(A, b, alg, α₀, α₁)
        @test b ≈ (α₀*I+α₁*A)*x
        @test hist.converged > 0
    end
end

# Test GMRES with restart
@testset "GMRES with restarts" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        A = rand(T,(N,N)).-one(T)/2
        A = I-T(9/10)*A/maximum(abs, eigvals(A))
        b = rand(T,N)
        alg = GMRES(krylovdim = 3*n, maxiter = 50, reltol = 10*N*eps(real(T)))
        x, hist = @inferred linsolve(A, b, alg)
        @test b ≈ A*x + hist.residual

        A = rand(T,(N,N)).-one(T)/2
        α₀ = maximum(abs, eigvals(A))
        α₁ = -rand(T)
        α₁ *= T(9)/T(10)/abs(α₁)
        x, hist = @inferred linsolve(A, b, alg, α₀, α₁)
        @test b ≈ (α₀*I+α₁*A)*x + hist.residual
    end
end
