# Test CG complete
@testset "CG small problem" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, (n, n))
        A = sqrt(A * A')
        b = rand(T, n)
        alg = CG(; maxiter=2n, tol=tolerance(T) * norm(b), verbosity=2) # because of loss of orthogonality, we choose maxiter = 2n
        x, info = @constinferred linsolve(A, b;
                                          ishermitian=true, isposdef=true, maxiter=2n,
                                          krylovdim=1, rtol=tolerance(T),
                                          verbosity=1)
        @test info.converged > 0
        @test b ≈ A * x
        x, info = @constinferred linsolve(A, b, x;
                                          ishermitian=true, isposdef=true, maxiter=2n,
                                          krylovdim=1, rtol=tolerance(T))
        @test info.numops == 1

        A = rand(T, (n, n))
        A = sqrt(A * A')
        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @constinferred linsolve(A, b, zerovector(b), alg, α₀, α₁)
        @test b ≈ (α₀ * I + α₁ * A) * x
        @test info.converged > 0
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (n, n))
        A += A'
        b = MinimalVec{IP}(rand(T, (n,)))
        alg = CG(; maxiter=2n, tol=tolerance(T) * norm(b), verbosity=2) # because of loss of orthogonality, we choose maxiter = 2n
        x, info = @constinferred linsolve(wrapop(A), b; ishermitian=true, isposdef=true,
                                          maxiter=2n, krylovdim=1, rtol=tolerance(T),
                                          verbosity=1)
        @test info.converged > 0
        @test unwrap(b) ≈ A * unwrap(x)
        x, info = @constinferred linsolve(wrapop(A), b, x; ishermitian=true, isposdef=true,
                                          maxiter=2n, krylovdim=1, rtol=tolerance(T))
        @test info.numops == 1

        A = rand(T, (n, n))
        A += A'
        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @constinferred linsolve(wrapop(A), b, zerovector(b), alg, α₀, α₁)
        @test unwrap(b) ≈ (α₀ * I + α₁ * A) * unwrap(x)
        @test info.converged > 0
    end
end

# Test CG complete
@testset "CG large problem" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, (N, N))
        A = sqrt(sqrt(A * A')) / N
        b = rand(T, N)
        x, info = @constinferred linsolve(A, b;
                                          isposdef=true, maxiter=1, krylovdim=N,
                                          rtol=tolerance(T))
        @test b ≈ A * x + info.residual

        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @constinferred linsolve(A, b, α₀, α₁;
                                          isposdef=true, maxiter=1, krylovdim=N,
                                          rtol=tolerance(T))
        @test b ≈ (α₀ * I + α₁ * A) * x + info.residual
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (N, N))
        A += A'
        b = MinimalVec{IP}(rand(T, (N,)))
        x, info = @constinferred linsolve(wrapop(A), b; isposdef=true, maxiter=1,
                                          krylovdim=N, rtol=tolerance(T))
        @test unwrap(b) ≈ A * unwrap(x) + unwrap(info.residual)

        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @constinferred linsolve(wrapop(A), b, zerovector(b), α₀, α₁;
                                          isposdef=true, maxiter=1, krylovdim=N,
                                          rtol=tolerance(T))
        @test unwrap(b) ≈ (α₀ * I + α₁ * A) * unwrap(x) + unwrap(info.residual)
    end
end

# Test GMRES complete
@testset "GMRES full factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, (n, n)) .- one(T) / 2
        b = rand(T, n)
        alg = GMRES(; krylovdim=n, maxiter=2, tol=tolerance(T) * norm(b), verbosity=2)
        x, info = @constinferred linsolve(A, b; krylovdim=n, maxiter=2,
                                          rtol=tolerance(T), verbosity=1)
        @test info.converged > 0
        @test b ≈ A * x
        x, info = @constinferred linsolve(A, b, x; krylovdim=n, maxiter=2,
                                          rtol=tolerance(T))
        @test info.numops == 1

        A = rand(T, (n, n))
        α₀ = rand(T)
        α₁ = -rand(T)
        x, info = @constinferred(linsolve(A, b, zerovector(b), alg, α₀, α₁))
        @test b ≈ (α₀ * I + α₁ * A) * x
        @test info.converged > 0
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (n, n))
        b = MinimalVec{IP}(rand(T, (n,)))
        alg = GMRES(; krylovdim=n, maxiter=2, tol=tolerance(T) * norm(b), verbosity=2)
        x, info = @constinferred linsolve(wrapop(A), b; krylovdim=n, maxiter=2,
                                          rtol=tolerance(T), verbosity=1)
        @test info.converged > 0
        @test unwrap(b) ≈ A * unwrap(x)
        x, info = @constinferred linsolve(wrapop(A), b, x; krylovdim=n, maxiter=2,
                                          rtol=tolerance(T))
        @test info.numops == 1

        A = rand(T, (n, n))
        α₀ = rand(T)
        α₁ = -rand(T)
        x, info = @constinferred(linsolve(wrapop(A), b, zerovector(b), alg, α₀, α₁))
        @test unwrap(b) ≈ (α₀ * I + α₁ * A) * unwrap(x)
        @test info.converged > 0
    end
end

# Test GMRES with restart
@testset "GMRES with restarts" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, (N, N)) .- one(T) / 2
        A = I - T(9 / 10) * A / maximum(abs, eigvals(A))
        b = rand(T, N)
        x, info = @constinferred linsolve(A, b; krylovdim=3 * n,
                                          maxiter=50, rtol=tolerance(T))
        @test b ≈ A * x + info.residual

        A = rand(T, (N, N)) .- one(T) / 2
        α₀ = maximum(abs, eigvals(A))
        α₁ = -rand(T)
        α₁ *= T(9) / T(10) / abs(α₁)
        x, info = @constinferred linsolve(A, b, α₀, α₁; krylovdim=3 * n,
                                          maxiter=50, rtol=tolerance(T))
        @test b ≈ (α₀ * I + α₁ * A) * x + info.residual
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (N, N)) .- one(T) / 2
        b = MinimalVec{IP}(rand(T, (N,)))
        x, info = @constinferred linsolve(wrapop(A), b; krylovdim=3 * n,
                                          maxiter=50, rtol=tolerance(T))
        @test unwrap(b) ≈ A * unwrap(x) + unwrap(info.residual)

        A = rand(T, (N, N)) .- one(T) / 2
        α₀ = maximum(abs, eigvals(A))
        α₁ = -rand(T)
        α₁ *= T(9) / T(10) / abs(α₁)
        x, info = @constinferred linsolve(wrapop(A), b, zerovector(b), α₀, α₁;
                                          krylovdim=3 * n,
                                          maxiter=50, rtol=tolerance(T))
        @test unwrap(b) ≈ (α₀ * I + α₁ * A) * unwrap(x) + unwrap(info.residual)
    end
end

# Test BICGStab
@testset "BiCGStab" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, (n, n)) .- one(T) / 2
        A = I - T(9 / 10) * A / maximum(abs, eigvals(A))
        b = rand(T, n)
        alg = BiCGStab(; maxiter=4n, tol=tolerance(T) * norm(b), verbosity=2)
        x, info = @constinferred linsolve(A, b, zerovector(b), alg)
        @test info.converged > 0
        @test b ≈ A * x
        x, info = @constinferred linsolve(A, b, x, alg)
        @test info.numops == 1

        A = rand(T, (N, N)) .- one(T) / 2
        b = rand(T, N)
        α₀ = maximum(abs, eigvals(A))
        α₁ = -rand(T)
        α₁ *= T(9) / T(10) / abs(α₁)
        alg = BiCGStab(; maxiter=2, tol=tolerance(T) * norm(b), verbosity=1)
        x, info = @constinferred linsolve(A, b, zerovector(b), alg, α₀,
                                          α₁)
        @test b ≈ (α₀ * I + α₁ * A) * x + info.residual
        alg = BiCGStab(; maxiter=10 * N, tol=tolerance(T) * norm(b), verbosity=0)
        x, info = @constinferred linsolve(A, b, x, alg, α₀, α₁)
        @test info.converged > 0
        @test b ≈ (α₀ * I + α₁ * A) * x
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (n, n)) .- one(T) / 2
        b = MinimalVec{IP}(rand(T, (n,)))
        alg = BiCGStab(; maxiter=4n, tol=tolerance(T) * norm(b), verbosity=2)
        x, info = @constinferred linsolve(wrapop(A), b, zerovector(b), alg)
        @test info.converged > 0
        @test unwrap(b) ≈ A * unwrap(x)
        x, info = @constinferred linsolve(wrapop(A), b, x, alg)
        @test info.numops == 1

        A = rand(T, (N, N)) .- one(T) / 2
        b = MinimalVec{IP}(rand(T, (N,)))
        α₀ = maximum(abs, eigvals(A))
        α₁ = -rand(T)
        α₁ *= T(9) / T(10) / abs(α₁)
        alg = BiCGStab(; maxiter=2, tol=tolerance(T) * norm(b), verbosity=1)
        x, info = @constinferred linsolve(wrapop(A), b, zerovector(b), alg, α₀, α₁)
        @test unwrap(b) ≈ (α₀ * I + α₁ * A) * unwrap(x) + unwrap(info.residual)
        alg = BiCGStab(; maxiter=10 * N, tol=tolerance(T) * norm(b), verbosity=0)
        x, info = @constinferred linsolve(wrapop(A), b, x, alg, α₀, α₁)
        @test info.converged > 0
        @test unwrap(b) ≈ (α₀ * I + α₁ * A) * unwrap(x)
    end
end
