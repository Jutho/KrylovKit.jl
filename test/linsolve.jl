# Test CG complete
@testset "CG small problem" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T,(n,n))
        A = sqrt(A*A')
        b = rand(T,n)
        alg = CG(maxiter = 2n, tol = 10n*eps(real(T))*norm(b)) # because of loss of orthogonality, we choose maxiter = 2n
        x, info = @inferred linsolve(wrapop(A), wrapvec(b), wrapvec(zero(b)), alg)
        @test b ≈ A*unwrapvec(x)
        @test info.converged > 0

        A = rand(T,(n,n))
        A = sqrt(A*A')
        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @inferred linsolve(wrapop(A), wrapvec(b), wrapvec(zero(b)), alg, α₀, α₁)
        @test b ≈ (α₀*I+α₁*A)*unwrapvec(x)
        @test info.converged > 0
    end
end

# Test CG complete
@testset "CG large problem" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T,(N,N))
        A = sqrt(A*A')/N
        b = rand(T,N)
        x, info = @inferred linsolve(wrapop(A), wrapvec(b);
                    isposdef = true, maxiter = 1, krylovdim = N, rtol = 10*N*eps(real(T)))
        @test b ≈ A*unwrapvec(x) + unwrapvec(info.residual)

        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @inferred linsolve(wrapop(A), wrapvec(b), α₀, α₁;
                    isposdef = true, maxiter = 1, krylovdim = N, rtol = 10*N*eps(real(T)))
        @test b ≈ (α₀*I+α₁*A)*unwrapvec(x) + unwrapvec(info.residual)
    end
end

# Test GMRES complete
@testset "GMRES full factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T,(n,n)).-one(T)/2
        b = rand(T,n)
        alg = GMRES(krylovdim = n, maxiter = 2, tol = 2*n*eps(real(T))*norm(b))
        x, info = @inferred linsolve(wrapop(A), wrapvec(b), wrapvec(zero(b)), alg)
        @test info.converged > 0
        @test b ≈ A*unwrapvec(x)

        A = rand(T,(n,n))
        α₀ = rand(T)
        α₁ = -rand(T)
        x, info = @inferred(linsolve(wrapop(A), wrapvec(b), wrapvec(zero(b)), alg, α₀, α₁))
        @test b ≈ (α₀*I+α₁*A)*unwrapvec(x)
        @test info.converged > 0
    end
end

# Test GMRES with restart
@testset "GMRES with restarts" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T,(N,N)).-one(T)/2
        A = I-T(9/10)*A/maximum(abs, eigvals(A))
        b = rand(T,N)
        x, info = @inferred linsolve(wrapop(A), wrapvec(b); krylovdim = 3*n, maxiter = 50, rtol = 10*N*eps(real(T)))
        @test b ≈ A*unwrapvec(x) + unwrapvec(info.residual)

        A = rand(T,(N,N)).-one(T)/2
        α₀ = maximum(abs, eigvals(A))
        α₁ = -rand(T)
        α₁ *= T(9)/T(10)/abs(α₁)
        x, info = @inferred linsolve(wrapop(A), wrapvec(b), α₀, α₁; krylovdim = 3*n, maxiter = 50, rtol = 10*N*eps(real(T)))
        @test b ≈ (α₀*I+α₁*A)*unwrapvec(x) + unwrapvec(info.residual)
    end
end

@testset "full lsmr" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (n,n))
            v = rand(T,n);
            w = rand(T,n);
            alg = LSMR(orth = orth, krylovdim = 2*n, maxiter = 1, atol = 10*n*eps(real(T)),btol = 10*n*eps(real(T)))
            S, info = @inferred linsolve(wrapop(A), wrapvec(v),wrapvec(w), alg)
            @test info.converged > 0
            @test v≈A*unwrapvec(S)+unwrapvec(info.residual)
        end
    end
end

@testset "iterative lsmr" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (N,N))
            v = rand(T,N);
            w = rand(T,N);
            alg = LSMR(orth = orth, krylovdim = N, maxiter = 50, atol = 10*N*eps(real(T)),btol = 10*N*eps(real(T)))
            S, info = @inferred linsolve(wrapop(A), wrapvec(v),wrapvec(w), alg)
            @test info.converged > 0

            @test v≈A*unwrapvec(S)+unwrapvec(info.residual)
        end
    end
end
