# Test CG complete
@testset "CG small problem ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (n, n))
        A = sqrt(A * A')
        b = rand(T, n)
        alg = CG(; maxiter=2n, tol=tolerance(T) * norm(b), verbosity=2) # because of loss of orthogonality, we choose maxiter = 2n
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode));
                                          ishermitian=true, isposdef=true, maxiter=2n,
                                          krylovdim=1, rtol=tolerance(T),
                                          verbosity=1)
        @test info.converged > 0
        @test unwrapvec(b) ≈ A * unwrapvec(x)
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x;
                                          ishermitian=true, isposdef=true, maxiter=2n,
                                          krylovdim=1, rtol=tolerance(T))
        @test info.numops == 1

        A = rand(T, (n, n))
        A = sqrt(A * A')
        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                                          wrapvec(zerovector(b), Val(mode)), alg, α₀, α₁)
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x)
        @test info.converged > 0
    end
end

# Test CG complete
@testset "CG large problem ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (N, N))
        A = sqrt(sqrt(A * A')) / N
        b = rand(T, N)
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode));
                                          isposdef=true, maxiter=1, krylovdim=N,
                                          rtol=tolerance(T))
        @test unwrapvec(b) ≈ A * unwrapvec(x) + unwrapvec(info.residual)

        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                                          α₀, α₁;
                                          isposdef=true, maxiter=1, krylovdim=N,
                                          rtol=tolerance(T))
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x) + unwrapvec(info.residual)
    end
end

# Test GMRES complete
@testset "GMRES full factorization ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (n, n)) .- one(T) / 2
        b = rand(T, n)
        alg = GMRES(; krylovdim=n, maxiter=2, tol=tolerance(T) * norm(b), verbosity=2)
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode));
                                          krylovdim=n, maxiter=2,
                                          rtol=tolerance(T), verbosity=1)
        @test info.converged == 1
        @test unwrapvec(b) ≈ A * unwrapvec(x)
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x,
                                          alg)
        @test info.numops == 1

        nreal = (T <: Real) ? n : 2n
        algr = GMRES(; krylovdim=nreal, maxiter=2, tol=tolerance(T) * norm(b), verbosity=2)
        xr, infor = @constinferred reallinsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                                                zerovector(x), algr)
        @test infor.converged == 1
        @test unwrapvec(x) ≈ unwrapvec(xr)

        A = rand(T, (n, n))
        α₀ = rand(T)
        α₁ = -rand(T)
        x, info = @constinferred(linsolve(A, b, zerovector(b), alg, α₀, α₁))
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x)
        @test info.converged == 1

        if mode == :vector && T <: Complex
            B = rand(T, (n, n))
            f = buildrealmap(A, B)
            α₀ = rand(real(T))
            α₁ = -rand(real(T))
            xr, infor = @constinferred reallinsolve(f, b, zerovector(b), algr, α₀, α₁)
            @test infor.converged == 1
            @test b ≈ (α₀ * xr + α₁ * A * xr + α₁ * B * conj(xr))
        end
    end
end

# Test GMRES with restart
@testset "GMRES with restarts ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (N, N)) .- one(T) / 2
        A = I - T(9 / 10) * A / maximum(abs, eigvals(A))
        b = rand(T, N)
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode));
                                          krylovdim=3 * n,
                                          maxiter=50, rtol=tolerance(T))
        @test unwrapvec(b) ≈ A * unwrapvec(x) + unwrapvec(info.residual)

        alg = GMRES(; krylovdim=3 * n, maxiter=50, tol=tolerance(T) * norm(b), verbosity=2)
        xr, infor = @constinferred reallinsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                                                zerovector(x), alg)
        @test unwrapvec(b) ≈ A * unwrapvec(xr) + unwrapvec(infor.residual)

        A = rand(T, (N, N)) .- one(T) / 2
        α₀ = maximum(abs, eigvals(A))
        α₁ = -9 * rand(T) / 10
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), α₀,
                                          α₁; krylovdim=3 * n,
                                          maxiter=50, rtol=tolerance(T))
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x) + unwrapvec(info.residual)

        if mode == :vector && T <: Complex
            A = rand(T, (N, N)) .- one(T) / 2
            B = rand(T, (N, N)) .- one(T) / 2
            f = buildrealmap(A, B)
            α₀ = 1
            α₁ = -1 / (maximum(abs, eigvals(A)) + maximum(abs, eigvals(B)))
            xr, infor = @constinferred reallinsolve(f, b, zerovector(b), alg, α₀, α₁)
            @test b ≈ (α₀ * xr + α₁ * A * xr + α₁ * B * conj(xr)) + infor.residual
        end
    end
end

# Test BICGStab
@testset "BiCGStab ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (n, n)) .- one(T) / 2
        A = I - T(9 / 10) * A / maximum(abs, eigvals(A))
        b = rand(T, n)
        alg = BiCGStab(; maxiter=4n, tol=tolerance(T) * norm(b), verbosity=2)
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                                          wrapvec(zerovector(b), Val(mode)), alg)
        @test info.converged > 0
        @test unwrapvec(b) ≈ A * unwrapvec(x)
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x,
                                          alg)
        @test info.numops == 1

        A = rand(T, (N, N)) .- one(T) / 2
        b = rand(T, N)
        α₀ = maximum(abs, eigvals(A))
        α₁ = -9 * rand(real(T)) / 10
        alg = BiCGStab(; maxiter=2, tol=tolerance(T) * norm(b), verbosity=1)
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                                          wrapvec(zerovector(b), Val(mode)), alg, α₀,
                                          α₁)
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x) + unwrapvec(info.residual)
        alg = BiCGStab(; maxiter=10 * N, tol=tolerance(T) * norm(b), verbosity=0)
        x, info = @constinferred linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x,
                                          alg, α₀, α₁)
        @test info.converged > 0
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x)

        xr, infor = @constinferred reallinsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                                                zerovector(x), alg, α₀, α₁)
        @test infor.converged > 0
        @test unwrapvec(xr) ≈ unwrapvec(x)

        if mode == :vector && T <: Complex
            A = rand(T, (N, N)) .- one(T) / 2
            B = rand(T, (N, N)) .- one(T) / 2
            f = buildrealmap(A, B)
            α₀ = 1
            α₁ = -1 / (maximum(abs, eigvals(A)) + maximum(abs, eigvals(B)))
            xr, infor = @constinferred reallinsolve(f, b, zerovector(b), alg, α₀, α₁)
            @test info.converged > 0
            @test b ≈ (α₀ * xr + α₁ * A * xr + α₁ * B * conj(xr))
        end
    end
end
