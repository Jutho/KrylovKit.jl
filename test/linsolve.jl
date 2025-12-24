# Test CG complete
@testset "CG small problem ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (n, n))
        A = sqrt(A * A')
        b = rand(T, n)
        alg = CG(; maxiter = 2n, tol = tolerance(T) * norm(b), verbosity = SILENT_LEVEL) # because of loss of orthogonality, we choose maxiter = 2n
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            ishermitian = true, isposdef = true, maxiter = 2n,
            krylovdim = 1, rtol = tolerance(T),
            verbosity = SILENT_LEVEL
        )
        @test info.converged > 0
        @test unwrapvec(b) ≈ A * unwrapvec(x)
        @test_logs linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            ishermitian = true, isposdef = true, maxiter = 2n,
            krylovdim = 1, rtol = tolerance(T),
            verbosity = SILENT_LEVEL
        )
        @test_logs linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            ishermitian = true, isposdef = true, maxiter = 2n,
            krylovdim = 1, rtol = tolerance(T),
            verbosity = WARN_LEVEL
        )
        @test_logs (:info,) (:info,) linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            ishermitian = true, isposdef = true, maxiter = 2n,
            krylovdim = 1, rtol = tolerance(T),
            verbosity = STARTSTOP_LEVEL
        )
        @test_logs min_level = Logging.Warn linsolve(
            wrapop(A, Val(mode)),
            wrapvec(b, Val(mode));
            ishermitian = true, isposdef = true,
            maxiter = 2n,
            krylovdim = 1, rtol = tolerance(T),
            verbosity = EACHITERATION_LEVEL
        )

        x, info = linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        @test info.numops == 1
        @test_logs linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        alg = CG(; maxiter = 2n, tol = tolerance(T) * norm(b), verbosity = WARN_LEVEL)
        @test_logs linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        alg = CG(; maxiter = 2n, tol = tolerance(T) * norm(b), verbosity = STARTSTOP_LEVEL)
        @test_logs (:info,) linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        alg = CG(; maxiter = 2n, tol = tolerance(T) * norm(b), verbosity = SILENT_LEVEL)

        A = rand(T, (n, n))
        A = sqrt(A * A')
        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg, α₀, α₁
        )
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
        x₀ = rand(T, N)
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(x₀, Val(mode));
            isposdef = true, maxiter = 1, krylovdim = N,
            rtol = tolerance(T)
        )
        @test unwrapvec(b) ≈ A * unwrapvec(x) + unwrapvec(info.residual)
        if info.converged == 0
            @test_logs linsolve(
                wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                wrapvec(x₀, Val(mode));
                isposdef = true, maxiter = 1, krylovdim = N,
                rtol = tolerance(T), verbosity = SILENT_LEVEL
            )
            @test_logs (:warn,) linsolve(
                wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                wrapvec(x₀, Val(mode));
                isposdef = true, maxiter = 1, krylovdim = N,
                rtol = tolerance(T), verbosity = WARN_LEVEL
            )
            @test_logs (:info,) (:warn,) linsolve(
                wrapop(A, Val(mode)),
                wrapvec(b, Val(mode)),
                wrapvec(x₀, Val(mode));
                isposdef = true, maxiter = 1, krylovdim = N,
                rtol = tolerance(T),
                verbosity = STARTSTOP_LEVEL
            )
        end

        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            α₀, α₁;
            isposdef = true, maxiter = 1, krylovdim = N,
            rtol = tolerance(T)
        )
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
        alg = GMRES(;
            krylovdim = n, maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = SILENT_LEVEL
        )
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            krylovdim = n, maxiter = 2,
            rtol = tolerance(T), verbosity = SILENT_LEVEL
        )
        @test info.converged == 1
        @test unwrapvec(b) ≈ A * unwrapvec(x)
        @test_logs linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            krylovdim = n, maxiter = 2,
            rtol = tolerance(T), verbosity = SILENT_LEVEL
        )
        @test_logs linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            krylovdim = n, maxiter = 2,
            rtol = tolerance(T), verbosity = WARN_LEVEL
        )
        @test_logs (:info,) (:info,) linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode));
            krylovdim = n, maxiter = 2,
            rtol = tolerance(T), verbosity = STARTSTOP_LEVEL
        )
        @test_logs min_level = Logging.Warn linsolve(
            wrapop(A, Val(mode)),
            wrapvec(b, Val(mode));
            krylovdim = n, maxiter = 2,
            rtol = tolerance(T),
            verbosity = EACHITERATION_LEVEL
        )

        alg = GMRES(;
            krylovdim = n, maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = SILENT_LEVEL
        )
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x,
            alg
        )
        @test info.numops == 1
        @test_logs linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        alg = GMRES(;
            krylovdim = n, maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = WARN_LEVEL
        )
        @test_logs linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        alg = GMRES(;
            krylovdim = n, maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = STARTSTOP_LEVEL
        )
        @test_logs (:info,) linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        alg = GMRES(;
            krylovdim = n, maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = SILENT_LEVEL
        )

        nreal = (T <: Real) ? n : 2n
        algr = GMRES(;
            krylovdim = nreal, maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = SILENT_LEVEL
        )
        xr, infor = @constinferred reallinsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            zerovector(x), algr
        )
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
        x₀ = rand(T, N)
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(x₀, Val(mode));
            krylovdim = 3 * n,
            maxiter = 50, rtol = tolerance(T)
        )
        @test unwrapvec(b) ≈ A * unwrapvec(x) + unwrapvec(info.residual)
        if info.converged == 0
            @test_logs linsolve(
                wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                wrapvec(x₀, Val(mode));
                krylovdim = 3 * n,
                maxiter = 50, rtol = tolerance(T), verbosity = SILENT_LEVEL
            )
            @test_logs (:warn,) linsolve(
                wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                wrapvec(x₀, Val(mode));
                krylovdim = 3 * n,
                maxiter = 50, rtol = tolerance(T),
                verbosity = WARN_LEVEL
            )
            @test_logs (:info,) (:warn,) linsolve(
                wrapop(A, Val(mode)),
                wrapvec(b, Val(mode)),
                wrapvec(x₀, Val(mode));
                krylovdim = 3 * n,
                maxiter = 50, rtol = tolerance(T),
                verbosity = STARTSTOP_LEVEL
            )
        end

        alg = GMRES(;
            krylovdim = 3 * n, maxiter = 50, tol = tolerance(T) * norm(b),
            verbosity = SILENT_LEVEL
        )
        xr, infor = @constinferred reallinsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            zerovector(x), alg
        )
        @test unwrapvec(b) ≈ A * unwrapvec(xr) + unwrapvec(infor.residual)

        A = rand(T, (N, N)) .- one(T) / 2
        α₀ = maximum(abs, eigvals(A))
        α₁ = -9 * rand(T) / 10
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), α₀,
            α₁; krylovdim = 3 * n,
            maxiter = 50, rtol = tolerance(T)
        )
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

# Test BiCGStab
@testset "BiCGStab small problem ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (n, n)) .- one(T) / 2
        A = I - T(9 / 10) * A / maximum(abs, eigvals(A))
        b = rand(T, n)
        alg = BiCGStab(; maxiter = 4n, tol = tolerance(T) * norm(b), verbosity = SILENT_LEVEL)
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )
        @test info.converged > 0
        @test unwrapvec(b) ≈ A * unwrapvec(x)
        alg = BiCGStab(; maxiter = 4n, tol = tolerance(T) * norm(b), verbosity = SILENT_LEVEL)
        @test_logs linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )
        alg = BiCGStab(; maxiter = 4n, tol = tolerance(T) * norm(b), verbosity = WARN_LEVEL)
        @test_logs linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )
        alg = BiCGStab(; maxiter = 4n, tol = tolerance(T) * norm(b), verbosity = STARTSTOP_LEVEL)
        @test_logs (:info,) (:info,) linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )
        alg = BiCGStab(;
            maxiter = 4n, tol = tolerance(T) * norm(b),
            verbosity = EACHITERATION_LEVEL
        )
        @test_logs min_level = Logging.Warn linsolve(
            wrapop(A, Val(mode)),
            wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )
        alg = BiCGStab(; maxiter = 4n, tol = tolerance(T) * norm(b), verbosity = SILENT_LEVEL)

        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x,
            alg
        )
        @test info.numops == 1
        alg = BiCGStab(; maxiter = 4n, tol = tolerance(T) * norm(b), verbosity = SILENT_LEVEL)
        @test_logs linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        alg = BiCGStab(; maxiter = 4n, tol = tolerance(T) * norm(b), verbosity = WARN_LEVEL)
        @test_logs linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        alg = BiCGStab(; maxiter = 4n, tol = tolerance(T) * norm(b), verbosity = STARTSTOP_LEVEL)
        @test_logs (:info,) linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)
        alg = BiCGStab(; maxiter = 4n, tol = tolerance(T) * norm(b), verbosity = SILENT_LEVEL)

        α₀ = rand(real(T)) + 1
        α₁ = rand(real(T))
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg, α₀, α₁
        )
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x)
        @test info.converged > 0
    end
end

@testset "BiCGStab large problem ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (N, N)) .- one(T) / 2
        b = rand(T, N)
        α₀ = maximum(abs, eigvals(A))
        α₁ = -9 * rand(real(T)) / 10
        alg = BiCGStab(; maxiter = 2, tol = tolerance(T) * norm(b), verbosity = SILENT_LEVEL)
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg, α₀,
            α₁
        )
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x) + unwrapvec(info.residual)
        if info.converged == 0
            @test_logs linsolve(
                wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                wrapvec(zerovector(b), Val(mode)), alg, α₀, α₁
            )
            alg = BiCGStab(; maxiter = 2, tol = tolerance(T) * norm(b), verbosity = WARN_LEVEL)
            @test_logs (:warn,) linsolve(
                wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
                wrapvec(zerovector(b), Val(mode)), alg, α₀, α₁
            )
            alg = BiCGStab(;
                maxiter = 2, tol = tolerance(T) * norm(b),
                verbosity = STARTSTOP_LEVEL
            )
            @test_logs (:info,) (:warn,) linsolve(
                wrapop(A, Val(mode)),
                wrapvec(b, Val(mode)),
                wrapvec(zerovector(b), Val(mode)), alg,
                α₀, α₁
            )
        end

        alg = BiCGStab(; maxiter = 10 * N, tol = tolerance(T) * norm(b), verbosity = SILENT_LEVEL)
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x,
            alg, α₀, α₁
        )
        @test info.converged > 0
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x)

        xr, infor = @constinferred reallinsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            zerovector(x), alg, α₀, α₁
        )
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
