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

# Test GMRES constructors with and without ndeflate
@testset "GMRES-DR constructors" begin
    orth = KrylovDefaults.orth
    alg = GMRES(orth, KrylovDefaults.maxiter[], KrylovDefaults.krylovdim[], 1.0e-10, WARN_LEVEL)
    @test alg.ndeflate == 0
    @test alg == GMRES(; tol = 1.0e-10, verbosity = WARN_LEVEL)
    @test GMRES{typeof(orth), Float64}(orth, 10, 20, 1.0e-10, WARN_LEVEL).ndeflate == 0
    @test GMRES(; krylovdim = 30, ndeflate = 10).ndeflate == 10
    @test_throws ArgumentError GMRES(; krylovdim = 10, ndeflate = 10)
    @test_throws ArgumentError GMRES(; ndeflate = -1)
end

# Test GMRES-DR without restart
@testset "GMRES-DR full factorization ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (n, n)) .- one(T) / 2
        b = rand(T, n)
        alg = GMRES(;
            krylovdim = n, ndeflate = div(n, 3), maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = SILENT_LEVEL
        )
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )
        @test info.converged == 1
        @test unwrapvec(b) ≈ A * unwrapvec(x)

        # without restart, GMRES-DR is GMRES
        algg = GMRES(;
            krylovdim = n, maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = SILENT_LEVEL
        )
        xg, infog = linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), algg
        )
        @test unwrapvec(x) ≈ unwrapvec(xg)
        @test info.numops == infog.numops

        @test_logs linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )
        alg = GMRES(;
            krylovdim = n, ndeflate = div(n, 3), maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = WARN_LEVEL
        )
        @test_logs linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )
        alg = GMRES(;
            krylovdim = n, ndeflate = div(n, 3), maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = STARTSTOP_LEVEL
        )
        @test_logs (:info,) (:info,) linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )
        alg = GMRES(;
            krylovdim = n, ndeflate = div(n, 3), maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = EACHITERATION_LEVEL
        )
        @test_logs min_level = Logging.Warn linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)),
            wrapvec(zerovector(b), Val(mode)), alg
        )

        alg = GMRES(;
            krylovdim = n, ndeflate = div(n, 3), maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = SILENT_LEVEL
        )
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x,
            alg
        )
        @test info.numops == 1
        alg = GMRES(;
            krylovdim = n, ndeflate = div(n, 3), maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = STARTSTOP_LEVEL
        )
        @test_logs (:info,) linsolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), x, alg)

        nreal = (T <: Real) ? n : 2n
        algr = GMRES(;
            krylovdim = nreal, ndeflate = div(nreal, 3), maxiter = 2, tol = tolerance(T) * norm(b),
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
        alg = GMRES(;
            krylovdim = n, ndeflate = div(n, 3), maxiter = 2, tol = tolerance(T) * norm(b),
            verbosity = SILENT_LEVEL
        )
        x, info = @constinferred(linsolve(A, b, zerovector(b), alg, α₀, α₁))
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A) * unwrapvec(x)
        @test info.converged == 1
    end
end

# A with four eigenvalues close to zero (a complex conjugate pair for real T) and the others
# in [1, 10], on which restarted GMRES stagnates
function deflationproblem(T, N)
    D = zeros(T, N, N)
    D[1, 1] = 1 // 100
    D[2, 2] = 2 // 100
    D[3:4, 3:4] = [1 -2; 2 1] // 100
    for i in 5:N
        D[i, i] = 1 + 9 * (i - 5) // (N - 5)
    end
    D += triu(rand(T, N, N) .- one(T) / 2, 1) / N
    Q = Matrix(qr(rand(T, N, N) .- one(T) / 2).Q)
    return Q * D * Q'
end

# Test GMRES-DR with restarts
@testset "GMRES-DR with restarts ($mode)" for mode in (:vector, :inplace, :outplace)
    # in single precision, `tolerance(T)` is beyond the attainable accuracy for this problem
    scalartypes = mode === :vector ? (Float64, ComplexF64) : (ComplexF64,)
    @testset for T in scalartypes
        A = deflationproblem(T, N)
        b = rand(T, N)
        x₀ = zerovector(b)
        tol = tolerance(T) * norm(b)
        algg = GMRES(; krylovdim = n, maxiter = 50, tol, verbosity = SILENT_LEVEL)
        alg = GMRES(; krylovdim = n, ndeflate = 4, maxiter = 50, tol, verbosity = SILENT_LEVEL)

        xg, infog = linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), wrapvec(x₀, Val(mode)), algg
        )
        @test infog.converged == 0
        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), wrapvec(x₀, Val(mode)), alg
        )
        @test info.converged == 1
        @test info.numops < infog.numops / 4
        @test unwrapvec(b) ≈ A * unwrapvec(x)
        @test unwrapvec(b) ≈ A * unwrapvec(x) + unwrapvec(info.residual)

        alg2 = GMRES(; krylovdim = n, ndeflate = 4, maxiter = 2, tol, verbosity = SILENT_LEVEL)
        @test_logs linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), wrapvec(x₀, Val(mode)), alg2
        )
        alg2 = GMRES(; krylovdim = n, ndeflate = 4, maxiter = 2, tol, verbosity = WARN_LEVEL)
        @test_logs (:warn,) linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), wrapvec(x₀, Val(mode)), alg2
        )
        alg2 = GMRES(;
            krylovdim = n, ndeflate = 4, maxiter = 2, tol, verbosity = STARTSTOP_LEVEL
        )
        @test_logs (:info,) (:warn,) linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), wrapvec(x₀, Val(mode)), alg2
        )

        x, info = @constinferred linsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), wrapvec(x₀, Val(mode));
            krylovdim = n, ndeflate = 4, maxiter = 50, rtol = tolerance(T)
        )
        @test info.converged == 1
        @test unwrapvec(b) ≈ A * unwrapvec(x)

        # α₀ + α₁ * A₁ = A
        α₀ = T(2)
        α₁ = T <: Real ? -T(1 // 2) : T(1 - im) / 2
        A₁ = (A - α₀ * I) / α₁
        x, info = @constinferred linsolve(
            wrapop(A₁, Val(mode)), wrapvec(b, Val(mode)), wrapvec(x₀, Val(mode)),
            alg, α₀, α₁
        )
        @test info.converged == 1
        @test unwrapvec(b) ≈ (α₀ * I + α₁ * A₁) * unwrapvec(x)

        # as a real linear map, A has twice as many eigenvalues close to zero for complex T
        nreal = (T <: Real) ? 1 : 2
        algr = GMRES(;
            krylovdim = nreal * n, ndeflate = nreal * 4, maxiter = 50, tol,
            verbosity = SILENT_LEVEL
        )
        xr, infor = @constinferred reallinsolve(
            wrapop(A, Val(mode)), wrapvec(b, Val(mode)), zerovector(x), algr
        )
        @test infor.converged == 1
        @test unwrapvec(b) ≈ A * unwrapvec(xr)

        if mode == :vector && T <: Complex
            B = rand(T, (N, N)) .- one(T) / 2
            f = buildrealmap(A, B / N)
            xr, infor = @constinferred reallinsolve(f, b, zerovector(b), algr)
            @test b ≈ (A * xr + B / N * conj(xr)) + infor.residual
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
