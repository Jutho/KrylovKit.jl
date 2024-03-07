function ϕ(A, v, p)
    m = LinearAlgebra.checksquare(A)
    length(v) == m ||
        throw(DimensionMismatch("second dimension of A, $m, does not match length of x, $(length(v))"))
    p == 0 && return exp(A) * v
    A′ = fill!(similar(A, m + p, m + p), 0)
    copyto!(view(A′, 1:m, 1:m), A)
    copyto!(view(A′, 1:m, m + 1), v)
    for k in 1:(p - 1)
        A′[m + k, m + k + 1] = 1
    end
    return exp(A′)[1:m, end]
end

@testset "Lanczos - expintegrator full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (n, n)) .- one(T) / 2
            A = (A + A') / 2
            V = one(A)
            W = zero(A)
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=2, tol=tolerance(T),
                          verbosity=2)
            for k in 1:n
                W[:, k] = first(@constinferred exponentiate(A, 1, view(V, :, k), alg))
            end
            @test W ≈ exp(A)

            pmax = 5
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=2, tol=tolerance(T),
                          verbosity=1)
            for t in (rand(real(T)), -rand(real(T)), im * randn(real(T)),
                      randn(real(T)) + im * randn(real(T)))
                for p in 1:pmax
                    u = ntuple(i -> rand(T, n), p + 1)
                    w, info = @constinferred expintegrator(A, t, u, alg)
                    w2 = exp(t * A) * u[1]
                    for j in 1:p
                        w2 .+= t^j * ϕ(t * A, u[j + 1], j)
                    end
                    @test info.converged > 0
                    @test w2 ≈ w
                end
            end
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (n, n)) .- one(T) / 2
        A = (A + A') / 2
        V = one(A)
        W = zero(A)
        alg = Lanczos(; krylovdim=n, maxiter=2, tol=tolerance(T), verbosity=2)
        for k in 1:n
            W[:, k] = unwrap(first(@constinferred exponentiate(wrapop(A), 1,
                                                               MinimalVec{IP}(view(V, :, k)),
                                                               alg)))
        end
        @test W ≈ exp(A)

        pmax = 5
        alg = Lanczos(; krylovdim=n, maxiter=2, tol=tolerance(T), verbosity=1)
        for t in (rand(real(T)), -rand(real(T)), im * randn(real(T)),
                  randn(real(T)) + im * randn(real(T)))
            for p in 1:pmax
                u = ntuple(i -> MinimalVec{IP}(rand(T, n)), p + 1)
                w, info = @constinferred expintegrator(wrapop(A), t, u, alg)
                w2 = exp(t * A) * unwrap(u[1])
                for j in 1:p
                    w2 .+= t^j * ϕ(t * A, unwrap(u[j + 1]), j)
                end
                @test info.converged > 0
                @test w2 ≈ unwrap(w)
            end
        end
    end
end

@testset "Arnoldi - expintegrator full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (n, n)) .- one(T) / 2
            V = one(A)
            W = zero(A)
            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=2, tol=tolerance(T),
                          verbosity=2)
            for k in 1:n
                W[:, k] = first(@constinferred exponentiate(A, 1, view(V, :, k), alg))
            end
            @test W ≈ exp(A)

            pmax = 5
            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=2, tol=tolerance(T),
                          verbosity=1)
            for t in (rand(real(T)), -rand(real(T)), im * randn(real(T)),
                      randn(real(T)) + im * randn(real(T)))
                for p in 1:pmax
                    u = ntuple(i -> rand(T, n), p + 1)
                    w, info = @constinferred expintegrator(A, t, u, alg)
                    w2 = exp(t * A) * u[1]
                    for j in 1:p
                        w2 .+= t^j * ϕ(t * A, u[j + 1], j)
                    end
                    @test info.converged > 0
                    @test w2 ≈ w
                end
            end
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (n, n)) .- one(T) / 2
        V = one(A)
        W = zero(A)
        alg = Arnoldi(; krylovdim=n, maxiter=2, tol=tolerance(T), verbosity=2)
        for k in 1:n
            W[:, k] = unwrap(first(@constinferred exponentiate(wrapop(A), 1,
                                                               MinimalVec{IP}(view(V, :, k)),
                                                               alg)))
        end
        @test W ≈ exp(A)

        pmax = 5
        alg = Arnoldi(; krylovdim=n, maxiter=2, tol=tolerance(T), verbosity=1)
        for t in (rand(real(T)), -rand(real(T)), im * randn(real(T)),
                  randn(real(T)) + im * randn(real(T)))
            for p in 1:pmax
                u = ntuple(i -> MinimalVec{IP}(rand(T, n)), p + 1)
                w, info = @constinferred expintegrator(wrapop(A), t, u, alg)
                w2 = exp(t * A) * unwrap(u[1])
                for j in 1:p
                    w2 .+= t^j * ϕ(t * A, unwrap(u[j + 1]), j)
                end
                @test info.converged > 0
                @test w2 ≈ unwrap(w)
            end
        end
    end
end

@testset "Lanczos - expintegrator iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (N, N)) .- one(T) / 2
            A = (A + A') / 2
            s = norm(eigvals(A), 1)
            rmul!(A, 1 / (10 * s))
            pmax = 5
            for t in (rand(real(T)), -rand(real(T)), im * randn(real(T)),
                      randn(real(T)) + im * randn(real(T)))
                for p in 1:pmax
                    u = ntuple(i -> rand(T, N), p + 1)
                    w1, info = @constinferred expintegrator(A, t, u...;
                                                            maxiter=100, krylovdim=n,
                                                            eager=true)
                    @assert info.converged > 0
                    w2 = exp(t * A) * u[1]
                    for j in 1:p
                        w2 .+= t^j * ϕ(t * A, u[j + 1], j)
                    end
                    @test w2 ≈ w1
                    w1, info = @constinferred expintegrator(A, t, u...;
                                                            maxiter=100, krylovdim=n,
                                                            tol=1e-3, eager=true)
                    @test w1 ≈ w2 atol = 1e-2 * abs(t)
                end
            end
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (N, N)) .- one(T) / 2
        A = (A + A') / 2
        s = norm(eigvals(A), 1)
        rmul!(A, 1 / (10 * s))
        pmax = 5
        for t in (rand(real(T)), -rand(real(T)), im * randn(real(T)),
                  randn(real(T)) + im * randn(real(T)))
            for p in 1:pmax
                u = ntuple(i -> MinimalVec{IP}(rand(T, N)), p + 1)
                w1, info = @constinferred expintegrator(wrapop(A), t, u...;
                                                        maxiter=100, krylovdim=n,
                                                        eager=true)
                @assert info.converged > 0
                w2 = exp(t * A) * unwrap(u[1])
                for j in 1:p
                    w2 .+= t^j * ϕ(t * A, unwrap(u[j + 1]), j)
                end
                @test w2 ≈ unwrap(w1)
                w1, info = @constinferred expintegrator(wrapop(A), t, u...;
                                                        maxiter=100, krylovdim=n,
                                                        tol=1e-3, eager=true)
                @test unwrap(w1) ≈ w2 atol = 1e-2 * abs(t)
            end
        end
    end
end

@testset "Arnoldi - expintegrator iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (N, N)) .- one(T) / 2
            s = norm(eigvals(A), 1)
            rmul!(A, 1 / (10 * s))
            pmax = 5
            for t in (rand(real(T)), -rand(real(T)), im * randn(real(T)),
                      randn(real(T)) + im * randn(real(T)))
                for p in 1:pmax
                    u = ntuple(i -> rand(T, N), p + 1)
                    w1, info = @constinferred expintegrator(A, t, u...;
                                                            maxiter=100, krylovdim=n,
                                                            eager=true)
                    @test info.converged > 0
                    w2 = exp(t * A) * u[1]
                    for j in 1:p
                        w2 .+= t^j * ϕ(t * A, u[j + 1], j)
                    end
                    @test w2 ≈ w1
                    w1, info = @constinferred expintegrator(A, t, u...;
                                                            maxiter=100, krylovdim=n,
                                                            tol=1e-3, eager=true)
                    @test w1 ≈ w2 atol = 1e-2 * abs(t)
                end
            end
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (N, N)) .- one(T) / 2
        s = norm(eigvals(A), 1)
        rmul!(A, 1 / (10 * s))
        pmax = 5
        for t in (rand(real(T)), -rand(real(T)), im * randn(real(T)),
                  randn(real(T)) + im * randn(real(T)))
            for p in 1:pmax
                u = ntuple(i -> MinimalVec{IP}(rand(T, N)), p + 1)
                w1, info = @constinferred expintegrator(wrapop(A), t, u...;
                                                        maxiter=100, krylovdim=n,
                                                        eager=true)
                @test info.converged > 0
                w2 = exp(t * A) * unwrap(u[1])
                for j in 1:p
                    w2 .+= t^j * ϕ(t * A, unwrap(u[j + 1]), j)
                end
                @test w2 ≈ unwrap(w1)
                w1, info = @constinferred expintegrator(wrapop(A), t, u...;
                                                        maxiter=100, krylovdim=n,
                                                        tol=1e-3, eager=true)
                @test unwrap(w1) ≈ w2 atol = 1e-2 * abs(t)
            end
        end
    end
end

@testset "Arnoldi - expintegrator fixed point branch" begin
    @testset for T in (ComplexF32, ComplexF64) # less probable that :LR eig is degenerate
        A = rand(T, (N, N))
        v₀ = rand(T, N)
        λs, vs, infoR = eigsolve(A, v₀, 1, :LR)
        @test infoR.converged > 0
        r = vs[1]
        A = A - λs[1] * I
        λs, vs, infoL = eigsolve(A', v₀, 1, :LR)
        @test infoL.converged > 0
        l = vs[1]
        w1, info1 = expintegrator(A, 1000.0, v₀)
        @test info1.converged > 0
        @test abs(dot(r, w1)) / norm(r) / norm(w1) ≈ 1 atol = 1e-4
        v₁ = rand(T, N)
        v₁ -= r * dot(l, v₁) / dot(l, r)
        w2, info2 = expintegrator(A, 1000.0, v₀, v₁)
        @test info2.converged > 0
        @test A * w2 ≈ -v₁
    end
end
