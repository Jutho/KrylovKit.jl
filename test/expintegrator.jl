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

@testset "Lanczos - expintegrator full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            A = (A + A') / 2
            V = one(A)
            W = zero(A)
            alg = Lanczos(;
                orth = orth, krylovdim = n, maxiter = 2, tol = tolerance(T),
                verbosity = STARTSTOP_LEVEL
            )
            for k in 1:n
                w, = @test_logs (:info,) exponentiate(
                    wrapop(A, Val(mode)), 1,
                    wrapvec(
                        view(V, :, k),
                        Val(mode)
                    ), alg
                )
                W[:, k] = unwrapvec(w)
            end
            @test W ≈ exp(A)

            pmax = 5
            alg = Lanczos(;
                orth = orth, krylovdim = n, maxiter = 2, tol = tolerance(T),
                verbosity = SILENT_LEVEL
            )
            for t in (
                    rand(real(T)), -rand(real(T)), im * randn(real(T)),
                    randn(real(T)) + im * randn(real(T)),
                )
                for p in 1:pmax
                    u = ntuple(i -> rand(T, n), p + 1)
                    w, info = @constinferred expintegrator(
                        wrapop(A, Val(mode)), t,
                        wrapvec.(u, Ref(Val(mode))), alg
                    )
                    w2 = exp(t * A) * u[1]
                    for j in 1:p
                        w2 .+= t^j * ϕ(t * A, u[j + 1], j)
                    end
                    @test info.converged > 0
                    @test w2 ≈ unwrapvec(w)
                end
            end
        end
    end
end

@testset "Arnoldi - expintegrator full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            V = one(A)
            W = zero(A)
            alg = Arnoldi(;
                orth = orth, krylovdim = n, maxiter = 2, tol = tolerance(T),
                verbosity = STARTSTOP_LEVEL
            )
            for k in 1:n
                w, = @test_logs (:info,) exponentiate(
                    wrapop(A, Val(mode)), 1,
                    wrapvec(
                        view(V, :, k),
                        Val(mode)
                    ), alg
                )
                W[:, k] = unwrapvec(w)
            end
            @test W ≈ exp(A)

            pmax = 5
            alg = Arnoldi(;
                orth = orth, krylovdim = n, maxiter = 2, tol = tolerance(T),
                verbosity = SILENT_LEVEL
            )
            for t in (
                    rand(real(T)), -rand(real(T)), im * randn(real(T)),
                    randn(real(T)) + im * randn(real(T)),
                )
                for p in 1:pmax
                    u = ntuple(i -> rand(T, n), p + 1)
                    w, info = @constinferred expintegrator(
                        wrapop(A, Val(mode)), t,
                        wrapvec.(u, Ref(Val(mode))), alg
                    )
                    w2 = exp(t * A) * u[1]
                    for j in 1:p
                        w2 .+= t^j * ϕ(t * A, u[j + 1], j)
                    end
                    @test info.converged > 0
                    @test w2 ≈ unwrapvec(w)
                end
            end
        end
    end
end

@testset "Lanczos - expintegrator iteratively ($mode)" for mode in
    (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = (1 // 2) .* (rand(T, (N, N)) .- one(T) / 2)
            A = (A + A') / 2
            pmax = 5
            for t in (
                    rand(real(T)), -rand(real(T)), im * randn(real(T)),
                    randn(real(T)) + im * randn(real(T)),
                )
                for p in 1:pmax
                    u = ntuple(i -> rand(T, N), p + 1)
                    w1, info = @constinferred expintegrator(
                        wrapop(A, Val(mode)), t,
                        wrapvec.(u, Ref(Val(mode)))...;
                        maxiter = 100, krylovdim = n,
                        eager = true
                    )
                    @test info.converged > 0
                    w2 = exp(t * A) * u[1]
                    for j in 1:p
                        w2 .+= t^j * ϕ(t * A, u[j + 1], j)
                    end
                    @test w2 ≈ unwrapvec(w1)
                    w1, info = @constinferred expintegrator(
                        wrapop(A, Val(mode)), t,
                        wrapvec.(u, Ref(Val(mode)))...;
                        maxiter = 100, krylovdim = n,
                        tol = 1.0e-3, eager = true
                    )
                    @test unwrapvec(w1) ≈ w2 atol = 1.0e-2 * abs(t)
                end
            end
        end
    end
end

@testset "Arnoldi - expintegrator iteratively ($mode)" for mode in
    (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = (1 // 2) .* (rand(T, (N, N)) .- one(T) / 2)
            pmax = 5
            for t in (
                    rand(real(T)), -rand(real(T)), im * randn(real(T)),
                    randn(real(T)) + im * randn(real(T)),
                )
                for p in 1:pmax
                    u = ntuple(i -> rand(T, N), p + 1)
                    w1, info = @constinferred expintegrator(
                        wrapop(A, Val(mode)), t,
                        wrapvec.(u, Ref(Val(mode)))...;
                        maxiter = 100, krylovdim = n,
                        eager = true
                    )
                    @test info.converged > 0
                    w2 = exp(t * A) * u[1]
                    for j in 1:p
                        w2 .+= t^j * ϕ(t * A, u[j + 1], j)
                    end
                end
            end
        end
    end
end

@testset "Arnoldi - expintegrator fixed point branch" begin
    @testset for T in (ComplexF32, ComplexF64) # less probable that :LR eig is repeated
        A = rand(T, (N, N)) / 10
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
        @test abs(dot(r, w1)) / norm(r) / norm(w1) ≈ 1 atol = 1.0e-4
        v₁ = rand(T, N)
        v₁ -= r * dot(l, v₁) / dot(l, r)
        w2, info2 = expintegrator(A, 1000.0, v₀, v₁)
        @test info2.converged > 0
        @test A * w2 ≈ -v₁
    end
end
