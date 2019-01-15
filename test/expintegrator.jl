function ϕ(A, v, p)
    m = LinearAlgebra.checksquare(A)
    length(v) == m || throw(DimensionMismatch("second dimension of A, $m, does not match length of x, $(length(v))"))
    p == 0 && return exp(A)*v
    A′ = fill!(similar(A, m+p, m+p), 0)
    copyto!(view(A′, 1:m, 1:m), A)
    copyto!(view(A′, 1:m, m+1), v)
    for k = 1:p-1
        A′[m+k, m+k+1] = 1
    end
    return exp(A′)[1:m, end]
end

@testset "Lanczos - expintegrator full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            A = (A+A')/2
            V = one(A)
            W = zero(A)
            alg = Lanczos(orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            for k = 1:n
                W[:,k], =  @inferred exponentiate(A, 1, view(V,:,k), alg)
            end
            @test W ≈ exp(A)

            pmax = 5
            for t in (rand(real(T)), -rand(real(T)), im*randn(real(T)), randn(real(T))+im*randn(real(T)))
                for p = 1:pmax
                    u = ntuple(i->rand(T, n), p+1)
                    w, = @inferred expintegrator(A, t, u, alg)
                    w2 = exp(t*A)*u[1]
                    for j = 1:p
                        w2 .+= t^j*ϕ(t*A, u[j+1], j)
                    end
                    @test w2 ≈ w
                end
            end
        end
    end
end

@testset "Arnoldi - expintegrator full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            V = one(A)
            W = zero(A)
            alg = Arnoldi(orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            for k = 1:n
                W[:,k], =  @inferred exponentiate(A, 1, view(V,:,k), alg)
            end
            @test W ≈ exp(A)

            pmax = 5
            for t in (rand(real(T)), -rand(real(T)), im*randn(real(T)), randn(real(T))+im*randn(real(T)))
                for p = 1:pmax
                    u = ntuple(i->rand(T, n), p+1)
                    w, = @inferred expintegrator(A, t, u, alg)
                    w2 = exp(t*A)*u[1]
                    for j = 1:p
                        w2 .+= t^j*ϕ(t*A, u[j+1], j)
                    end
                    @test w2 ≈ w
                end
            end
        end
    end
end

@testset "Lanczos - expintegrator iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            A = (A+A')/2
            pmax = 5
            for t in (rand(real(T)), -rand(real(T)), im*randn(real(T)), randn(real(T))+im*randn(real(T)))
                s = maximum(real(eigvals(t*A)))
                if s > 1
                    t *= 1/s
                end
                for p = 1:pmax
                    u = ntuple(i->rand(T, N), p+1)
                    w1, info = @inferred expintegrator(A, t, u...; maxiter = 100, krylovdim = n)
                    @assert info.converged > 0
                    w2 = exp(t*A)*u[1]
                    for j = 1:p
                        w2 .+= t^j*ϕ(t*A, u[j+1], j)
                    end
                    @test w2 ≈ w1
                    w1, info = @inferred expintegrator(A, t, u...; maxiter = 100, krylovdim = n, tol = 1e-3)
                    @test norm(w1 - w2) < 1e-2*abs(t)
                end
            end
        end
    end
end

@testset "Arnoldi - expintegrator iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            pmax = 5
            for t in (rand(real(T)), -rand(real(T)), im*randn(real(T)), randn(real(T))+im*randn(real(T)))
                s = maximum(real(eigvals(t*A)))
                if s > 1
                    t *= 1/s
                end
                for p = 1:pmax
                    u = ntuple(i->rand(T, N), p+1)
                    w1, info = @inferred expintegrator(A, t, u...; maxiter = 100, krylovdim = n)
                    @test info.converged > 0
                    w2 = exp(t*A)*u[1]
                    for j = 1:p
                        w2 .+= t^j*ϕ(t*A, u[j+1], j)
                    end
                    @test w2 ≈ w1
                    w1, info = @inferred expintegrator(A, t, u...; maxiter = 100, krylovdim = n, tol = 1e-3)
                    @test norm(w1 - w2) < 1e-2*abs(t)
                end
            end
        end
    end
end
