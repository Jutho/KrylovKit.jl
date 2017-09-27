using KrylovKit
using Base.Test

const n = 10
const N = 100

include("factorize.jl")

# Test GMRES complete
for T in (Float32, Float64, Complex64, Complex128)
    A = rand(T,(n,n)).-one(T)/2
    b = rand(T,n)
    alg = GMRES(krylovdim = n, maxiter = 1, reltol = 2*n*eps(real(T)))
    x, hist = @inferred linsolve(A, b, alg)
    @test hist.converged > 0
    @test x ≈ A\b

    α₀ = rand(T)
    α₁ = rand(T)
    x, hist = @inferred linsolve(A, b, alg, α₀, α₁)
    @test hist.converged > 0
    @test x ≈ (α₀*I+α₁*A)\b
end

# Test GMRES with restart
for T in (Float32, Float64, Complex64, Complex128)
    A = rand(T,(N,N)).-one(T)/2
    A = I-9*A/10/maximum(abs, eigvals(A))
    b = rand(T,N)
    alg = GMRES(krylovdim = n, maxiter = 20, reltol = 10*N*eps(real(T)))
    x, hist = @inferred linsolve(A, b, alg)
    @test hist.converged > 0
    @test x ≈ A\b

    α₀ = rand(T)
    α₁ = rand(T)
    x, hist = @inferred linsolve(A, b, alg, α₀, α₁)
    @test hist.converged > 0
    @test x ≈ (α₀*I+α₁*A)\b
end
