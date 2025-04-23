@testset "apply on BlockVec" begin
    for T in [Float32, Float64, ComplexF64]
    T = Float32
        A0 = rand(T,N,N);
        A0 = A0' * A0;
        x₀ = KrylovKit.BlockVec([rand(T,N) for _ in 1:n], T)
        for A in [A0, x -> A0*x]
            y = KrylovKit.apply(A, x₀)
            @test isapprox(hcat(y.vec...), A0 * hcat(x₀.vec...); atol=tolerance(T))
        end
    end
    T = ComplexF64
    A0 = rand(T,N,N);
    A0 = A0' * A0
    f(x,y) = x' * A0 * y
    A(x::InnerProductVec) = KrylovKit.InnerProductVec(A0 * x[], x.dotf)
    x₀ = KrylovKit.BlockVec([InnerProductVec(rand(T,N), f) for _ in 1:n], T)
    y = KrylovKit.apply(A, x₀)
    @test isapprox(hcat([y[i].vec for i in 1:n]...), A0 * hcat([x₀[i].vec for i in 1:n]...); atol=tolerance(T))
end

@testset "copy! for BlockVec" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        x = KrylovKit.BlockVec([rand(T,N) for _ in 1:n], T)
        y = KrylovKit.BlockVec([rand(T,N) for _ in 1:n], T)
        KrylovKit.copy!(y, x)
        @test isapprox(y.vec, x.vec; atol=tolerance(T))
    end
    T = ComplexF64
    f = (x,y) -> x' * y
    x = KrylovKit.BlockVec([InnerProductVec(rand(T,N), f) for _ in 1:n], T)
    y = KrylovKit.BlockVec([InnerProductVec(rand(T,N), f) for _ in 1:n], T)
    KrylovKit.copy!(y, x)
    @test isapprox([y.vec[i].vec for i in 1:n], [x.vec[i].vec for i in 1:n]; atol=tolerance(T))
end

using KrylovKit,Random,Test,LinearAlgebra
N =10
n = 5
Random.seed!(1234)
T = ComplexF64;
A = rand(T,N,N);
A = A' * A;
x₀ = rand(T,N);
alg = Lanczos(; krylovdim = n, maxiter = 10, tol = 1e-10)
eigsolve(A, x₀, n, :SR, alg)

using KrylovKit,Random,Test,LinearAlgebra
N =100
Random.seed!(1234)
T = ComplexF64;
A = rand(T,N,N);
A = A' * A;
x₀ = rand(T,N);
alg = Lanczos(; maxiter = 3, tol = 1e-10,blockmode=true, blocksize = 5)
vlues1, vectors1, info1 = eigsolve(A, x₀, 10, :SR, alg)

Random.seed!(1234)
T = ComplexF64;
A = rand(T,N,N);
A = A' * A;
x₀ = rand(T,N);
alg = Lanczos(; maxiter = 3, tol = 1e-10,blockmode=true, blocksize = 5)
vlues2, vectors2, info2 = eigsolve(A, x₀, 10, :SR, alg)

norm(vlues1 - vlues2)
norm(vectors1 - vectors2)
