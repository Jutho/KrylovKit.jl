using KrylovKit, LinearAlgebra, Random, Test

@testset "apply on BlockVec" begin
    N = 100
    n = 10
    tolerance(T) = sqrt(eps(real(oneunit(T))))
    for T in [Float32, Float64, ComplexF64]
        A0 = rand(T,N,N);
        A0 = A0' * A0;
        x₀ = KrylovKit.BlockVec{T}([rand(T,N) for _ in 1:n])
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
    x₀ = KrylovKit.BlockVec{T}([InnerProductVec(rand(T,N), f) for _ in 1:n])
    y = KrylovKit.apply(A, x₀)
    @test isapprox(hcat([y[i].vec for i in 1:n]...), A0 * hcat([x₀[i].vec for i in 1:n]...); atol=tolerance(T))
end

@testset "copy! for BlockVec" begin
    N = 100
    n = 10
    tolerance(T) = sqrt(eps(real(oneunit(T))))
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        x = KrylovKit.BlockVec{T}([rand(T,N) for _ in 1:n])
        y = KrylovKit.BlockVec{T}([rand(T,N) for _ in 1:n])
        KrylovKit.copy!(y, x)
        @test isapprox(y.vec, x.vec; atol=tolerance(T))
    end
    T = ComplexF64
    f = (x,y) -> x' * y
    x = KrylovKit.BlockVec{T}([InnerProductVec(rand(T,N), f) for _ in 1:n])
    y = KrylovKit.BlockVec{T}([InnerProductVec(rand(T,N), f) for _ in 1:n])
    KrylovKit.copy!(y, x)
    @test isapprox([y.vec[i].vec for i in 1:n], [x.vec[i].vec for i in 1:n]; atol=tolerance(T))
end


