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


