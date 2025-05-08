@testset "apply on BlockVec" begin
    for mode in (:vector, :inplace, :outplace)
        scalartypes = mode === :vector ? (Float32, Float64, ComplexF64) :
                      (ComplexF64,)
        @testset for T in scalartypes
            mode = :inplace
            T = ComplexF64
            A = rand(T, N, N) .- one(T) / 2
            A = (A + A') / 2
            wx₀ = KrylovKit.BlockVec{T}([wrapvec(rand(T, N), Val(mode)) for _ in 1:n])
            wy = KrylovKit.apply(wrapop(A, Val(mode)), wx₀)
            y = unwrapvec.(wy)
            x₀ = unwrapvec.(wx₀)
            @test isapprox(hcat(y...), A * hcat(x₀...); atol=tolerance(T))
        end
    end
    T = ComplexF64
    A = rand(T, N, N) .- one(T) / 2
    A = (A + A') / 2
    f(x, y) = x' * A * y
    Af(x::InnerProductVec) = KrylovKit.InnerProductVec(A * x[], x.dotf)
    x₀ = KrylovKit.BlockVec{T}([InnerProductVec(rand(T, N), f) for _ in 1:n])
    y = KrylovKit.apply(Af, x₀)
    @test isapprox(hcat([y[i].vec for i in 1:n]...), A * hcat([x₀[i].vec for i in 1:n]...);
                   atol=tolerance(T))
end

@testset "initialize for BlockVec" begin
    for mode in (:vector, :inplace, :outplace)
        scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                      (ComplexF64,)
        @testset for T in scalartypes
            block0 = KrylovKit.initialize(wrapvec(rand(T, N), Val(mode)), n)
            @test block0 isa KrylovKit.BlockVec
            @test length(block0) == n
            Tv = mode === :vector ? Vector{T} :
                 mode === :inplace ? MinimalVec{true,Vector{T}} :
                 MinimalVec{false,Vector{T}}

            @test Tuple(typeof(block0).parameters) == (Tv, T)
        end
    end

    # test for abtract type
    T = ComplexF64
    f(x, y) = x' * y
    x0 = InnerProductVec(rand(T, N), f)
    block0 = KrylovKit.initialize(x0, n)
    @test block0 isa KrylovKit.BlockVec
    @test length(block0) == n
    @test Tuple(typeof(block0).parameters) == (typeof(x0), T)
end

@testset "copy for BlockVec" begin
    for mode in (:vector, :inplace, :outplace)
        scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                      (ComplexF64,)
        @testset for T in scalartypes
            block0 = KrylovKit.BlockVec{T}([wrapvec(rand(T, N), Val(mode)) for _ in 1:n])
            block1 = copy(block0)
            @test typeof(block0) == typeof(block1)
            @test unwrapvec.(block0.vec) == unwrapvec.(block1.vec)
        end
    end

    # test for abtract type
    T = ComplexF64
    f(x, y) = x' * y
    block0 = KrylovKit.BlockVec{T}([InnerProductVec(rand(T, N), f) for _ in 1:n])
    block1 = copy(block0)
    @test typeof(block0) == typeof(block1)
    @test [block0.vec[i].vec for i in 1:n] == [block1.vec[i].vec for i in 1:n]
end
