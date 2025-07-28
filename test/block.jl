@testset "Block constructor" begin
    for mode in (:vector, :inplace, :outplace)
        scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                      (ComplexF64,)
        @testset for T in scalartypes
            x₀ = Block([wrapvec(rand(T, n), Val(mode)) for _ in 1:n])
            x₁ = Block([wrapvec(rand(T, n), Val(mode)) for _ in 1:n])
            @test typeof(x₀) == typeof(x₁)
        end
    end
    T = ComplexF64
    A = rand(T, n, n) .- one(T) / 2
    A = A' * A + I
    f(x, y) = x' * A * y
    x₀ = Block([InnerProductVec(rand(T, n), f) for _ in 1:n])
    x₁ = Block([InnerProductVec(rand(T, n), f) for _ in 1:n])
    @test typeof(x₀) == typeof(x₁)
end

@testset "apply on Block" begin
    for mode in (:vector, :inplace, :outplace)
        scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                      (ComplexF64,)
        @testset for T in scalartypes
            A = rand(T, n, n) .- one(T) / 2
            A = (A + A') / 2
            wx₀ = Block([wrapvec(rand(T, n), Val(mode)) for _ in 1:n])
            wy = KrylovKit.apply(wrapop(A, Val(mode)), wx₀)
            y = unwrapvec.(wy)
            x₀ = unwrapvec.(wx₀)
            @test isapprox(hcat(y...), A * hcat(x₀...); atol=tolerance(T))
        end
    end
    T = ComplexF64
    A = rand(T, n, n) .- one(T) / 2
    A = A' * A + I
    f(x, y) = x' * A * y
    Af(x::InnerProductVec) = KrylovKit.InnerProductVec(A * x[], x.dotf)
    x₀ = Block([InnerProductVec(rand(T, n), f) for _ in 1:n])
    y = KrylovKit.apply(Af, x₀)
    @test isapprox(hcat([y[i].vec for i in 1:n]...), A * hcat([x₀[i].vec for i in 1:n]...);
                   atol=tolerance(T))
end

@testset "copy for Block" begin
    for mode in (:vector, :inplace, :outplace)
        scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                      (ComplexF64,)
        @testset for T in scalartypes
            block0 = Block([wrapvec(rand(T, N), Val(mode)) for _ in 1:n])
            block1 = copy(block0)
            @test typeof(block0) == typeof(block1)
            @test unwrapvec.(block0.vec) == unwrapvec.(block1.vec)
        end
    end

    # test for abtract type
    T = ComplexF64
    A = rand(T, N, N) .- one(T) / 2
    A = A' * A + I
    f(x, y) = x' * A * y
    block0 = Block([InnerProductVec(rand(T, N), f) for _ in 1:n])
    block1 = copy(block0)
    @test typeof(block0) == typeof(block1)
    @test [block0.vec[i].vec for i in 1:n] == [block1.vec[i].vec for i in 1:n]
end

#=
block_inner(x,y):
M[i,j] = inner(x[i],y[j])
=#
@testset "block_inner for non-full vectors $mode" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF64) : (ComplexF64,)
    @testset for T in scalartypes
        A = [rand(T, N) for _ in 1:n]
        B = [rand(T, N) for _ in 1:n]
        M0 = hcat(A...)' * hcat(B...)
        BlockA = Block(wrapvec.(A, Val(mode)))
        BlockB = Block(wrapvec.(B, Val(mode)))
        M = KrylovKit.block_inner(BlockA, BlockB)
        @test eltype(M) == T
        @test isapprox(M, M0; atol=relax_tol(T))
    end
end

@testset "block_inner for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N) .- one(T) / 2
    H = H' * H + I
    ip(x, y) = x' * H * y
    X = [InnerProductVec(rand(T, N), ip) for _ in 1:n]
    Y = [InnerProductVec(rand(T, N), ip) for _ in 1:n]
    BlockX = Block(X)
    BlockY = Block(Y)
    M = KrylovKit.block_inner(BlockX, BlockY)
    Xm = hcat([X[i].vec for i in 1:n]...)
    Ym = hcat([Y[i].vec for i in 1:n]...)
    M0 = Xm' * H * Ym
    @test eltype(M) == T
    @test isapprox(M, M0; atol=relax_tol(T))
end

@testset "block_reorthogonalize! for non-full vectors $mode" for mode in
                                                                 (:vector, :inplace,
                                                                  :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, n, n) .- one(T) / 2
        A = (A + A') / 2
        x₀ = [wrapvec(rand(T, N), Val(mode)) for i in 1:n]
        x₁ = [wrapvec(rand(T, N), Val(mode)) for i in 1:(2 * n)]
        b₀ = Block(x₀)
        b₁ = Block(x₁)
        KrylovKit.block_qr!(b₁, tolerance(T))
        orthobasis_x₁ = KrylovKit.OrthonormalBasis(b₁.vec)
        KrylovKit.block_reorthogonalize!(b₀, orthobasis_x₁)
        @test norm(KrylovKit.block_inner(b₀, b₁)) < tolerance(T)
    end
end

@testset "block_reorthogonalize! for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N) .- one(T) / 2
    H = H' * H + I
    ip(x, y) = x' * H * y

    x₀ = [InnerProductVec(rand(T, N), ip) for i in 1:n]
    x₁ = [InnerProductVec(rand(T, N), ip) for i in 1:(2 * n)]
    b₀ = Block(x₀)
    b₁ = Block(x₁)
    KrylovKit.block_qr!(b₁, tolerance(T))
    orthobasis_x₁ = KrylovKit.OrthonormalBasis(b₁.vec)
    KrylovKit.block_reorthogonalize!(b₀, orthobasis_x₁)
    @test norm(KrylovKit.block_inner(b₀, b₁)) < tolerance(T)
end

@testset "block_qr! for non-full vectors $mode" for mode in
                                                    (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, n, n) .- one(T) / 2
        B = copy(A)
        Av = [A[:, i] for i in 1:size(A, 2)]
        # A is a non-full rank matrix
        Av[n ÷ 2] = sum(Av[(n ÷ 2 + 1):end] .* rand(T, n - n ÷ 2))
        Bv = deepcopy(Av)
        wAv = wrapvec.(Av, Val(mode))
        R, gi = KrylovKit.block_qr!(Block(wAv), tolerance(T))
        Av1 = [unwrapvec(wAv[i]) for i in gi]
        @test length(gi) < n
        @test eltype(R) == eltype(A) == T
        @test isapprox(hcat(Av1...) * R, hcat(Bv...); atol=tolerance(T))
        @test isapprox(hcat(Av1...)' * hcat(Av1...), I; atol=tolerance(T))
    end
end

@testset "block_qr! for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N) .- one(T) / 2
    H = H' * H + I
    ip(x, y) = x' * H * y
    X = [InnerProductVec(rand(T, N), ip) for _ in 1:n]

    # Make sure X is not full rank
    X[end] = sum(X[1:(end - 1)] .* rand(T, n - 1))
    Xcopy = deepcopy(X)
    R, gi = KrylovKit.block_qr!(Block(X), tolerance(T))

    @test length(gi) < n
    @test eltype(R) == T
    BlockX = Block(X[gi])
    @test isapprox(KrylovKit.block_inner(BlockX, BlockX), I; atol=tolerance(T))
    @test isapprox(hcat([X[i].vec for i in gi]...) * R,
                   hcat([Xcopy[i].vec for i in 1:n]...); atol=tolerance(T))
end
