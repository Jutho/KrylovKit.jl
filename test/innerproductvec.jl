#=
block_inner!(M,x,y):
M[i,j] = inner(x[i],y[j])
=#
@testset "block_inner! for non-full vectors $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A = [rand(T, N) for _ in 1:n]
    B = [rand(T, N) for _ in 1:n]
    M = Matrix{T}(undef, n, n)
    BlockA = KrylovKit.BlockVec(A, T)
    BlockB = KrylovKit.BlockVec(B, T)
    KrylovKit.block_inner!(M, BlockA, BlockB)
    M0 = hcat(BlockA.vec...)' * hcat(BlockB.vec...)
    @test eltype(M) == T
    @test isapprox(M, M0; atol = relax_tol(T))
end

@testset "block_inner! for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N);
    H = H'*H + I;
    H = (H + H')/2;
    ip(x,y) = x'*H*y
    X₁ = InnerProductVec(rand(T, N), ip);
    X = [similar(X₁) for _ in 1:n];
    X[1] = X₁;
    for i in 2:n
        X[i] = InnerProductVec(rand(T, N), ip)
    end
    Y₁ = InnerProductVec(rand(T, N), ip);
    Y = [similar(Y₁) for _ in 1:n];
    Y[1] = Y₁;
    for i in 2:n
        Y[i] = InnerProductVec(rand(T, N), ip)
    end    
    M = Matrix{T}(undef, n, n);
    BlockX = KrylovKit.BlockVec(X, T)
    BlockY = KrylovKit.BlockVec(Y, T)
    KrylovKit.block_inner!(M, BlockX, BlockY);
    Xm = hcat([X[i].vec for i in 1:n]...);
    Ym = hcat([Y[i].vec for i in 1:n]...);
    M0 = Xm' * H * Ym;
    @test eltype(M) == T
    @test isapprox(M, M0; atol = relax_tol(T))
end

@testset "block_randn_like" begin
    @testset for T in [Float32,Float64,ComplexF32,ComplexF64]
        v = InnerProductVec(rand(T, n), x -> x'*x)
        sv = KrylovKit.block_randn_like(v, n)
        @test length(sv) == n
        @test eltype(sv[2].vec) == T
        @test sv[2].dotf == v.dotf

        u = rand(T, n)
        su = KrylovKit.block_randn_like(u, n)
        @test length(su) == n
        @test eltype(su[2]) == T
    end
end

@testset "block_mul!" begin
    T = ComplexF64
    f = x -> x'*x
    A = [InnerProductVec(rand(T, N), f) for _ in 1:n]
    Acopy = [InnerProductVec(rand(T, N), f) for _ in 1:n]
    KrylovKit.copy!(Acopy, A)
    B = [InnerProductVec(rand(T, N), f) for _ in 1:n]
    M = rand(T, n, n)
    alpha = rand(T)
    beta = rand(T)
    BlockA = KrylovKit.BlockVec(A, T)
    BlockB = KrylovKit.BlockVec(B, T)
    KrylovKit.block_mul!(BlockA, BlockB, M, alpha, beta)
    @test isapprox(hcat([BlockA.vec[i].vec for i in 1:n]...), beta * hcat([Acopy[i].vec for i in 1:n]...) + alpha * hcat([BlockB.vec[i].vec for i in 1:n]...) * M; atol = tolerance(T))
end