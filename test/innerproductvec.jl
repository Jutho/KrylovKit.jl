#=
block_inner!(M,x,y):
M[i,j] = inner(x[i],y[j])
=#
@testset "block_inner! for non-full vectors $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A = [rand(T, N) for _ in 1:n]
    B = [rand(T, N) for _ in 1:n]
    M = Matrix{T}(undef, n, n)
    KrylovKit.block_inner!(M, A, B)
    M0 = hcat(A...)' * hcat(B...)
    @test eltype(M) == T
    @test isapprox(M, M0; atol = 1e4 * eps(real(T)))
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
    KrylovKit.block_inner!(M, X, Y);
    Xm = hcat([X[i].vec for i in 1:n]...);
    Ym = hcat([Y[i].vec for i in 1:n]...);
    M0 = Xm' * H * Ym;
    @test eltype(M) == T
    @test isapprox(M, M0; atol = eps(real(T))^(0.5))
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

@testset "copyto! for InnerProductVec" begin
    T = ComplexF64
    f = x -> x'*x
    v = InnerProductVec(rand(T, n), f)
    w = InnerProductVec(rand(T, n), f)
    KrylovKit.copyto!(v, w)
    @test v.vec == w.vec
end

@testset "block_mul!" begin
    T = ComplexF64
    f = x -> x'*x
    A = [InnerProductVec(rand(T, N), f) for _ in 1:n]
    Acopy = [InnerProductVec(rand(T, N), f) for _ in 1:n]
    KrylovKit.copyto!(Acopy, A)
    B = [InnerProductVec(rand(T, N), f) for _ in 1:n]
    M = rand(T, n, n)
    alpha = rand(T)
    beta = rand(T)
    KrylovKit.block_mul!(A, B, M, alpha, beta)
    @test isapprox(hcat([A[i].vec for i in 1:n]...), beta * hcat([Acopy[i].vec for i in 1:n]...) + alpha * hcat([B[i].vec for i in 1:n]...) * M)
end