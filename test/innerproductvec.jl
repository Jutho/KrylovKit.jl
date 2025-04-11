#=
blockinner!(M,x,y):
M[i,j] = inner(x[i],y[j])
=#
@testset "blockinner! for non-full vectors $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A = [rand(T, N) for _ in 1:n]
    B = [rand(T, N) for _ in 1:n]
    M = Matrix{T}(undef, n, n)
    KrylovKit.blockinner!(M, A, B)
    M0 = hcat(A...)' * hcat(B...)
    @test eltype(M) == T
    @test isapprox(M, M0; atol = 1e4 * eps(real(T)))
end

@testset "blockinner! for abstract inner product" begin
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
    KrylovKit.blockinner!(M, X, Y);
    Xm = hcat([X[i].vec for i in 1:n]...);
    Ym = hcat([Y[i].vec for i in 1:n]...);
    M0 = Xm' * H * Ym;
    @test eltype(M) == T
    @test isapprox(M, M0; atol = eps(real(T))^(0.5))
end

@testset "similar_rand" begin
    @testset for T in [Float32,Float64,ComplexF32,ComplexF64]
        v = InnerProductVec(rand(T, n), x -> x'*x)
        sv = KrylovKit.similar_rand(v, n)
        @test length(sv) == n
        @test eltype(sv[2].vec) == T
        @test sv[2].dotf == v.dotf

        u = rand(T, n)
        su = KrylovKit.similar_rand(u, n)
        @test length(su) == n
        @test eltype(su[2]) == T
    end
end