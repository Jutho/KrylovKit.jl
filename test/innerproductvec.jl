#=
inner!(M,x,y):
M[i,j] = inner(x[i],y[j])
=#
@testset "inner! for non-full vectors $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A = [rand(T, N) for _ in 1:n]
    B = [rand(T, N) for _ in 1:n]
    M = Matrix{T}(undef, n, n)
    KrylovKit.inner!(M, A, B)
    M0 = hcat(A...)' * hcat(B...)
    @test eltype(M) == T
    @test isapprox(M, M0; atol = 1e4 * eps(real(T)))
end

@testset "inner! for abstract inner product" begin
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
    KrylovKit.inner!(M, X, Y);
    Xm = hcat([X[i].vec for i in 1:n]...);
    Ym = hcat([Y[i].vec for i in 1:n]...);
    M0 = Xm' * H * Ym;
    @test eltype(M) == T
    @test isapprox(M, M0; atol = eps(real(T))^(0.5))
end