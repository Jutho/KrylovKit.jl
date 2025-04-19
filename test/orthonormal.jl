@testset "abstract_qr! for non-full vectors $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A = rand(T, N, n)
    B = copy(A)
    Av = [A[:, i] for i in 1:size(A, 2)]
    # A is a non-full rank matrix
    Av[n÷2] = sum(Av[n÷2+1:end] .* rand(T, n - n ÷ 2))
    Bv = copy(Av)
    R, gi = KrylovKit.abstract_qr!(KrylovKit.BlockVec(Av, T), qr_tol(T))
    @test length(gi) < n
    @test eltype(R) == eltype(eltype(A)) == T
    @test isapprox(hcat(Av[gi]...) * R, hcat(Bv...); atol = tolerance(T))
    @test isapprox(hcat(Av[gi]...)' * hcat(Av[gi]...), I; atol = tolerance(T))
end

@testset "abstract_qr! for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N);
    H = H'*H + I;
    H = (H + H')/2;
    ip(x,y) = x'*H*y
    X₁ = InnerProductVec(rand(T, N), ip)
    X = [similar(X₁) for _ in 1:n];
    X[1] = X₁
    for i in 2:n-1
        X[i] = InnerProductVec(rand(T, N), ip)
    end

    # Make sure X is not full rank
    X[end] = sum(X[1:end-1] .* rand(T, n-1))
    Xcopy = deepcopy(X)
    R, gi = KrylovKit.abstract_qr!(KrylovKit.BlockVec(X, T), qr_tol(T))

    @test length(gi) < n
    @test eltype(R) == T
    BlockX = KrylovKit.BlockVec(X[gi], T)
    @test isapprox(KrylovKit.block_inner(BlockX,BlockX), I; atol=tolerance(T))
    ΔX = norm.(mul_test(X[gi],R) - Xcopy)
    @test isapprox(norm(ΔX), T(0); atol=tolerance(T))
end

@testset "ortho_basis! for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N);
    H = H'*H + I;
    H = (H + H')/2;
    ip(x,y) = x'*H*y
    x₀ = [InnerProductVec(rand(T, N), ip) for i in 1:n]
    x₁ = [InnerProductVec(rand(T, N), ip) for i in 1:2*n]
    Blockx₀ = KrylovKit.BlockVec(x₀, T)
    Blockx₁ = KrylovKit.BlockVec(x₁, T)
    KrylovKit.abstract_qr!(Blockx₁, qr_tol(T))
    KrylovKit.ortho_basis!(Blockx₀, Blockx₁)
    @test norm(KrylovKit.block_inner(Blockx₀, Blockx₁)) < 2* tolerance(T)
end

