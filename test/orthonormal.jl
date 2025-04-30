@testset "abstract_qr! for non-full vectors $T" for T in (Float32, Float64, ComplexF32,
                                                          ComplexF64)
    A = rand(T, N, n)
    B = copy(A)
    Av = [A[:, i] for i in 1:size(A, 2)]
    # A is a non-full rank matrix
    Av[n ÷ 2] = sum(Av[(n ÷ 2 + 1):end] .* rand(T, n - n ÷ 2))
    Bv = copy(Av)
    R, gi = KrylovKit.abstract_qr!(KrylovKit.BlockVec{T}(Av), qr_tol(T))
    @test length(gi) < n
    @test eltype(R) == eltype(eltype(A)) == T
    @test isapprox(hcat(Av[gi]...) * R, hcat(Bv...); atol=tolerance(T))
    @test isapprox(hcat(Av[gi]...)' * hcat(Av[gi]...), I; atol=tolerance(T))
end

@testset "abstract_qr! for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N)
    H = H' * H + I
    H = (H + H') / 2
    ip(x, y) = x' * H * y
    X₁ = InnerProductVec(rand(T, N), ip)
    X = [similar(X₁) for _ in 1:n]
    X[1] = X₁
    for i in 2:(n - 1)
        X[i] = InnerProductVec(rand(T, N), ip)
    end

    # Make sure X is not full rank
    X[end] = sum(X[1:(end - 1)] .* rand(T, n - 1))
    Xcopy = deepcopy(X)
    R, gi = KrylovKit.abstract_qr!(KrylovKit.BlockVec{T}(X), qr_tol(T))

    @test length(gi) < n
    @test eltype(R) == T
    BlockX = KrylovKit.BlockVec{T}(X[gi])
    @test isapprox(KrylovKit.block_inner(BlockX, BlockX), I; atol=tolerance(T))
    ΔX = norm.(mul_test(X[gi], R) - Xcopy)
    @test isapprox(norm(ΔX), T(0); atol=tolerance(T))
end

@testset "ortho_basis! for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N)
    H = H' * H + I
    H = (H + H') / 2
    ip(x, y) = x' * H * y

    x₀ = [InnerProductVec(rand(T, N), ip) for i in 1:n]
    x₁ = [InnerProductVec(rand(T, N), ip) for i in 1:(2 * n)]
    b₀ = KrylovKit.BlockVec{T}(x₀)
    b₁ = KrylovKit.BlockVec{T}(x₁)
    KrylovKit.abstract_qr!(b₁, qr_tol(T))

    orthobasis_x₁ = KrylovKit.OrthonormalBasis(b₁.vec)
    KrylovKit.ortho_basis!(b₀, orthobasis_x₁)
    @test norm(KrylovKit.block_inner(b₀, b₁)) < tolerance(T)
end
