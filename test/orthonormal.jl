@testset "abstract_qr! for non-full vectors $mode" for mode in
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
        R, gi = KrylovKit.abstract_qr!(KrylovKit.BlockVec{T}(wAv), tolerance(T))
        Av1 = [unwrapvec(wAv[i]) for i in gi]
        @test hcat(Av1...)' * hcat(Av1...) ≈ I
        @test length(gi) < n
        @test eltype(R) == eltype(eltype(A)) == T

        norm(hcat(Av1...) * R - hcat(Bv...))
        @test isapprox(hcat(Av1...) * R, hcat(Bv...); atol=tolerance(T))
        @test isapprox(hcat(Av1...)' * hcat(Av1...), I; atol=tolerance(T))
    end
end

@testset "abstract_qr! for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N) .- one(T) / 2
    H = H' * H + I
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
    R, gi = KrylovKit.abstract_qr!(KrylovKit.BlockVec{T}(X), tolerance(T))

    @test length(gi) < n
    @test eltype(R) == T
    BlockX = KrylovKit.BlockVec{T}(X[gi])
    @test isapprox(KrylovKit.block_inner(BlockX, BlockX), I; atol=tolerance(T))
    ΔX = norm.([sum(X[gi] .* R[:, i]) for i in 1:size(R, 2)] - Xcopy)
    @test isapprox(norm(ΔX), T(0); atol=tolerance(T))
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
        b₀ = KrylovKit.BlockVec{T}(x₀)
        b₁ = KrylovKit.BlockVec{T}(x₁)
        KrylovKit.abstract_qr!(b₁, tolerance(T))
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
    b₀ = KrylovKit.BlockVec{T}(x₀)
    b₁ = KrylovKit.BlockVec{T}(x₁)
    KrylovKit.abstract_qr!(b₁, tolerance(T))
    orthobasis_x₁ = KrylovKit.OrthonormalBasis(b₁.vec)
    KrylovKit.block_reorthogonalize!(b₀, orthobasis_x₁)
    @test norm(KrylovKit.block_inner(b₀, b₁)) < tolerance(T)
end
