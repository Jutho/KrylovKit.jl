#=
block_inner(x,y):
M[i,j] = inner(x[i],y[j])
=#
@testset "block_inner for non-full vectors $mode" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = [rand(T, N) for _ in 1:n]
        B = [rand(T, N) for _ in 1:n]
        M0 = hcat(A...)' * hcat(B...)
        BlockA = KrylovKit.BlockVec{T}(wrapvec.(A, Val(mode)))
        BlockB = KrylovKit.BlockVec{T}(wrapvec.(B, Val(mode)))
        M = KrylovKit.block_inner(BlockA, BlockB)
        @test eltype(M) == T
        @test isapprox(M, M0; atol=relax_tol(T))
    end
end

@testset "block_inner for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N)
    H = H' * H + I
    H = (H + H') / 2
    ip(x, y) = x' * H * y
    X = [InnerProductVec(rand(T, N), ip) for _ in 1:n]
    Y = [InnerProductVec(rand(T, N), ip) for _ in 1:n]
    BlockX = KrylovKit.BlockVec{T}(X)
    BlockY = KrylovKit.BlockVec{T}(Y)
    M = KrylovKit.block_inner(BlockX, BlockY)
    Xm = hcat([X[i].vec for i in 1:n]...)
    Ym = hcat([Y[i].vec for i in 1:n]...)
    M0 = Xm' * H * Ym
    @test eltype(M) == T
    @test isapprox(M, M0; atol=relax_tol(T))
end

@testset "compute_residual! $mode" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, N, N) .- one(T) / 2
        A = A + A'
        M = rand(T, n, n)
        M = M' + M
        B = qr(rand(T, n, n)).R
        X0 = KrylovKit.BlockVec{T}([wrapvec(rand(T, N), Val(mode)) for _ in 1:n])
        X1 = KrylovKit.BlockVec{T}([wrapvec(rand(T, N), Val(mode)) for _ in 1:n])
        AX1 = KrylovKit.BlockVec{T}([wrapvec(rand(T, N), Val(mode)) for _ in 1:n])
        AX1copy = deepcopy(AX1)
        KrylovKit.compute_residual!(AX1, X1, M, X0, B)
        _bw2m(X) = hcat(unwrapvec.(X)...)
        @test isapprox(_bw2m(AX1), _bw2m(AX1copy) - _bw2m(X1) * M - _bw2m(X0) * B;
                       atol=tolerance(T))
    end
end

@testset "compute_residual! for abstract inner product" begin
    T = ComplexF64
    A = rand(T, N, N)
    A = A * A' + I
    ip(x, y) = x' * A * y
    M = rand(T, n, n)
    M = M' + M
    B = qr(rand(T, n, n)).R
    X0 = KrylovKit.BlockVec{T}([InnerProductVec(rand(T, N), ip) for _ in 1:n])
    X1 = KrylovKit.BlockVec{T}([InnerProductVec(rand(T, N), ip) for _ in 1:n])
    AX1 = KrylovKit.BlockVec{T}([InnerProductVec(rand(T, N), ip) for _ in 1:n])
    AX1copy = deepcopy(AX1)
    KrylovKit.compute_residual!(AX1, X1, M, X0, B)

    @test isapprox(hcat([AX1.vec[i].vec for i in 1:n]...),
                   hcat([AX1copy.vec[i].vec for i in 1:n]...) -
                   hcat([X1.vec[i].vec for i in 1:n]...) * M -
                   hcat([X0.vec[i].vec for i in 1:n]...) * B; atol=tolerance(T))
end
