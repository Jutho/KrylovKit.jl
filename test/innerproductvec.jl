#=
block_inner(x,y):
M[i,j] = inner(x[i],y[j])
=#
@testset "block_inner for non-full vectors $T" for T in (Float32, Float64, ComplexF32,
                                                         ComplexF64)
    A = [rand(T, N) for _ in 1:n]
    B = [rand(T, N) for _ in 1:n]
    M = Matrix{T}(undef, n, n)
    BlockA = KrylovKit.BlockVec{T}(A)
    BlockB = KrylovKit.BlockVec{T}(B)
    M = KrylovKit.block_inner(BlockA, BlockB)
    M0 = hcat(BlockA.vec...)' * hcat(BlockB.vec...)
    @test eltype(M) == T
    @test isapprox(M, M0; atol=relax_tol(T))
end

@testset "block_inner for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N)
    H = H' * H + I
    H = (H + H') / 2
    ip(x, y) = x' * H * y
    X₁ = InnerProductVec(rand(T, N), ip)
    X = [similar(X₁) for _ in 1:n]
    X[1] = X₁
    for i in 2:n
        X[i] = InnerProductVec(rand(T, N), ip)
    end
    Y₁ = InnerProductVec(rand(T, N), ip)
    Y = [similar(Y₁) for _ in 1:n]
    Y[1] = Y₁
    for i in 2:n
        Y[i] = InnerProductVec(rand(T, N), ip)
    end
    BlockX = KrylovKit.BlockVec{T}(X)
    BlockY = KrylovKit.BlockVec{T}(Y)
    M = KrylovKit.block_inner(BlockX, BlockY)
    Xm = hcat([X[i].vec for i in 1:n]...)
    Ym = hcat([Y[i].vec for i in 1:n]...)
    M0 = Xm' * H * Ym
    @test eltype(M) == T
    @test isapprox(M, M0; atol=relax_tol(T))
end

#@testset "compute_residual! for abstract inner product" begin
    T = ComplexF64
    A = rand(T, N, N)
    A = A * A' + I
    ip(x,y) = x' * A * y
    M = rand(T, n, n)
    M = M' + M
    B = qr(rand(T, n, n)).R
    X0 = KrylovKit.BlockVec{T}([InnerProductVec(rand(T, N), ip) for _ in 1:n])
    X1 = KrylovKit.BlockVec{T}([InnerProductVec(rand(T, N), ip) for _ in 1:n])
    AX1 = KrylovKit.apply(A, X1)
    AX1copy = deepcopy(AX1)
    KrylovKit.compute_residual!(AX1, X1, M, X0, B)
    @test isapprox(hcat([AX1.vec[i] for i in 1:n]...),
                   hcat([AX1copy.vec[i] for i in 1:n]...) -
                   M * hcat([X1.vec[i] for i in 1:n]...) -
                   B * hcat([X0.vec[i] for i in 1:n]...); atol=tolerance(T))
end

x = InnerProductVec(rand(T, N), ip)
A
KrylovKit.apply(A, x)