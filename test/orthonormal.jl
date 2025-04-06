@testset "abstract_qr! for non-full vectors $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    T = ComplexF32
    A = rand(T, N, n)
    B = copy(A)
    Av = [A[:, i] for i in 1:size(A, 2)]
    # A is a non-full rank matrix
    Av[n÷2] = sum(Av[n÷2+1:end] .* rand(T, n - n ÷ 2))
    Bv = copy(Av)
    R, gi = KrylovKit.abstract_qr!(Av, T)
    @test length(gi) < n
    @test eltype(R) == eltype(eltype(A)) == T
    @test isapprox(hcat(Av[gi]...) * R, hcat(Bv...); atol = 1e4 * eps(real(T)))
    @test isapprox(hcat(Av[gi]...)' * hcat(Av[gi]...), I; atol = 1e4 * eps(real(T)))
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
    R, gi = KrylovKit.abstract_qr!(X, T)

    @test length(gi) < n
    @test eltype(R) == T
    @test isapprox(KrylovKit.blockinner(X[gi],X[gi],S = T), I; atol=1e4*eps(real(T)))
    ΔX = norm.(mul_test(X[gi],R) - Xcopy)
    @test isapprox(norm(ΔX), T(0); atol=1e4*eps(real(T)))
end

@testset "ortho_basis! for abstract inner product" begin
    T = ComplexF64
    H = rand(T, N, N);
    H = H'*H + I;
    H = (H + H')/2;
    ip(x,y) = x'*H*y
    x₀ = [InnerProductVec(rand(T, N), ip) for i in 1:n]
    x₁ = [InnerProductVec(rand(T, N), ip) for i in 1:2*n]
    tmp = zeros(T, 2*n, n)
    KrylovKit.abstract_qr!(x₁, T)
    KrylovKit.ortho_basis!(x₀, x₁, tmp)
    @test norm(KrylovKit.blockinner(x₀, x₁; S = T)) < eps(real(T))^0.5
end

