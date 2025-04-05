@testset "abstract_qr! for non-full vectors $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    n = 1000
    m = 10
    A = rand(T, n, m)
    B = copy(A)
    Av = [A[:, i] for i in 1:size(A, 2)]
    # A is a non-full rank matrix
    Av[m÷2] = sum(Av[m÷2+1:end] .* rand(T, m - m ÷ 2))
    Bv = copy(Av)
    R, gi = KrylovKit.abstract_qr!(Av, T)
    @test length(gi) < m
    @test eltype(R) == eltype(eltype(A)) == T
    @test isapprox(hcat(Av[gi]...) * R, hcat(Bv...); atol = 1e4 * eps(real(T)))
    @test isapprox(hcat(Av[gi]...)' * hcat(Av[gi]...), I; atol = 1e4 * eps(real(T)))
end

