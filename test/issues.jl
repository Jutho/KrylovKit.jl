# https://github.com/Jutho/KrylovKit.jl/issues/100
@testset "Issue #100" begin
    N = 32 # needs to be large enough to trigger shrinking
    A = rand(N, N)
    A += A'
    v₀ = [rand(N ÷ 2), rand(N ÷ 2)]

    vals, vecs, = eigsolve(v₀, 4, :LM; ishermitian=true) do v
        v′ = vcat(v...)
        y = A * v′
        return [y[1:(N ÷ 2)], y[(N ÷ 2 + 1):end]]
    end

    vals2, vecs2, = eigsolve(A, 4, :LM; ishermitian=true)
    @test vals ≈ vals2
    for (v, v′) in zip(vecs, vecs2)
        @test abs(inner(vcat(v...), v′)) ≈ 1
    end
end
