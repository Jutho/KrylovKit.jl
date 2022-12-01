
@testset "GolubYe - geneigsolve full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (n, n)) .- one(T) / 2
            A = (A + A') / 2
            B = rand(T, (n, n)) .- one(T) / 2
            B = sqrt(B * B')
            v = rand(T, (n,))
            alg = GolubYe(; orth=orth, krylovdim=n, maxiter=1, tol=precision(T),
                          verbosity=1)
            n1 = div(n, 2)
            D1, V1, info = @constinferred geneigsolve((wrapop(A), wrapop(B)), wrapvec(v),
                                                      n1, :SR; orth=orth, krylovdim=n,
                                                      maxiter=1, tol=precision(T),
                                                      ishermitian=true, isposdef=true,
                                                      verbosity=2)
            @test KrylovKit.geneigselector((A, B), eltype(v); orth=orth, krylovdim=n,
                                           maxiter=1, tol=precision(T), ishermitian=true,
                                           isposdef=true) isa GolubYe
            n2 = n - n1
            D2, V2, info = @constinferred geneigsolve((wrapop(A), wrapop(B)), wrapvec(v),
                                                      n2, :LR, alg)
            @test vcat(D1[1:n1], reverse(D2[1:n2])) ≈ eigvals(A, B)

            U1 = hcat(unwrapvec.(V1)...)
            U2 = hcat(unwrapvec.(V2)...)
            @test U1' * B * U1 ≈ I
            @test U2' * B * U2 ≈ I

            @test A * U1 ≈ B * U1 * Diagonal(D1)
            @test A * U2 ≈ B * U2 * Diagonal(D2)
        end
    end
end

@testset "GolubYe - geneigsolve iteratively" begin
    @testset for T in (Float64, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (N, N)) .- one(T) / 2
            A = (A + A') / 2
            B = rand(T, (N, N)) .- one(T) / 2
            B = sqrt(B * B')
            v = rand(T, (N,))
            alg = GolubYe(; orth=orth, krylovdim=3 * n, maxiter=100,
                          tol=cond(B) * precision(T))
            D1, V1, info1 = @constinferred geneigsolve((wrapop(A), wrapop(B)), wrapvec(v),
                                                       n, :SR, alg)
            D2, V2, info2 = geneigsolve((wrapop(A), wrapop(B)), wrapvec(v), n, :LR, alg)

            l1 = info1.converged
            l2 = info2.converged
            @test l1 > 0
            @test l2 > 0
            @test D1[1:l1] ≊ eigvals(A, B)[1:l1]
            @test D2[1:l2] ≊ eigvals(A, B)[N:-1:(N - l2 + 1)]

            U1 = hcat(unwrapvec.(V1)...)
            U2 = hcat(unwrapvec.(V2)...)
            @test U1' * B * U1 ≈ I
            @test U2' * B * U2 ≈ I

            R1 = hcat(unwrapvec.(info1.residual)...)
            R2 = hcat(unwrapvec.(info2.residual)...)
            @test A * U1 ≈ B * U1 * Diagonal(D1) + R1
            @test A * U2 ≈ B * U2 * Diagonal(D2) + R2
        end
    end
end
