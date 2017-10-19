using KrylovKit
using Test
using KrylovKit: hschur!, schur2eigvals, schur2eigvecs, permuteschur!

@testset "Dense Schur factorisation and associated methods: eltype $S" for S in (Float32, Float64, Complex64, Complex128)
    H = hessfact(rand(S,n,n))[:H]

    # schur factorisation of Hessenberg matrix
    T,U,w = hschur!(copy(H))
    @test H*U ≈ U*T
    @test schur2eigvals(T) ≈ w

    # full eigenvector computation
    V = schur2eigvecs(T)
    @test T*V ≈ V*Diagonal(w)

    # selected eigenvector computation
    p = randperm(n)
    select = p[1:(n>>1)]
    V2 = schur2eigvecs(T, select)
    @test T*V2 ≈ V2*Diagonal(w[select])

    # permuting / reordering schur: take permutations that keep 2x2 blocks together in real case
    p = sortperm(w, by=real)
    T2, U2 = permuteschur!(copy(T), copy(U), p)
    @test H*U2 ≈ U2*T2
    @test schur2eigvals(T2) ≈ w[p]

    p = sortperm(w, by=abs)
    T2, U2 = permuteschur!(copy(T), copy(U), p)
    @test H*U2 ≈ U2*T2
    @test schur2eigvals(T2) ≈ w[p]
end
