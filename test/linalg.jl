using KrylovKit: OrthonormalBasis, householder, rows, cols, hschur!, schur2eigvals, schur2eigvecs, permuteschur!

@testset "Orthonormalize with algorithm $alg" for alg in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
    @testset for S in (Float32, Float64, ComplexF32, ComplexF64)
        b = OrthonormalBasis{Vector{S}}()
        A = randn(S,(n,n))
        v, r, x = orthonormalize(A[:,1], b, alg)
        @test r ≈ norm(A[:,1])
        @test norm(v) ≈ 1
        @test length(x) == 0
        push!(b, v)
        for i = 2:n
            v, r, x = orthonormalize(A[:,i], b, alg)
            @test norm([r,norm(x)]) ≈ norm(A[:,i])
            @test norm(v) ≈ 1
            @test length(x) == i-1
            push!(b, v)
        end
        U = hcat(b...)
        @test U'*U ≈ I
        v = randn(S, n)
        @test U*v ≈ b*v
    end
end

@testset "Givens and Householder" begin
    @testset for S in (Float32, Float64, ComplexF32, ComplexF64)
        U, = svd!(randn(S,(n,n)))
        b = OrthonormalBasis(map(copy,cols(U)))
        v = randn(S,(n,))
        g, r = givens(v, 1, n)
        @test rmul!(U,g) ≈ hcat(rmul!(b,g)...)
        h, r = householder(v, axes(v,1), 3)
        @test r ≈ norm(v)
        v2 = lmul!(h, copy(v))
        v3 = zero(v2); v3[3] = r;
        @test v2 ≈ v3
        @test lmul!(h, one(U)) ≈ rmul!(one(U), h)
        @test lmul!(h, one(U))' ≈ lmul!(h', one(U))
        @test rmul!(U, h) ≈ hcat(rmul!(b,h)...)
    end
end

@testset "Rows and cols iterator" begin
    @testset for S in (Float32, Float64, ComplexF32, ComplexF64)
        A = randn(S, (n,n))
        rowiter = rows(A)
        @test typeof(first(rowiter)) == eltype(rowiter)
        @test all(t->t[1]==t[2], zip(rowiter, [A[i,:] for i=1:n]))
        coliter = cols(A)
        @test typeof(first(coliter)) == eltype(coliter)
        @test all(t->t[1]==t[2], zip(coliter, [A[:,i] for i=1:n]))
    end
end

@testset "Dense Schur factorisation and associated methods" begin
    @testset for S in (Float32, Float64, ComplexF32, ComplexF64)
        H = convert(Matrix, hessenberg(rand(S,n,n)).H) # convert for compatibility with 1.3

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
end
