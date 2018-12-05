@testset "GolubYe - eigsolve full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            A = (A+A')/2
            B = rand(T,(n,n)) .- one(T)/2
            B = sqrt(B*B')
            v = rand(T,(n,))
            alg = GolubYe(orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            n1 = div(n,2)
            D1, V1, info = geneigsolve(A, B, n1, :SR; orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            @test KrylovKit.geneigselector(A, B, eltype(v); orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T))) isa GolubYe
            n2 = n-n1
            D2, V2, info = @inferred geneigsolve(A, B, v, n2, :LR, alg)
            @test vcat(D1[1:n1],reverse(D2[1:n2])) ≈ eigvals(A, B)

            U1 = hcat(V1...)
            U2 = hcat(V2...)
            @test U1'*B*U1 ≈ I
            @test U2'*B*U2 ≈ I

            @test A*U1 ≈ B*U1*Diagonal(D1)
            @test A*U2 ≈ B*U2*Diagonal(D2)
        end
    end
end

@testset "GolubYe - eigsolve iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            A = (A+A')/2
            B = rand(T,(N,N)) .- one(T)/2
            B = sqrt(B*B')
            v = rand(T,(N,))
            alg = GolubYe(orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            n1 = div(n,2)
            D1, V1, info1 = @inferred geneigsolve(A, B, v, n1, :SR, alg)
            n2 = n-n1
            D2, V2, info2 = geneigsolve(A, B, v, n2, :LR, alg)

            l1 = info1.converged
            l2 = info2.converged
            @test D1[1:l1] ≈ eigvals(A, B)[1:l1]
            @test D2[1:l2] ≈ eigvals(A, B)[N:-1:N-l2+1]

            U1 = hcat(V1...)
            U2 = hcat(V2...)
            @test U1'*B*U1 ≈ I
            @test U2'*B*U2 ≈ I

            R1 = hcat(info1.residual...)
            R2 = hcat(info2.residual...)
            @test A*U1 ≈ B*U1*Diagonal(D1) + R1
            @test A*U2 ≈ B*U2*Diagonal(D2) + R2
        end
    end
end
