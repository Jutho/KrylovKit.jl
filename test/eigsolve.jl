# Test lanczos eigsolve full
@testset "Lanczos - Eigenvalue full" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            A = (A+A')/2
            v = rand(T,(n,))
            alg = Lanczos(orth; krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            n1 = div(n,2)
            D1, V1, info = @inferred eigsolve(A, v, n1, :SR, alg)
            n2 = n-n1
            D2, V2, info = eigsolve(A, v, n2, :LR, alg)
            @test vcat(D1,reverse(D2)) ≈ eigvals(A)
            U1 = hcat(V1...)
            U2 = hcat(V2...)
            @test A*U1 ≈ U1*Diagonal(D1)
            @test A*U2 ≈ U2*Diagonal(D2)
        end
    end
end

@testset "Lanczos - Eigenvalue iteratively" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            A = (A+A')/2
            v = rand(T,(N,))
            alg = Lanczos(orth; krylovdim = n, maxiter = 300, tol = 10*n*eps(real(T)))
            n1 = div(n,2)
            D1, V1, info1 = @inferred eigsolve(A, v, n1, :SR, alg)
            n2 = n-n1
            D2, V2, info2 = eigsolve(A, v, n2, :LR, alg)
            @test D1 ≈ eigvals(A)[1:n1]
            @test D2 ≈ eigvals(A)[N:-1:N-n2+1]
            U1 = hcat(V1...)
            U2 = hcat(V2...)
            R1 = hcat(info1.residual...)
            R2 = hcat(info2.residual...)
            @test A*U1 ≈ U1*Diagonal(D1) + R1
            @test A*U2 ≈ U2*Diagonal(D2) + R2
        end
    end
end

# Test lanczos eigsolve full
@testset "Arnoldi - Eigenvalue full" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            v = rand(T,(n,))
            alg = Arnoldi(orth; krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            n1 = div(n,2)
            D1, V1, info1 = @inferred eigsolve(A, v, n1, :SR, alg)
            n2 = n-n1
            D2, V2, info2 = eigsolve(A, v, n2, :LR, alg)
            D = sort(sort(eigvals(A), by=imag, rev=true), alg=MergeSort, by=real)
            if length(D1) > n1
                D2′ = sort(sort(D2[1:end-2], by=imag, rev=true), alg=MergeSort, by=real)
                @test vcat(D1,D2′) ≈ D
            else
                D2′ = sort(sort(D2, by=imag, rev=true), alg=MergeSort, by=real)
                @test vcat(D1,D2′) ≈ D
            end
            U1 = hcat(V1...)
            U2 = hcat(V2...)
            @test A*U1 ≈ U1*Diagonal(D1)
            @test A*U2 ≈ U2*Diagonal(D2)

            if T<:Complex
                n1 = div(n,2)
                D1, V1, info = eigsolve(A, v, n1, :SI, alg)
                n2 = n-n1
                D2, V2, info = eigsolve(A, v, n2, :LI, alg)
                D = sort(eigvals(A), by=imag)
                @test vcat(D1,reverse(D2)) ≈ D
                U1 = hcat(V1...)
                U2 = hcat(V2...)
                @test A*U1 ≈ U1*Diagonal(D1)
                @test A*U2 ≈ U2*Diagonal(D2)
            end
        end
    end
end

@testset "Arnoldi - Eigenvalue iteratively" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            v = rand(T,(N,))
            alg = Arnoldi(orth; krylovdim = 3*n, maxiter = 300, tol = 10*n*eps(real(T)))
            D1, V1, info1 = @inferred eigsolve(A, v, n, :SR, alg)
            D2, V2, info2 = eigsolve(A, v, n, :LR, alg)
            D3, V3, info3 = eigsolve(A, v, n, :LM, alg)
            D = sort(eigvals(A), by=imag, rev=true)
            @test D1 ≈ sort(D, alg=MergeSort, by=real)[1:length(D1)]
            @test D2 ≈ sort(D, alg=MergeSort, by=real, rev=true)[1:length(D2)]
            @test D3 ≈ sort(D, alg=MergeSort, by=abs, rev=true)[1:length(D3)]

            U1 = hcat(V1...)
            U2 = hcat(V2...)
            U3 = hcat(V3...)
            R1 = hcat(info1.residual...)
            R2 = hcat(info2.residual...)
            R3 = hcat(info3.residual...)
            @test A*U1 ≈ U1*Diagonal(D1) + R1
            @test A*U2 ≈ U2*Diagonal(D2) + R2
            @test A*U3 ≈ U3*Diagonal(D3) + R3

            if T<:Complex
                D1, V1, info1 = eigsolve(A, v, n, :SI, alg)
                D2, V2, info2 = eigsolve(A, v, n, :LI, alg)
                D = eigvals(A)
                @test D1 ≈ sort(D, by=imag)[1:n]
                @test D2 ≈ sort(D, by=imag, rev=true)[1:n]

                U1 = hcat(V1...)
                U2 = hcat(V2...)
                R1 = hcat(info1.residual...)
                R2 = hcat(info2.residual...)
                @test A*U1 ≈ U1*Diagonal(D1) + R1
                @test A*U2 ≈ U2*Diagonal(D2) + R2
            end
        end
    end
end
