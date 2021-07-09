function ≊(list1::AbstractVector, list2::AbstractVector)
    length(list1) == length(list2) || return false
    n = length(list1)
    ind2 = collect(1:n)
    p = sizehint!(Int[], n)
    for i = 1:n
        j = argmin(abs.(view(list2, ind2) .- list1[i]))
        p = push!(p, ind2[j])
        ind2 = deleteat!(ind2, j)
    end
    return list1 ≈ view(list2, p)
end

@testset "Lanczos - eigsolve full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            A = (A+A')/2
            v = rand(T,(n,))
            alg = Lanczos(orth = orth, krylovdim = 2*n, maxiter = 1, tol = 10*n*eps(real(T)))
            n1 = div(n,2)
            D1, V1, info = eigsolve(wrapop(A), wrapvec(v), n1, :SR; orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            @test KrylovKit.eigselector(A, eltype(v); orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T))) isa Lanczos
            n2 = n-n1
            D2, V2, info = @inferred eigsolve(wrapop(A), wrapvec(v), n2, :LR, alg)
            @test vcat(D1[1:n1],reverse(D2[1:n2])) ≊ eigvals(A)

            U1 = hcat(unwrapvec.(V1)...)
            U2 = hcat(unwrapvec.(V2)...)
            @test U1'*U1 ≈ I
            @test U2'*U2 ≈ I

            @test A*U1 ≈ U1*Diagonal(D1)
            @test A*U2 ≈ U2*Diagonal(D2)
        end
    end
end

@testset "Lanczos - eigsolve iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            A = (A+A')/2
            v = rand(T,(N,))
            alg = Lanczos(orth = orth, krylovdim = n, maxiter = 18,
                            tol = 10*n*eps(real(T)), eager = true)
            n1 = div(n,2)
            D1, V1, info1 = @inferred eigsolve(wrapop(A), wrapvec(v), n1, :SR, alg)
            n2 = n-n1
            D2, V2, info2 = eigsolve(wrapop(A), wrapvec(v), n2, :LR, alg)

            l1 = info1.converged
            l2 = info2.converged
            @test D1[1:l1] ≈ eigvals(A)[1:l1]
            @test D2[1:l2] ≈ eigvals(A)[N:-1:N-l2+1]

            U1 = hcat(unwrapvec.(V1)...)
            U2 = hcat(unwrapvec.(V2)...)
            @test U1'*U1 ≈ I
            @test U2'*U2 ≈ I

            R1 = hcat(unwrapvec.(info1.residual)...)
            R2 = hcat(unwrapvec.(info2.residual)...)
            @test A*U1 ≈ U1*Diagonal(D1) + R1
            @test A*U2 ≈ U2*Diagonal(D2) + R2
        end
    end
end

@testset "Arnoldi - eigsolve full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            v = rand(T,(n,))
            alg = Arnoldi(orth = orth, krylovdim = 2*n, maxiter = 1, tol = 10*n*eps(real(T)))
            n1 = div(n,2)
            D1, V1, info1 = eigsolve(wrapop(A), wrapvec(v), n1, :SR; orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            @test KrylovKit.eigselector(A, eltype(v); orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T))) isa Arnoldi
            n2 = n-n1
            D2, V2, info2 = @inferred eigsolve(wrapop(A), wrapvec(v), n2, :LR, alg)
            D = sort(sort(eigvals(A), by=imag, rev=true), alg=MergeSort, by=real)
            D2′ = sort(sort(D2, by=imag, rev=true), alg=MergeSort, by=real)
            @test vcat(D1[1:n1],D2′[end-n2+1:end]) ≈ D

            U1 = hcat(unwrapvec.(V1)...)
            U2 = hcat(unwrapvec.(V2)...)
            @test A*U1 ≈ U1*Diagonal(D1)
            @test A*U2 ≈ U2*Diagonal(D2)

            if T<:Complex
                n1 = div(n,2)
                D1, V1, info = eigsolve(wrapop(A), wrapvec(v), n1, :SI, alg)
                n2 = n-n1
                D2, V2, info = eigsolve(wrapop(A), wrapvec(v), n2, :LI, alg)
                D = sort(eigvals(A), by=imag)

                @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ D

                U1 = hcat(unwrapvec.(V1)...)
                U2 = hcat(unwrapvec.(V2)...)
                @test A*U1 ≈ U1*Diagonal(D1)
                @test A*U2 ≈ U2*Diagonal(D2)
            end
        end
    end
end

@testset "Arnoldi - eigsolve iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            v = rand(T,(N,))
            alg = Arnoldi(orth = orth, krylovdim = 3*n, maxiter = 10,
                            tol = 10*n*eps(real(T)), eager = true)
            D1, V1, info1 = @inferred eigsolve(wrapop(A), wrapvec(v), n, :SR, alg)
            D2, V2, info2 = eigsolve(wrapop(A), wrapvec(v), n, :LR, alg)
            D3, V3, info3 = eigsolve(wrapop(A), wrapvec(v), n, :LM, alg)
            D = sort(eigvals(A), by=imag, rev=true)

            l1 = info1.converged
            l2 = info2.converged
            l3 = info3.converged
            @test D1[1:l1] ≊ sort(D, alg=MergeSort, by=real)[1:l1]
            @test D2[1:l2] ≊ sort(D, alg=MergeSort, by=real, rev=true)[1:l2]
            # sorting by abs does not seem very reliable if two distinct eigenvalues are close
            # in absolute value, so we perform a second sort afterwards using the real part
            @test D3[1:l3] ≊ sort(D, by = abs, rev=true)[1:l3]

            U1 = hcat(unwrapvec.(V1)...)
            U2 = hcat(unwrapvec.(V2)...)
            U3 = hcat(unwrapvec.(V3)...)
            R1 = hcat(unwrapvec.(info1.residual)...)
            R2 = hcat(unwrapvec.(info2.residual)...)
            R3 = hcat(unwrapvec.(info3.residual)...)
            @test A*U1 ≈ U1*Diagonal(D1) + R1
            @test A*U2 ≈ U2*Diagonal(D2) + R2
            @test A*U3 ≈ U3*Diagonal(D3) + R3

            if T<:Complex
                D1, V1, info1 = eigsolve(wrapop(A), wrapvec(v), n, :SI, alg)
                D2, V2, info2 = eigsolve(wrapop(A), wrapvec(v), n, :LI, alg)
                D = eigvals(A)

                l1 = info1.converged
                l2 = info2.converged
                @test D1[1:l1] ≈ sort(D, by=imag)[1:l1]
                @test D2[1:l2] ≈ sort(D, by=imag, rev=true)[1:l2]

                U1 = hcat(unwrapvec.(V1)...)
                U2 = hcat(unwrapvec.(V2)...)
                R1 = hcat(unwrapvec.(info1.residual)...)
                R2 = hcat(unwrapvec.(info2.residual)...)
                @test A*U1 ≈ U1*Diagonal(D1) + R1
                @test A*U2 ≈ U2*Diagonal(D2) + R2
            end
        end
    end
end

@testset "GolubYe - geneigsolve full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            A = (A+A')/2
            B = rand(T,(n,n)) .- one(T)/2
            B = sqrt(B*B')
            v = rand(T,(n,))
            alg = GolubYe(orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            n1 = div(n,2)
            D1, V1, info = geneigsolve((A, B), n1, :SR; orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            @test KrylovKit.geneigselector((A, B), eltype(v); orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T))) isa GolubYe
            n2 = n-n1
            D2, V2, info = @inferred geneigsolve((A, B), v, n2, :LR, alg)
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

@testset "GolubYe - geneigsolve iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            A = (A+A')/2
            B = rand(T,(N,N)) .- one(T)/2
            B = sqrt(B*B')
            v = rand(T,(N,))
            alg = GolubYe(orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            n1 = div(n,2)
            D1, V1, info1 = @inferred geneigsolve((A, B), v, n1, :SR, alg)
            n2 = n-n1
            D2, V2, info2 = geneigsolve((A, B), v, n2, :LR, alg)

            l1 = info1.converged
            l2 = info2.converged
            @test D1[1:l1] ≊ eigvals(A, B)[1:l1]
            @test D2[1:l2] ≊ eigvals(A, B)[N:-1:N-l2+1]

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
