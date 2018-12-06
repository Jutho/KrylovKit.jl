# Test complete Lanczos factorization
@testset "Complete Lanczos factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr) # tests fail miserably for cgs and mgs
            A = rand(T,(n,n))
            v = rand(T,(n,))
            A = (A+A')
            iter = LanczosIterator(A, v, orth)
            fact = initialize(iter)
            while length(fact) < n
                expand!(iter, fact)
            end

            V = hcat(basis(fact)...)
            H = rayleighquotient(fact)
            @test normres(fact) < 10*n*eps(real(T))
            @test V'*V ≈ I
            @test A*V ≈ V*H
        end
    end
end

# Test complete Arnoldi factorization
@testset "Complete Arnoldi factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n))
            v = rand(T,(n,))
            iter = ArnoldiIterator(A, v, orth)
            fact = initialize(iter)
            while length(fact) < n
                expand!(iter, fact)
            end

            V = hcat(basis(fact)...)
            H = rayleighquotient(fact)
            factor = (orth == cgs || orth == mgs ? 100 : 10)
            @test normres(fact) < factor*n*eps(real(T))
            @test V'*V ≈ I
            @test A*V ≈ V*H
        end
    end
end

# Test complete Arnoldi factorization
@testset "Complete Golub-Kahan-Lanczos factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n))
            v = A*rand(T,(n,)) # ensure v is in column space of A
            iter = GKLIterator(A, v, orth)
            fact = initialize(iter)
            while length(fact) < n
                expand!(iter, fact)
            end

            U = hcat(basis(fact, :U)...)
            V = hcat(basis(fact, :V)...)
            B = rayleighquotient(fact)
            @test normres(fact) < 10*n*eps(real(T))
            @test U'*U ≈ I
            @test V'*V ≈ I
            @test A*V ≈ U*B
            @test A'*U ≈ V*B'
        end
    end
end

# Test incomplete Lanczos factorization
@testset "Incomplete Lanczos factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr) # tests fail miserably for cgs and mgs
            A = rand(T,(N,N))
            v = rand(T,(N,))
            A = (A+A')
            iter = @inferred LanczosIterator(A, v, orth)
            krylovdim = n
            fact = @inferred initialize(iter)
            while normres(fact) > eps(real(T)) && length(fact) < krylovdim
                @inferred expand!(iter, fact)

                V = hcat(basis(fact)...)
                H = rayleighquotient(fact)
                r = residual(fact)
                β = normres(fact)
                e = rayleighextension(fact)
                @test V'*V ≈ I
                @test norm(r) ≈ β
                @test A*V ≈ V*H + r*e'
            end

            fact = @inferred shrink!(fact, div(n,2))
            V = hcat((@inferred basis(fact))...)
            H = @inferred rayleighquotient(fact)
            r = @inferred residual(fact)
            β = @inferred normres(fact)
            e = @inferred rayleighextension(fact)
            @test V'*V ≈ I
            @test norm(r) ≈ β
            @test A*V ≈ V*H + r*e'
        end
    end
end

# Test incomplete Arnoldi factorization
@testset "Incomplete Arnoldi factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N))
            v = rand(T,(N,))
            iter = @inferred ArnoldiIterator(A, v, orth)
            krylovdim = 3*n
            fact = @inferred initialize(iter)
            while normres(fact) > eps(real(T)) && length(fact) < krylovdim
                @inferred expand!(iter, fact)

                V = hcat(basis(fact)...)
                H = rayleighquotient(fact)
                r = residual(fact)
                β = normres(fact)
                e = rayleighextension(fact)
                @test V'*V ≈ I
                @test norm(r) ≈ β
                @test A*V ≈ V*H + r*e'
            end

            fact = @inferred shrink!(fact, div(n,2))
            V = hcat((@inferred basis(fact))...)
            H = @inferred rayleighquotient(fact)
            r = @inferred residual(fact)
            β = @inferred normres(fact)
            e = @inferred rayleighextension(fact)
            @test V'*V ≈ I
            @test norm(r) ≈ β
            @test A*V ≈ V*H + r*e'
        end
    end
end

# Test incomplete Arnoldi factorization
@testset "Incomplete GKL factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N))
            v = rand(T,(N,))
            iter = @inferred GKLIterator(A, v, orth)
            krylovdim = 3*n
            fact = @inferred initialize(iter)
            while normres(fact) > eps(real(T)) && length(fact) < krylovdim
                @inferred expand!(iter, fact)

                U = hcat(basis(fact, :U)...)
                V = hcat(basis(fact, :V)...)
                B = rayleighquotient(fact)
                r = residual(fact)
                β = normres(fact)
                e = rayleighextension(fact)
                @test U'*U ≈ I
                @test V'*V ≈ I
                @test norm(r) ≈ β
                @test A*V ≈ U*B + r*e'
                @test A'*U ≈ V*B'
            end

            fact = @inferred shrink!(fact, div(n,2))
            U = hcat((@inferred basis(fact, :U))...)
            V = hcat((@inferred basis(fact, :V))...)
            B = @inferred rayleighquotient(fact)
            r = @inferred residual(fact)
            β = @inferred normres(fact)
            e = @inferred rayleighextension(fact)
            @test U'*U ≈ I
            @test V'*V ≈ I
            @test norm(r) ≈ β
            @test A*V ≈ U*B + r*e'
            @test A'*U ≈ V*B'
        end
    end
end
