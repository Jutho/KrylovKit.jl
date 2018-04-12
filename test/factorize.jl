# Test complete Lanczos factorization
@testset "Complete Lanczos factorization" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr) # tests fail miserably for cgs and mgs
            A = rand(T,(n,n))
            v = rand(T,(n,))
            A = (A+A')
            iter = LanczosIterator(A, v, orth)
            fact = start(iter)
            while !done(iter, fact) && length(fact) < n
                _, fact = next(iter, fact)
            end

            V = hcat(basis(fact)...)
            H = rayleighquotient(fact)
            @test normres(fact) < 10*n*eps(real(T))
            G = V'*V
            @test G ≈ one(G)
            @test A*V ≈ V*H
        end
    end
end

# Test complete Arnoldi factorization
@testset "Complete Arnoldi factorization" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n))
            v = rand(T,(n,))
            iter = ArnoldiIterator(A, v, orth)
            fact = start(iter)
            while !done(iter, fact) && length(fact) < n
                _, fact = next(iter, fact)
            end

            V = hcat(basis(fact)...)
            H = rayleighquotient(fact)
            factor = (orth == cgs || orth == mgs ? 100 : 10)
            @test normres(fact) < factor*n*eps(real(T))
            G = V'*V
            @test G ≈ one(G)
            @test A*V ≈ V*H
        end
    end
end

# Test incomplete Lanczos factorization
@testset "Incomplete Lanczos factorization" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr) # tests fail miserably for cgs and mgs
            A = rand(T,(N,N))
            v = rand(T,(N,))
            A = (A+A')
            iter = @inferred LanczosIterator(A, v, orth)
            krylovdim = n
            fact = @inferred start(iter)
            @inferred next(iter, fact)
            @inferred done(iter, fact)
            while !done(iter, fact) && length(fact) < krylovdim
                _, fact = next(iter, fact)
            end

            V = hcat(basis(fact)...)
            H = @inferred rayleighquotient(fact)
            r = @inferred residual(fact)
            β = @inferred normres(fact)
            e = zeros(T,n)
            e[n] = one(T)
            G = V[:,1:n]'*V[:,1:n]
            @test G ≈ one(G)
            @test norm(r) ≈ β
            @test A*V ≈ V*H + r*e'
        end
    end
end

# Test incomplete Arnoldi factorization
@testset "Incomplete Arnoldi factorization" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N))
            v = rand(T,(N,))
            iter = @inferred ArnoldiIterator(A, v, orth)
            krylovdim = n
            fact = @inferred start(iter)
            @inferred next(iter, fact)
            @inferred done(iter, fact)
            while !done(iter, fact) && length(fact) < krylovdim
                _, fact = next(iter, fact)
            end

            V = hcat(basis(fact)...)
            H = @inferred rayleighquotient(fact)
            r = @inferred residual(fact)
            β = @inferred normres(fact)
            e = zeros(T,n)
            e[n] = one(T)
            G = V[:,1:n]'*V[:,1:n]
            @test G ≈ one(G)
            @test norm(r) ≈ β
            @test A*V ≈ V*H + r*e'
        end
    end
end
