# Test complete Lanczos factorization
@testset "Complete Lanczos factorization" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr) # tests fail miserably for cgs and mgs
            A = rand(T,(n,n))
            v = rand(T,(n,))
            A = (A+A')
            iter = LanczosIterator(A, v, orth)
            fact = start(iter)
            while !done(iter, fact)
                nr, fact = next(iter, fact)
            end

            V = hcat(basis(fact)...)[:,1:n]
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
            s = start(iter)
            while !done(iter, s)
                fact, s = next(iter, s)
            end

            V = hcat(basis(s)...)[:,1:n]
            H = rayleighquotient(s)
            factor = (orth == cgs || orth == mgs ? 100 : 10)
            @test normres(s) < factor*n*eps(real(T))
            G = V'*V
            @test G ≈ one(G)
            @test A*V[:,1:n] ≈ V[:,1:n]*H
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
            s = @inferred start(iter)
            @inferred next(iter, s)
            @inferred done(iter, s)
            while !done(iter, s) && s.k < krylovdim
                fact, s = next(iter, s)
            end

            V = hcat(basis(s)...)
            H = zeros(T,(n+1,n))
            H[1:n,:] = rayleighquotient(s)
            H[n+1,n] = normres(s)
            G = V[:,1:n]'*V[:,1:n]
            @test G ≈ one(G)
            @test A*V[:,1:n] ≈ V*H
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
            s = @inferred start(iter)
            @inferred next(iter, s)
            @inferred done(iter, s)
            while !done(iter, s) && s.k < krylovdim
                fact, s = next(iter, s)
            end

            V = hcat(basis(s)...)
            H = zeros(T,(n+1,n))
            H[1:n,:] = rayleighquotient(s)
            H[n+1,n] = normres(s)
            G = V[:,1:n]'*V[:,1:n]
            @test G ≈ one(G)
            @test A*V[:,1:n] ≈ V*H
        end
    end
end
