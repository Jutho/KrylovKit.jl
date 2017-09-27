# Test complete Lanczos factorization
for T in (Float32, Float64, Complex64, Complex128)
    for orth in (cgs2, mgs2, cgsr, mgsr) # tests fail miserably for cgs and mgs
        A = rand(T,(n,n))
        v = rand(T,(n,))
        A = (A+A')
        iter = LanczosIterator(A, v, orth)
        s = start(iter)
        while !done(iter, s)
            fact, s = next(iter, s)
        end

        V = hcat(basis(s)...)[:,1:n]
        H = matrix(s)
        @test normres(s) < 10*n*eps(real(T))
        @test V'*V ≈ one(V)
        @test A*V ≈ V*H
    end
end

# Test complete Arnoldi factorization
for T in (Float32, Float64, Complex64, Complex128)
    for orth in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
        A = rand(T,(n,n))
        v = rand(T,(n,))
        iter = ArnoldiIterator(A, v, orth)
        s = start(iter)
        while !done(iter, s)
            fact, s = next(iter, s)
        end

        V = hcat(basis(s)...)[:,1:n]
        H = matrix(s)
        @test normres(s) < 10*n*eps(real(T))
        @test V'*V ≈ one(V)
        @test A*V[:,1:n] ≈ V[:,1:n]*H
    end
end

# Test incomplete Lanczos factorization
for T in (Float32, Float64, Complex64, Complex128)
    for orth in (cgs2, mgs2, cgsr, mgsr) # tests fail miserably for cgs and mgs
        A = rand(T,(N,N))
        v = rand(T,(N,))
        A = (A+A')
        iter = LanczosIterator(A, v, orth; krylovdim = n)
        s = start(iter)

        @inferred LanczosIterator(A, v, orth; krylovdim = n)
        @inferred start(iter)
        @inferred next(iter, s)
        @inferred done(iter, s)

        while !done(iter, s)
            fact, s = next(iter, s)
        end

        V = hcat(basis(s)...)
        H = zeros(T,(n+1,n))
        H[1:n,:] = matrix(s)
        H[n+1,n] = normres(s)
        @test V[:,1:n]'*V[:,1:n] ≈ eye(T,n)
        @test A*V[:,1:n] ≈ V*H
    end
end

# Test incomplete Arnoldi factorization
for T in (Float32, Float64, Complex64, Complex128)
    for orth in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
        A = rand(T,(N,N))
        v = rand(T,(N,))
        iter = ArnoldiIterator(A, v, orth; krylovdim = n)
        s = start(iter)

        @inferred ArnoldiIterator(A, v, orth; krylovdim = n)
        @inferred start(iter)
        @inferred next(iter, s)
        @inferred done(iter, s)

        while !done(iter, s)
            fact, s = next(iter, s)
        end

        V = hcat(basis(s)...)
        H = zeros(T,(n+1,n))
        H[1:n,:] = matrix(s)
        H[n+1,n] = normres(s)
        @test V[:,1:n]'*V[:,1:n] ≈ eye(T,n)
        @test A*V[:,1:n] ≈ V*H
    end
end
