# Test complete Lanczos factorization
wrapop(A) = function (v::MinimalVec)
    return MinimalVec{isinplace(v)}(A * unwrap(v))
end

@testset "Complete Lanczos factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr) # tests fail miserably for cgs and mgs
            A = rand(T, (n, n))
            v = rand(T, (n,))
            A = (A + A')
            iter = LanczosIterator(A, v, orth)
            verbosity = 1
            fact = @constinferred initialize(iter; verbosity=verbosity)
            while length(fact) < n
                @constinferred expand!(iter, fact; verbosity=verbosity)
                verbosity = 0
            end

            V = stack(basis(fact))
            H = rayleighquotient(fact)
            @test normres(fact) < 10 * n * eps(real(T))
            @test V' * V ≈ I
            @test A * V ≈ V * H

            @constinferred initialize!(iter, deepcopy(fact); verbosity=1)
            states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
            @test rayleighquotient(last(states)) ≈ H
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (n, n))
        A += A'

        v = MinimalVec{IP}(rand(T, (n,)))
        iter = LanczosIterator(wrapop(A), v)
        verbosity = 1
        fact = @constinferred initialize(iter; verbosity=verbosity)
        while length(fact) < n
            @constinferred expand!(iter, fact; verbosity=verbosity)
            verbosity = 0
        end

        V = stack(unwrap, basis(fact))
        H = rayleighquotient(fact)
        @test normres(fact) < 10 * n * eps(real(T))
        @test V' * V ≈ I
        @test A * V ≈ V * H

        @constinferred initialize!(iter, deepcopy(fact); verbosity=1)
        states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
        @test rayleighquotient(last(states)) ≈ H
    end
end

# Test complete Arnoldi factorization
@testset "Complete Arnoldi factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (n, n))
            v = rand(T, (n,))
            iter = ArnoldiIterator(A, v, orth)
            verbosity = 1
            fact = @constinferred initialize(iter; verbosity=verbosity)
            while length(fact) < n
                @constinferred expand!(iter, fact; verbosity=verbosity)
                verbosity = 0
            end

            V = stack(basis(fact))
            H = rayleighquotient(fact)
            factor = (orth == cgs || orth == mgs ? 250 : 10)
            @test normres(fact) < factor * n * eps(real(T))
            @test V' * V ≈ I
            @test A * V ≈ V * H

            @constinferred initialize!(iter, deepcopy(fact); verbosity=1)
            states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
            @test rayleighquotient(last(states)) ≈ H
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (n, n))

        v = MinimalVec{IP}(rand(T, (n,)))
        iter = ArnoldiIterator(wrapop(A), v)
        verbosity = 1
        fact = @constinferred initialize(iter; verbosity=verbosity)
        while length(fact) < n
            @constinferred expand!(iter, fact; verbosity=verbosity)
            verbosity = 0
        end

        V = stack(unwrap, basis(fact))
        H = rayleighquotient(fact)
        @test normres(fact) < 10 * n * eps(real(T))
        @test V' * V ≈ I
        @test A * V ≈ V * H

        @constinferred initialize!(iter, deepcopy(fact); verbosity=1)
        states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
        @test rayleighquotient(last(states)) ≈ H
    end
end

# Test incomplete Lanczos factorization
@testset "Incomplete Lanczos factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64, Complex{Int})
        @testset for orth in (cgs2, mgs2, cgsr, mgsr) # tests fail miserably for cgs and mgs
            if T == Complex{Int}
                A = rand(-100:100, (N, N)) + im * rand(-100:100, (N, N))
                v = rand(-100:100, (N,))
            else
                A = rand(T, (N, N))
                v = rand(T, (N,))
            end
            A = (A + A')
            iter = @constinferred LanczosIterator(A, v, orth)
            krylovdim = n
            fact = @constinferred initialize(iter)
            while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
                @constinferred expand!(iter, fact)

                Ṽ, H, r, β, e = fact
                V = stack(Ṽ)
                @test V' * V ≈ I
                @test norm(r) ≈ β
                @test A * V ≈ V * H + r * e'
            end

            fact = @constinferred shrink!(fact, div(n, 2))
            B = @constinferred basis(fact)
            V = stack(B)
            H = @constinferred rayleighquotient(fact)
            r = @constinferred residual(fact)
            β = @constinferred normres(fact)
            e = @constinferred rayleighextension(fact)
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ V * H + r * e'
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (N, N))
        A += A'

        v = MinimalVec{IP}(rand(T, (N,)))
        iter = @constinferred LanczosIterator(wrapop(A), v)
        krylovdim = n
        fact = @constinferred initialize(iter)
        while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
            @constinferred expand!(iter, fact)

            Ṽ, H, r̃, β, e = fact
            V = stack(unwrap, Ṽ)
            r = unwrap(r̃)
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ V * H + r * e'
        end

        fact = @constinferred shrink!(fact, div(n, 2))
        V = stack(unwrap, @constinferred basis(fact))
        H = @constinferred rayleighquotient(fact)
        r = unwrap(@constinferred residual(fact))
        β = @constinferred normres(fact)
        e = @constinferred rayleighextension(fact)
        @test V' * V ≈ I
        @test norm(r) ≈ β
        @test A * V ≈ V * H + r * e'
    end
end

# Test incomplete Arnoldi factorization
@testset "Incomplete Arnoldi factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64, Complex{Int})
        @testset for orth in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
            if T == Complex{Int}
                A = rand(-100:100, (N, N)) + im * rand(-100:100, (N, N))
                v = rand(-100:100, (N,))
            else
                A = rand(T, (N, N))
                v = rand(T, (N,))
            end
            iter = @constinferred ArnoldiIterator(A, v, orth)
            krylovdim = 3 * n
            fact = @constinferred initialize(iter)
            while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
                @constinferred expand!(iter, fact)

                Ṽ, H, r, β, e = fact
                V = stack(Ṽ)
                @test V' * V ≈ I
                @test norm(r) ≈ β
                @test A * V ≈ V * H + r * e'
            end

            fact = @constinferred shrink!(fact, div(n, 2))
            V = stack(@constinferred basis(fact))
            H = @constinferred rayleighquotient(fact)
            r = @constinferred residual(fact)
            β = @constinferred normres(fact)
            e = @constinferred rayleighextension(fact)
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ V * H + r * e'
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (N, N))

        v = MinimalVec{IP}(rand(T, (N,)))
        iter = @constinferred ArnoldiIterator(wrapop(A), v)
        krylovdim = 3 * n
        fact = @constinferred initialize(iter)
        while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
            @constinferred expand!(iter, fact)

            Ṽ, H, r̃, β, e = fact
            V = stack(unwrap, Ṽ)
            r = unwrap(r̃)
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ V * H + r * e'
        end

        fact = @constinferred shrink!(fact, div(n, 2))
        V = stack(unwrap, @constinferred basis(fact))
        H = @constinferred rayleighquotient(fact)
        r = unwrap(@constinferred residual(fact))
        β = @constinferred normres(fact)
        e = @constinferred rayleighextension(fact)
        @test V' * V ≈ I
        @test norm(r) ≈ β
        @test A * V ≈ V * H + r * e'
    end
end
