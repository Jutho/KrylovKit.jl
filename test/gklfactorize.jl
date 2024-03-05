wrapop(A) = function (v, flag=Val(false))
    if flag === Val(true)
        return MinimalVec{isinplace(v)}(A' * unwrap(v))
    else
        return MinimalVec{isinplace(v)}(A * unwrap(v))
    end
end

function wrapop2(A)
    return (x -> MinimalVec{false}(A * unwrap(x)),
            y -> MinimalVec{true}(A' * unwrap(y)))
end

# Test complete Golub-Kahan-Lanczos factorization
@testset "Complete Golub-Kahan-Lanczos factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (n, n))
            v = A * rand(T, (n,)) # ensure v is in column space of A
            iter = GKLIterator(A, v, orth)
            verbosity = 1
            fact = @constinferred initialize(iter; verbosity=verbosity)
            while length(fact) < n
                @constinferred expand!(iter, fact; verbosity=verbosity)
                verbosity = 0
            end

            U = stack(basis(fact, :U))
            V = stack(basis(fact, :V))
            B = rayleighquotient(fact)
            @test normres(fact) < 10 * n * eps(real(T))
            @test U' * U ≈ I
            @test V' * V ≈ I
            @test A * V ≈ U * B
            @test A' * U ≈ V * B'

            @constinferred initialize!(iter, deepcopy(fact); verbosity=1)
            states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
            @test rayleighquotient(last(states)) ≈ B
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (n, n))
        v = MinimalVec{IP}(A * rand(T, (n,))) # ensure v is in column space of A
        iter = GKLIterator(wrapop(A), v)
        verbosity = 1
        fact = @constinferred initialize(iter; verbosity=verbosity)
        while length(fact) < n
            @constinferred expand!(iter, fact; verbosity=verbosity)
            verbosity = 0
        end

        U = stack(unwrap, basis(fact, :U))
        V = stack(unwrap, basis(fact, :V))
        B = rayleighquotient(fact)
        @test normres(fact) < 10 * n * eps(real(T))
        @test V' * V ≈ I
        @test U' * U ≈ I
        @test A * V ≈ U * B
        @test A' * U ≈ V * B'

        @constinferred initialize!(iter, deepcopy(fact); verbosity=1)
        states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
        @test rayleighquotient(last(states)) ≈ B
    end

    @testset "MixedVec" begin
        T = ComplexF64
        A = rand(T, (n, n))
        v = MinimalVec{false}(A * rand(T, (n,))) # ensure v is in column space of A

        iter = GKLIterator(wrapop2(A), v)
        verbosity = 1
        fact = @constinferred initialize(iter; verbosity=verbosity)
        while length(fact) < n
            @constinferred expand!(iter, fact; verbosity=verbosity)
            verbosity = 0
        end

        U = stack(unwrap, basis(fact, :U))
        V = stack(unwrap, basis(fact, :V))
        B = rayleighquotient(fact)
        @test normres(fact) < 10 * n * eps(real(T))
        @test V' * V ≈ I
        @test U' * U ≈ I
        @test A * V ≈ U * B
        @test A' * U ≈ V * B'

        @constinferred initialize!(iter, deepcopy(fact); verbosity=1)
        states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
        @test rayleighquotient(last(states)) ≈ B
    end
end

# Test incomplete Golub-Kahan-Lanczos factorization
@testset "Incomplete Golub-Kahan-Lanczos factorization" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64, Complex{Int})
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            if T == Complex{Int}
                A = rand(-100:100, (N, N)) + im * rand(-100:100, (N, N))
                v = rand(-100:100, (N,))
            else
                A = rand(T, (N, N))
                v = rand(T, (N,))
            end
            iter = @constinferred GKLIterator(A, v, orth)
            krylovdim = 3 * n
            fact = @constinferred initialize(iter)
            while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
                @constinferred expand!(iter, fact)

                Ũ, Ṽ, B, r, β, e = fact
                U = stack(Ũ)
                V = stack(Ṽ)
                @test U' * U ≈ I
                @test V' * V ≈ I
                @test norm(r) ≈ β
                @test A * V ≈ U * B + r * e'
                @test A' * U ≈ V * B'
            end

            fact = @constinferred shrink!(fact, div(n, 2))
            U = stack(@constinferred basis(fact, :U))
            V = stack(@constinferred basis(fact, :V))
            B = @constinferred rayleighquotient(fact)
            r = @constinferred residual(fact)
            β = @constinferred normres(fact)
            e = @constinferred rayleighextension(fact)
            @test U' * U ≈ I
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ U * B + r * e'
            @test A' * U ≈ V * B'
        end
    end

    @testset "MinimalVec{$IP}" for IP in (true, false)
        T = ComplexF64
        A = rand(T, (N, N))
        v = MinimalVec{IP}(rand(T, (N,)))
        iter = @constinferred GKLIterator(wrapop(A), v)
        krylovdim = 3 * n
        fact = @constinferred initialize(iter)
        while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
            @constinferred expand!(iter, fact)
            Ũ, Ṽ, B, r̃, β, e = fact
            U = stack(unwrap, Ũ)
            V = stack(unwrap, Ṽ)
            r = unwrap(r̃)
            @test U' * U ≈ I
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ U * B + r * e'
            @test A' * U ≈ V * B'
        end

        fact = @constinferred shrink!(fact, div(n, 2))
        U = stack(unwrap, @constinferred basis(fact, :U))
        V = stack(unwrap, @constinferred basis(fact, :V))
        B = @constinferred rayleighquotient(fact)
        r = unwrap(@constinferred residual(fact))
        β = @constinferred normres(fact)
        e = @constinferred rayleighextension(fact)
        @test U' * U ≈ I
        @test V' * V ≈ I
        @test norm(r) ≈ β
        @test A * V ≈ U * B + r * e'
        @test A' * U ≈ V * B'
    end

    @testset "MixedVec" begin
        T = ComplexF64
        A = rand(T, (N, N))
        v = MinimalVec{false}(rand(T, (N,)))
        iter = @constinferred GKLIterator(wrapop2(A), v)
        krylovdim = 3 * n
        fact = @constinferred initialize(iter)
        while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
            @constinferred expand!(iter, fact)
            Ũ, Ṽ, B, r̃, β, e = fact
            U = stack(unwrap, Ũ)
            V = stack(unwrap, Ṽ)
            r = unwrap(r̃)
            @test U' * U ≈ I
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ U * B + r * e'
            @test A' * U ≈ V * B'
        end

        fact = @constinferred shrink!(fact, div(n, 2))
        U = stack(unwrap, @constinferred basis(fact, :U))
        V = stack(unwrap, @constinferred basis(fact, :V))
        B = @constinferred rayleighquotient(fact)
        r = unwrap(@constinferred residual(fact))
        β = @constinferred normres(fact)
        e = @constinferred rayleighextension(fact)
        @test U' * U ≈ I
        @test V' * V ≈ I
        @test norm(r) ≈ β
        @test A * V ≈ U * B + r * e'
        @test A' * U ≈ V * B'
    end
end
