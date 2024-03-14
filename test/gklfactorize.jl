# Test complete Golub-Kahan-Lanczos factorization
@testset "Complete Golub-Kahan-Lanczos factorization ($mode)" for mode in
                                                                  (:vector, :inplace,
                                                                   :outplace, :mixed)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n))
            v = A * rand(T, (n,)) # ensure v is in column space of A
            iter = GKLIterator(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), orth)
            verbosity = 1
            fact = @constinferred initialize(iter; verbosity=verbosity)
            while length(fact) < n
                @constinferred expand!(iter, fact; verbosity=verbosity)
                verbosity = 0
            end

            U = stack(unwrapvec, basis(fact, :U))
            V = stack(unwrapvec, basis(fact, :V))
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
end

# Test incomplete Golub-Kahan-Lanczos factorization
@testset "Incomplete Golub-Kahan-Lanczos factorization ($mode)" for mode in
                                                                    (:vector, :inplace,
                                                                     :outplace, :mixed)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            if T == Complex{Int}
                A = rand(-100:100, (N, N)) + im * rand(-100:100, (N, N))
                v = rand(-100:100, (N,))
            else
                A = rand(T, (N, N))
                v = rand(T, (N,))
            end
            iter = @constinferred GKLIterator(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                              orth)
            krylovdim = 3 * n
            fact = @constinferred initialize(iter)
            while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
                @constinferred expand!(iter, fact)
                Ũ, Ṽ, B, r̃, β, e = fact
                U = stack(unwrapvec, Ũ)
                V = stack(unwrapvec, Ṽ)
                r = unwrapvec(r̃)
                @test U' * U ≈ I
                @test V' * V ≈ I
                @test norm(r) ≈ β
                @test A * V ≈ U * B + r * e'
                @test A' * U ≈ V * B'
            end

            fact = @constinferred shrink!(fact, div(n, 2))
            U = stack(unwrapvec, @constinferred basis(fact, :U))
            V = stack(unwrapvec, @constinferred basis(fact, :V))
            B = @constinferred rayleighquotient(fact)
            r = unwrapvec(@constinferred residual(fact))
            β = @constinferred normres(fact)
            e = @constinferred rayleighextension(fact)
            @test U' * U ≈ I
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ U * B + r * e'
            @test A' * U ≈ V * B'
        end
    end
end
