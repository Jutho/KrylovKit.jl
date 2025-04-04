# Test complete Lanczos factorization
@testset "Complete Lanczos factorization ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (cgs2,)
    using KrylovKit: EACHITERATION_LEVEL

    @testset for T in scalartypes
        @testset for orth in orths # tests fail miserably for cgs and mgs
            A = rand(T, (n, n))
            v = rand(T, (n,))
            A = (A + A')
            iter = LanczosIterator(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), orth)
            fact = @constinferred initialize(iter)
            @constinferred expand!(iter, fact)
            @test_logs initialize(iter; verbosity=EACHITERATION_LEVEL)
            @test_logs (:info,) initialize(iter; verbosity=EACHITERATION_LEVEL + 1)
            verbosity = EACHITERATION_LEVEL + 1
            while length(fact) < n
                if verbosity == EACHITERATION_LEVEL + 1
                    @test_logs (:info,) expand!(iter, fact; verbosity=verbosity)
                    verbosity = EACHITERATION_LEVEL
                else
                    @test_logs expand!(iter, fact; verbosity=verbosity)
                    verbosity = EACHITERATION_LEVEL + 1
                end
            end
            V = stack(unwrapvec, basis(fact))
            H = rayleighquotient(fact)
            @test normres(fact) < 10 * n * eps(real(T))
            @test V' * V ≈ I
            @test A * V ≈ V * H

            states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
            @test rayleighquotient(last(states)) ≈ H

            @constinferred shrink!(fact, n - 1)
            @test_logs (:info,) shrink!(fact, n - 2; verbosity=EACHITERATION_LEVEL + 1)
            @test_logs shrink!(fact, n - 3; verbosity=EACHITERATION_LEVEL)
            @constinferred initialize!(iter, deepcopy(fact))
            @test_logs initialize!(iter, deepcopy(fact); verbosity=EACHITERATION_LEVEL)
            @test_logs (:info,) initialize!(iter, deepcopy(fact);
                                            verbosity=EACHITERATION_LEVEL + 1)

            if T <: Complex
                A = rand(T, (n, n)) # test warnings for non-hermitian matrices
                v = rand(T, (n,))
                iter = LanczosIterator(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), orth)
                fact = @constinferred initialize(iter; verbosity=0)
                @constinferred expand!(iter, fact; verbosity=0)
                @test_logs initialize(iter; verbosity=0)
                @test_logs (:warn,) initialize(iter)
                verbosity = 1
                while length(fact) < n
                    if verbosity == 1
                        @test_logs (:warn,) expand!(iter, fact; verbosity=verbosity)
                        verbosity = 0
                    else
                        @test_logs expand!(iter, fact; verbosity=verbosity)
                        verbosity = 1
                    end
                end
            end
        end
    end
end

# Test complete Arnoldi factorization
@testset "Complete Arnoldi factorization ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs, mgs, cgs2, mgs2, cgsr, mgsr) : (cgs2,)

    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n))
            v = rand(T, (n,))
            iter = ArnoldiIterator(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), orth)
            fact = @constinferred initialize(iter)
            @constinferred expand!(iter, fact)
            @test_logs initialize(iter; verbosity=EACHITERATION_LEVEL)
            @test_logs (:info,) initialize(iter; verbosity=EACHITERATION_LEVEL + 1)
            verbosity = EACHITERATION_LEVEL + 1
            while length(fact) < n
                if verbosity == EACHITERATION_LEVEL + 1
                    @test_logs (:info,) expand!(iter, fact; verbosity=verbosity)
                    verbosity = EACHITERATION_LEVEL
                else
                    @test_logs expand!(iter, fact; verbosity=verbosity)
                    verbosity = EACHITERATION_LEVEL + 1
                end
            end
            V = stack(unwrapvec, basis(fact))
            H = rayleighquotient(fact)
            factor = (orth == cgs || orth == mgs ? 250 : 10)
            @test normres(fact) < factor * n * eps(real(T))
            @test V' * V ≈ I
            @test A * V ≈ V * H

            states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
            @test rayleighquotient(last(states)) ≈ H

            @constinferred shrink!(fact, n - 1)
            @test_logs (:info,) shrink!(fact, n - 2; verbosity=EACHITERATION_LEVEL + 1)
            @test_logs shrink!(fact, n - 3; verbosity=EACHITERATION_LEVEL)
            @constinferred initialize!(iter, deepcopy(fact))
            @test_logs initialize!(iter, deepcopy(fact); verbosity=EACHITERATION_LEVEL)
            @test_logs (:info,) initialize!(iter, deepcopy(fact);
                                            verbosity=EACHITERATION_LEVEL + 1)
        end
    end
end

# Test incomplete Lanczos factorization
@testset "Incomplete Lanczos factorization ($mode)" for mode in
                                                        (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ?
                  (Float32, Float64, ComplexF32, ComplexF64, Complex{Int}) : (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (cgs2,)

    @testset for T in scalartypes
        @testset for orth in orths # tests fail miserably for cgs and mgs
            if T === Complex{Int}
                A = rand(-100:100, (N, N)) + im * rand(-100:100, (N, N))
                v = rand(-100:100, (N,))
            else
                A = rand(T, (N, N))
                v = rand(T, (N,))
            end
            A = (A + A')
            iter = @constinferred LanczosIterator(wrapop(A, Val(mode)),
                                                  wrapvec(v, Val(mode)),
                                                  orth)
            krylovdim = n
            fact = @constinferred initialize(iter)
            while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
                @constinferred expand!(iter, fact)
                Ṽ, H, r̃, β, e = fact
                V = stack(unwrapvec, Ṽ)
                r = unwrapvec(r̃)
                @test V' * V ≈ I
                @test norm(r) ≈ β
                @test A * V ≈ V * H + r * e'
            end

            fact = @constinferred shrink!(fact, div(n, 2))
            V = stack(unwrapvec, @constinferred basis(fact))
            H = @constinferred rayleighquotient(fact)
            r = @constinferred unwrapvec(residual(fact))
            β = @constinferred normres(fact)
            e = @constinferred rayleighextension(fact)
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ V * H + r * e'
        end
    end
end

# Test incomplete Arnoldi factorization
@testset "Incomplete Arnoldi factorization ($mode)" for mode in
                                                        (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ?
                  (Float32, Float64, ComplexF32, ComplexF64, Complex{Int}) : (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (cgs2,)

    @testset for T in scalartypes
        @testset for orth in orths
            if T === Complex{Int}
                A = rand(-100:100, (N, N)) + im * rand(-100:100, (N, N))
                v = rand(-100:100, (N,))
            else
                A = rand(T, (N, N))
                v = rand(T, (N,))
            end
            iter = @constinferred ArnoldiIterator(wrapop(A, Val(mode)),
                                                  wrapvec(v, Val(mode)), orth)
            krylovdim = 3 * n
            fact = @constinferred initialize(iter)
            while normres(fact) > eps(float(real(T))) && length(fact) < krylovdim
                @constinferred expand!(iter, fact)
                Ṽ, H, r̃, β, e = fact
                V = stack(unwrapvec, Ṽ)
                r = unwrapvec(r̃)
                @test V' * V ≈ I
                @test norm(r) ≈ β
                @test A * V ≈ V * H + r * e'
            end

            fact = @constinferred shrink!(fact, div(n, 2))
            V = stack(unwrapvec, @constinferred basis(fact))
            H = @constinferred rayleighquotient(fact)
            r = unwrapvec(@constinferred residual(fact))
            β = @constinferred normres(fact)
            e = @constinferred rayleighextension(fact)
            @test V' * V ≈ I
            @test norm(r) ≈ β
            @test A * V ≈ V * H + r * e'
        end
    end
end

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
            fact = @constinferred initialize(iter)
            @constinferred expand!(iter, fact)
            @test_logs initialize(iter; verbosity=EACHITERATION_LEVEL)
            @test_logs (:info,) initialize(iter; verbosity=EACHITERATION_LEVEL + 1)
            verbosity = EACHITERATION_LEVEL + 1
            while length(fact) < n
                if verbosity == EACHITERATION_LEVEL + 1
                    @test_logs (:info,) expand!(iter, fact; verbosity=verbosity)
                    verbosity = EACHITERATION_LEVEL
                else
                    @test_logs expand!(iter, fact; verbosity=verbosity)
                    verbosity = EACHITERATION_LEVEL + 1
                end
            end
            U = stack(unwrapvec, basis(fact, Val(:U)))
            V = stack(unwrapvec, basis(fact, Val(:V)))
            B = rayleighquotient(fact)
            @test normres(fact) < 10 * n * eps(real(T))
            @test U' * U ≈ I
            @test V' * V ≈ I
            @test A * V ≈ U * B
            @test A' * U ≈ V * B'

            states = collect(Iterators.take(iter, n)) # collect tests size and eltype?
            @test rayleighquotient(last(states)) ≈ B

            @constinferred shrink!(fact, n - 1)
            @test_logs (:info,) shrink!(fact, n - 2; verbosity=EACHITERATION_LEVEL + 1)
            @test_logs shrink!(fact, n - 3; verbosity=EACHITERATION_LEVEL)
            @constinferred initialize!(iter, deepcopy(fact))
            @test_logs initialize!(iter, deepcopy(fact); verbosity=EACHITERATION_LEVEL)
            @test_logs (:info,) initialize!(iter, deepcopy(fact);
                                            verbosity=EACHITERATION_LEVEL + 1)
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
            U = stack(unwrapvec, @constinferred basis(fact, Val(:U)))
            V = stack(unwrapvec, @constinferred basis(fact, Val(:V)))
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

# TODO: add more tests for block lanczos

