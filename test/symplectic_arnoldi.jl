@testset "Symplectic Arnoldi" begin
    ε(x, y) = sign(y - x) / 2

    domain = 0:(n - 1)
    w(x) = 1

    skew_dot(u, v) = sum(
        ε(x, y) * w(x) * w(y) * u[x + 1] * v[y + 1]
            for x in domain, y in domain
    )

    function run_symplectic_arnoldi(orth, inplace_init)
        v₀ = SymplecticFormVec(fill(1 / sqrt(sum(w, domain)), n), skew_dot)
        itr = ArnoldiIterator(
            u -> SymplecticFormVec(domain .* u[], u.skewf),
            v₀,
            orth,
        )
        fact = initialize(itr)
        if inplace_init
            fact = initialize!(itr, fact)
        end
        for i in 1:(n - 1)
            expand!(itr, fact)
        end
        return stack(getindex, basis(fact).basis), rayleighquotient(fact), residual(fact), rayleighextension(fact)
    end

    function max_symplectic_error(W)
        max_err = 0.0
        for i in axes(W, 2), j in axes(W, 2)
            val = skew_dot(W[:, i], W[:, j])
            if isodd(i) && j == i + 1
                max_err = max(max_err, abs(val - 1))
            elseif isodd(j) && i == j + 1
                max_err = max(max_err, abs(val + 1))
            else
                max_err = max(max_err, abs(val))
            end
        end
        return max_err
    end

    v₀() = SymplecticFormVec(fill(1 / sqrt(sum(w, domain)), n), skew_dot)
    A(u) = SymplecticFormVec(domain .* u[], u.skewf)

    @testset "shrink!" begin
        itr = ArnoldiIterator(A, v₀(), ModifiedSymplecticGramSchmidt2(ESR2))
        fact = initialize(itr)
        for i in 1:(n - 3)
            expand!(itr, fact)
        end
        for k in (5, 4) # both odd and even lengths
            shrink!(fact, k)
            W = stack(getindex, basis(fact).basis)
            @test max_symplectic_error(W) < 1.0e-8
            @test Diagonal(domain) * W ≈
                W * rayleighquotient(fact) + residual(fact)[] * rayleighextension(fact)' atol = 1.0e-8
        end
    end

    @testset "iteration protocol" begin
        # with the negated operator, `normres` is negative already after `initialize`,
        # which should not terminate the iteration
        Aneg(u) = SymplecticFormVec(-domain .* u[], u.skewf)
        numiter = 0
        fact = nothing
        for f in ArnoldiIterator(Aneg, v₀(), ModifiedSymplecticGramSchmidt(ESR2))
            numiter += 1
            fact = f
            numiter == n - 1 && break
        end
        @test numiter == n - 1
        W = stack(getindex, basis(fact).basis)
        @test max_symplectic_error(W) < 1.0e-8
        @test -Diagonal(domain) * W ≈
            W * rayleighquotient(fact) + residual(fact)[] * rayleighextension(fact)' atol = 1.0e-8
    end

    @testset "skeworthonormalize" begin
        alg = ModifiedSymplecticGramSchmidt(ESR2)
        itr = ArnoldiIterator(A, v₀(), alg)
        fact = initialize(itr)
        for i in 1:4
            expand!(itr, fact)
        end
        u = SymplecticFormVec(cos.(1:n), skew_dot)
        for len in (4, 5) # even length: add odd vector; odd length: add even vector
            b = SymplecticBasis(basis(fact).basis[1:len])
            s, β, x = skeworthonormalize(u, b, Vector{Float64}(undef, len), alg)
            for i in 1:2:(len - 1)
                @test skew_dot(b[i][], s[]) ≈ 0 atol = 1.0e-8
                @test skew_dot(b[i + 1][], s[]) ≈ 0 atol = 1.0e-8
            end
            if iseven(len)
                @test norm(s[]) ≈ 1
            else
                @test skew_dot(b[len][], s[]) ≈ 1
            end
        end
    end

    for esr in (ESR1, ESR2, ESR3m)
        algs = (
            ClassicalSymplecticGramSchmidt(esr),
            ModifiedSymplecticGramSchmidt(esr),
            ClassicalSymplecticGramSchmidt2(esr),
            ModifiedSymplecticGramSchmidt2(esr),
            ClassicalSymplecticGramSchmidtIR(0.75, esr),
            ModifiedSymplecticGramSchmidtIR(0.75, esr),
        )
        @testset "$alg" for alg in algs
            W1, H1, r1, b1 = run_symplectic_arnoldi(alg, false)
            @test max_symplectic_error(W1) < 1.0e-8
            @test Diagonal(domain) * W1 ≈ W1 * H1 + r1[] * b1' atol = 1.0e-8

            W2, H2, r2, b2 = run_symplectic_arnoldi(alg, true)
            @test W1 ≈ W2
            @test H1 ≈ H2
            @test r1[] ≈ r2[] atol = 1.0e-8
            @test max_symplectic_error(W2) < 1.0e-8
            @test Diagonal(domain) * W2 ≈ W2 * H2 + r2[] * b2' atol = 1.0e-8
        end
    end
end
