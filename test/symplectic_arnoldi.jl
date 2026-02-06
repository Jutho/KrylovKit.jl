@testset "Symplectic Arnoldi" begin
    ε(x, y) = sign(y - x) / 2

    n = 8
    domain = 0:(n - 1)
    w(x) = 1

    skew_dot(u, v) = sum(
        ε(x, y) * w(x) * w(y) * u[x + 1] * v[y + 1]
            for x in domain, y in domain
    )

    function run_symplectic_arnoldi(orth)
        v₀ = InnerProductVec(fill(1 / sqrt(sum(w, domain)), n), skew_dot, norm)
        itr = ArnoldiIterator(
            u -> InnerProductVec(domain .* u[], u.dotf, u.normf),
            v₀,
            orth,
        )
        fact = initialize(itr)
        for _ in 1:(n - 1)
            expand!(itr, fact)
        end
        return stack(getindex, basis(fact).basis)
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

    algs = (
        ClassicalSymplecticGramSchmidt(),
        ModifiedSymplecticGramSchmidt(),
        ClassicalSymplecticGramSchmidt2(),
        ModifiedSymplecticGramSchmidt2(),
        ClassicalSymplecticGramSchmidtIR(0.75),
        ModifiedSymplecticGramSchmidtIR(0.75),
    )

    for alg in algs
        W = run_symplectic_arnoldi(alg)
        @test max_symplectic_error(W) < 1.0e-8
    end
end
