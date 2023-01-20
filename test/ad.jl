using LinearAlgebra: eigen
using ChainRulesTestUtils: _fdm
using Zygote, FiniteDifferences
using VectorInterface

@testset "Small linsolve AD test" begin
    @testset for T in (Float64, ComplexF64)
        A = 2 * (rand(T, (n, n)) .- one(T) / 2)
        b = 2 * (rand(T, n) .- one(T) / 2)
        b /= norm(b)
        alg = GMRES(; tol=cond(A) * eps(real(T)), krylovdim=n, maxiter=1)

        Av, matfromvec = to_vec(A)
        bv, vecfromvec = to_vec(b)

        function f(Av, bv)
            A′ = wrapop(matfromvec(Av))
            b′ = wrapvec(vecfromvec(bv))
            x′, info = linsolve(A′, b′, zerovector(b′), alg)
            return first(to_vec(unwrapvec(x′)))
        end

        JA, Jb = Zygote.jacobian(f, Av, bv)
        JA′, Jb′ = FiniteDifferences.jacobian(_fdm, f, Av, bv)

        @test JA ≈ JA′ rtol = cond(A) * precision(T)
        @test Jb ≈ Jb′ rtol = cond(A) * precision(T)
    end
end

@testset "Large linsolve AD test" begin
    for T in (Float64, ComplexF64)
        A = rand(T, (N, N)) .- one(T) / 2
        A = I - (9 // 10) * A / maximum(abs, eigvals(A))
        b = 2 * (rand(T, N) .- one(T) / 2)
        e = rand(T)
        f = rand(T)
        alg = GMRES(; tol=precision(T), krylovdim=20)

        Av, matfromvec = to_vec(A)
        bv, vecfromvec = to_vec(b)
        ev, scalfromvec = to_vec(e)
        fv, scalfromvec2 = to_vec(f)

        function f(Av, bv, ev, fv)
            A′ = wrapop(matfromvec(Av))
            b′ = wrapvec(vecfromvec(bv))
            e′ = scalfromvec(ev)
            f′ = scalfromvec2(fv)
            x′, info = linsolve(A′, b′, zerovector(b′), alg, e′, f′)
            return first(to_vec(unwrapvec(x′)))
        end

        JA, Jb, Je, Jf = Zygote.jacobian(f, Av, bv, ev, fv)
        JA′, Jb′, Je′, Jf′ = FiniteDifferences.jacobian(_fdm, f, Av, bv, ev, fv)

        @test JA ≈ JA′ rtol = cond(A) * precision(T) * length(A)
        @test Jb ≈ Jb′ rtol = cond(A) * precision(T)
        @test Je ≈ Je′ rtol = cond(A) * precision(T)
        @test Jf ≈ Jf′ rtol = cond(A) * precision(T)
    end
end

@testset "Small eigsolve AD test" begin
    for T in (Float64, ComplexF64)
        A = rand(T, (n, n)) .- one(T) / 2
        A /= norm(A)
        v = rand(T, n)
        alg = Arnoldi(; krylovdim=2 * n, maxiter=1, tol=eps(real((T)))^(3 / 4))

        Av, matfromvec = to_vec(A)
        v′ = wrapvec(v)

        function L1(Av)
            A = wrapop(matfromvec(Av))
            D, V, info = eigsolve(A, v′, 1, :LM, alg)
            D1 = first(D)
            V1 = first(V)
            return [abs(inner(V1, v′)), real(D1), imag(D1)]
        end

        function L2(Av)
            A = matfromvec(Av)
            vals, vecs = eigen(A; sortby=x -> -abs(x))
            D1 = first(vals)
            V1 = vecs[:, 1]
            return [abs(inner(V1, v)), real(D1), imag(D1)]
        end

        @test L1(Av) ≈ L2(Av)
        JA = Zygote.jacobian(L1, Av)
        JA′ = Zygote.jacobian(L2, Av)
        @test JA[1] ≈ JA′[1] atol = cond(A) * precision(T) * length(A)
    end
end

@testset "Large eigsolve AD test" begin
    for T in (Float64, ComplexF64)
        A = rand(T, (N, N)) .- one(T) / 2
        A /= norm(A)
        v = rand(T, N)
        alg = Arnoldi(; tol=precision(T), krylovdim=3 * n, maxiter=20)

        Av, matfromvec = to_vec(A)
        v′ = wrapvec(v)

        function L1(Av)
            A = wrapop(matfromvec(Av))
            D, V, info = eigsolve(A, v′, n, :LM, alg)
            overlaps = [abs(inner(V[i], v′)) for i in 1:n]
            return vcat(overlaps, real.(D[1:n]), imag.(D[1:n]))
        end

        function L2(Av)
            A = matfromvec(Av)
            vals, vecs = eigen(A; sortby=x -> -abs(x))
            overlaps = [abs(inner(vecs[:, i], v)) for i in 1:n]
            return vcat(overlaps, real.(vals[1:n]), imag.(vals[1:n]))
        end

        @test L1(Av) ≈ L2(Av)
        JA = Zygote.jacobian(L1, Av)
        JA′ = Zygote.jacobian(L2, Av)
        @test JA[1] ≈ JA′[1] atol = cond(A) * precision(T) * length(A)
    end
end