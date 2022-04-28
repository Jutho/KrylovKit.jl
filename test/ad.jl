module LinsolveAD
using KrylovKit, LinearAlgebra
using Random, Test
using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences

fdm = ChainRulesTestUtils._fdm
precision(T::Type{<:Number}) = eps(real(T))^(2 / 3)
n = 10
N = 30

function build_mat_example(A, b; tol=precision(eltype(A)), kwargs...)
    Avec, A_fromvec = to_vec(A)
    bvec, b_fromvec = to_vec(b)
    T = eltype(A)

    function mat_example(Av, bv)
        A′ = A_fromvec(Av)
        b′ = b_fromvec(bv)
        x, info = linsolve(A′, b′, zero(b′), GMRES(; tol=tol, kwargs...))
        info.converged == 0 && @warn "linsolve did not converge"
        xv, = to_vec(x)
        return xv
    end
    return mat_example, Avec, bvec
end

function build_fun_example(A, b, c, d, e, f; tol=precision(eltype(A)), kwargs...)
    Avec, matfromvec = to_vec(A)
    bvec, vecfromvec = to_vec(b)
    cvec, = to_vec(c)
    dvec, = to_vec(d)
    evec, scalarfromvec = to_vec(e)
    fvec, = to_vec(f)

    function fun_example(Av, bv, cv, dv, ev, fv)
        A′ = matfromvec(Av)
        b′ = vecfromvec(bv)
        c′ = vecfromvec(cv)
        d′ = vecfromvec(dv)
        e′ = scalarfromvec(ev)
        f′ = scalarfromvec(fv)

        x, info = linsolve(b′, zero(b′), GMRES(; tol=tol, kwargs...), e′, f′) do y
            A′ * y + c′ * dot(d′, y)
        end
        # info.converged > 0 || @warn "not converged"
        xv, = to_vec(x)
        return xv
    end
    return fun_example, Avec, bvec, cvec, dvec, evec, fvec
end

@testset "Small linsolve AD test" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = 2 * (rand(T, (n, n)) .- one(T) / 2)
        b = 2 * (rand(T, n) .- one(T) / 2)
        b /= norm(b)

        mat_example, Avec, bvec = build_mat_example(
            A, b; tol=cond(A) * eps(real(T)), krylovdim=n, maxiter=1
        )

        (JA, Jb) = FiniteDifferences.jacobian(fdm, mat_example, Avec, bvec)
        (JA′, Jb′) = Zygote.jacobian(mat_example, Avec, bvec)
        @test JA ≈ JA′ rtol = cond(A) * precision(T)
        @test Jb ≈ Jb′ rtol = cond(A) * precision(T)
    end
end

@testset "Large linsolve AD test" begin
    for T in (Float64, ComplexF64)
        A = rand(T, (N, N)) .- one(T) / 2
        A = I - (9 // 10) * A / maximum(abs, eigvals(A))
        b = 2 * (rand(T, N) .- one(T) / 2)
        c = 2 * (rand(T, N) .- one(T) / 2)
        d = 2 * (rand(T, N) .- one(T) / 2)
        e = rand(T)
        f = rand(T)

        fun_example, Avec, bvec, cvec, dvec, evec, fvec = build_fun_example(
            A, b, c, d, e, f; tol=precision(T), krylovdim=20
        )

        (JA, Jb, Jc, Jd, Je, Jf) = FiniteDifferences.jacobian(
            fdm, fun_example, Avec, bvec, cvec, dvec, evec, fvec
        )
        (JA′, Jb′, Jc′, Jd′, Je′, Jf′) = Zygote.jacobian(
            fun_example, Avec, bvec, cvec, dvec, evec, fvec
        )
        @test JA ≈ JA′
        @test Jb ≈ Jb′
        @test Jc ≈ Jc′
        @test Jd ≈ Jd′
        @test Je ≈ Je′
        @test Jf ≈ Jf′
    end
end
end

module EigsolveAD
using KrylovKit, LinearAlgebra
using Random, Test
using ChainRulesCore, ChainRulesTestUtils, Zygote

precision(T::Type{<:Number}) = eps(real(T))^(2 / 3)
n = 2

@testset "Lanczos - eigsolve AD full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, (n, n)) .- one(T) / 2
        A = Hermitian((A + A') / 2)
        v = rand(T, (n,))
        alg = Lanczos(; krylovdim=2 * n, maxiter=1, tol=precision(T))
        which = :LM
        function f(A)
            vals, vecs, info = eigsolve(A, v, n, which, alg)

            vecs_phased = map(vecs) do vec
                return vec ./ exp(angle(vec[1])im)
            end
            D = vcat(vals...)
            U = hcat(vecs_phased...)
            return D, U
        end

        function g(A)
            vals, vecs = eigen(A; sortby=x -> -abs(x))
            vecs_phased = map(1:size(vecs, 2)) do i
                return vecs[:, i] ./ exp(angle(vecs[1, i])im)
            end
            return vals, hcat(vecs_phased...)
        end

        function h(A)
            vals, vecs = eigsolve(v, n, which, alg) do x
                return A * x
            end
            vecs_phased = map(vecs) do vec
                return vec ./ exp(angle(vec[1])im)
            end
            return vcat(vals...), hcat(vecs_phased...)
        end

        y1, back1 = pullback(f, A)
        y2, back2 = pullback(g, A)
        y3, back3 = pullback(h, A)

        for i in 1:2
            @test y1[i] ≈ y2[i]
            @test y2[i] ≈ y3[i]
        end

        for i in 1:3
            Δvals = rand(T, (n,))
            Δvecs = rand(T, (n, n))
            @test first(back1((Δvals, Δvecs))) ≈ first(back2((Δvals, Δvecs)))
            @test first(back2((Δvals, Δvecs))) ≈ first(back3((Δvals, Δvecs)))
        end
    end
end

@testset "Arnoldi - eigsolve AD full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, (n, n)) .- one(T) / 2
        # A = [-1 -4; 1 -1]
        v = rand(T, (n,))

        alg = Arnoldi(; krylovdim=2 * n, maxiter=1, tol=precision(T))
        which = :LM

        function f(A)
            vals, vecs, _ = eigsolve(A, v, n, which, alg)
            vecs_phased = map(vecs) do vec
                return vec ./ exp(angle(vec[1])im)
            end
            D = vcat(vals...)
            U = hcat(vecs_phased...)
            @show D, U
            return D, U
        end

        function g(A)
            vals, vecs = eigen(A; sortby=x -> -abs(x))
            vecs_phased = map(1:size(vecs, 2)) do i
                return vecs[:, i] ./ exp(angle(vecs[1, i])im)
            end
            @show vals, vecs_phased
            return vals, hcat(vecs_phased...)
        end

        function h(A)
            vals, vecs, _ = eigsolve(v, n, which, alg) do x
                return A * x
            end
            vecs_phased = map(vecs) do vec
                return vec ./ exp(angle(vec[1])im)
            end
            return vcat(vals...), hcat(vecs_phased...)
        end

        y1, back1 = pullback(f, A)
        y2, back2 = pullback(g, A)
        y3, back3 = pullback(h, A)

        for i in 1:2
            @test y1[i] ≈ y2[i]
            @test y2[i] ≈ y3[i]
        end

        for i in 1:1
            Δvals = rand(T, (n,))
            Δvecs = rand(T, (n, n))
            @test first(back1((Δvals, Δvecs))) ≈ first(back2((Δvals, Δvecs)))
            @test first(back2((Δvals, Δvecs))) ≈ first(back3((Δvals, Δvecs)))
        end
    end
end
end