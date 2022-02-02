module LinsolveAD
    using KrylovKit, LinearAlgebra
    using Random, Test
    using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences

    fdm = ChainRulesTestUtils._fdm
    precision(T::Type{<:Number}) = eps(real(T))^(2/3)
    n = 10
    N = 30

    function build_mat_example(A, b; tol = precision(eltype(A)), kwargs...)
        Avec, A_fromvec = to_vec(A)
        bvec, b_fromvec = to_vec(b)
        T = eltype(A)

        function mat_example(Av, bv)
            A′ = A_fromvec(Av)
            b′ = b_fromvec(bv)
            x, info = linsolve(A′, b′, zero(b′), GMRES(; tol = tol, kwargs...))
            info.converged == 0 && @warn "linsolve did not converge"
            xv, = to_vec(x)
            return xv
        end
        return mat_example, Avec, bvec
    end

    function build_fun_example(A, b, c, d, e, f; tol = precision(eltype(A)), kwargs...)
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

            x, info = linsolve(b′, zero(b′), GMRES(; tol = tol, kwargs...), e′, f′) do y
                A′*y + c′ * dot(d′, y)
            end
            # info.converged > 0 || @warn "not converged"
            xv, = to_vec(x)
            return xv
        end
        return fun_example, Avec, bvec, cvec, dvec, evec, fvec
    end

    @testset "Small linsolve AD test" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            A = 2*(rand(T, (n,n)) .- one(T)/2)
            b = 2*(rand(T, n) .- one(T)/2)
            b /= norm(b)

            mat_example, Avec, bvec = build_mat_example(A, b; tol = cond(A)*eps(real(T)), krylovdim = n, maxiter = 1)

            (JA, Jb) = FiniteDifferences.jacobian(fdm, mat_example, Avec, bvec)
            (JA′, Jb′) = Zygote.jacobian(mat_example, Avec, bvec)
            @test JA ≈ JA′ rtol=cond(A)*precision(T)
            @test Jb ≈ Jb′ rtol=cond(A)*precision(T)
        end
    end

    @testset "Large linsolve AD test" begin
        for T in (Float64, ComplexF64)
            A = rand(T,(N,N)) .- one(T)/2
            A = I-(9//10)*A/maximum(abs, eigvals(A))
            b = 2*(rand(T, N) .- one(T)/2)
            c = 2*(rand(T, N) .- one(T)/2)
            d = 2*(rand(T, N) .- one(T)/2)
            e = rand(T)
            f = rand(T)

            fun_example, Avec, bvec, cvec, dvec, evec, fvec = build_fun_example(A, b, c, d, e, f; tol = precision(T), krylovdim = 20)

            (JA, Jb, Jc, Jd, Je, Jf) = FiniteDifferences.jacobian(fdm, fun_example,
                Avec, bvec, cvec, dvec, evec, fvec)
            (JA′, Jb′, Jc′, Jd′, Je′, Jf′) = Zygote.jacobian(fun_example, Avec, bvec, cvec, dvec, evec, fvec)
            @test JA ≈ JA′
            @test Jb ≈ Jb′
            @test Jc ≈ Jc′
            @test Jd ≈ Jd′
            @test Je ≈ Je′
            @test Jf ≈ Jf′
        end
    end
end
