module EigsolveAD
using KrylovKit, LinearAlgebra
using Random, Test, TestExtras
using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences
Random.seed!(987654321)

fdm = ChainRulesTestUtils._fdm
n = 10
N = 30

function build_mat_example(A, x, howmany::Int, which, alg, alg_rrule)
    Avec, A_fromvec = to_vec(A)
    xvec, x_fromvec = to_vec(x)

    vals, vecs, info = eigsolve(A, x, howmany, which, alg)
    info.converged < howmany && @warn "eigsolve did not converge"
    if eltype(A) <: Real && length(vals) > howmany &&
       vals[howmany] == conj(vals[howmany + 1])
        howmany += 1
    end

    function mat_example(Av, xv)
        Ã = A_fromvec(Av)
        x̃ = x_fromvec(xv)
        vals′, vecs′, info′ = eigsolve(Ã, x̃, howmany, which, alg; alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "eigsolve did not converge"
        catresults = vcat(vals′[1:howmany], vecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    function mat_example_fun(Av, xv)
        Ã = A_fromvec(Av)
        x̃ = x_fromvec(xv)
        f = x -> Ã * x
        vals′, vecs′, info′ = eigsolve(f, x̃, howmany, which, alg; alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "eigsolve did not converge"
        catresults = vcat(vals′[1:howmany], vecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    function mat_example_fd(Av, xv)
        Ã = A_fromvec(Av)
        x̃ = x_fromvec(xv)
        vals′, vecs′, info′ = eigsolve(Ã, x̃, howmany, which, alg; alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "eigsolve did not converge"
        for i in 1:howmany
            d = dot(vecs[i], vecs′[i])
            @assert abs(d) > sqrt(eps(real(eltype(A))))
            phasefix = abs(d) / d
            vecs′[i] = vecs′[i] * phasefix
        end
        catresults = vcat(vals′[1:howmany], vecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    return mat_example, mat_example_fun, mat_example_fd, Avec, xvec, vals, vecs, howmany
end

function build_fun_example(A, x, c, d, howmany::Int, which, alg, alg_rrule)
    Avec, matfromvec = to_vec(A)
    xvec, vecfromvec = to_vec(x)
    cvec, = to_vec(c)
    dvec, = to_vec(d)

    vals, vecs, info = eigsolve(x, howmany, which, alg) do y
        return A * y + c * dot(d, y)
    end
    info.converged < howmany && @warn "eigsolve did not converge"
    if eltype(A) <: Real && length(vals) > howmany &&
       vals[howmany] == conj(vals[howmany + 1])
        howmany += 1
    end

    fun_example_ad = let howmany′ = howmany
        function (Av, xv, cv, dv)
            Ã = matfromvec(Av)
            x̃ = vecfromvec(xv)
            c̃ = vecfromvec(cv)
            d̃ = vecfromvec(dv)

            vals′, vecs′, info′ = eigsolve(x̃, howmany′, which, alg;
                                           alg_rrule=alg_rrule) do y
                return Ã * y + c̃ * dot(d̃, y)
            end
            info′.converged < howmany′ && @warn "eigsolve did not converge"
            catresults = vcat(vals′[1:howmany′], vecs′[1:howmany′]...)
            if eltype(catresults) <: Complex
                return vcat(real(catresults), imag(catresults))
            else
                return catresults
            end
        end
    end

    fun_example_fd = let howmany′ = howmany
        function (Av, xv, cv, dv)
            Ã = matfromvec(Av)
            x̃ = vecfromvec(xv)
            c̃ = vecfromvec(cv)
            d̃ = vecfromvec(dv)

            vals′, vecs′, info′ = eigsolve(x̃, howmany′, which, alg;
                                           alg_rrule=alg_rrule) do y
                return Ã * y + c̃ * dot(d̃, y)
            end
            info′.converged < howmany′ && @warn "eigsolve did not converge"
            for i in 1:howmany′
                d = dot(vecs[i], vecs′[i])
                @assert abs(d) > sqrt(eps(real(eltype(A))))
                phasefix = abs(d) / d
                vecs′[i] = vecs′[i] * phasefix
            end
            catresults = vcat(vals′[1:howmany′], vecs′[1:howmany′]...)
            if eltype(catresults) <: Complex
                return vcat(real(catresults), imag(catresults))
            else
                return catresults
            end
        end
    end

    return fun_example_ad, fun_example_fd, Avec, xvec, cvec, dvec, vals, vecs, howmany
end

function build_hermitianfun_example(A, x, c, howmany::Int, which, alg, alg_rrule)
    Avec, matfromvec = to_vec(A)
    xvec, xvecfromvec = to_vec(x)
    cvec, cvecfromvec = to_vec(c)

    vals, vecs, info = eigsolve(x, howmany, which, alg) do y
        return Hermitian(A) * y + c * dot(c, y)
    end
    info.converged < howmany && @warn "eigsolve did not converge"

    function fun_example(Av, xv, cv)
        Ã = matfromvec(Av)
        x̃ = xvecfromvec(xv)
        c̃ = cvecfromvec(cv)

        vals′, vecs′, info′ = eigsolve(x̃, howmany, which, alg;
                                       alg_rrule=alg_rrule) do y
            return Hermitian(Ã) * y + c̃ * dot(c̃, y)
        end
        info′.converged < howmany && @warn "eigsolve did not converge"
        catresults = vcat(vals′[1:howmany], vecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    function fun_example_fd(Av, xv, cv)
        Ã = matfromvec(Av)
        x̃ = xvecfromvec(xv)
        c̃ = cvecfromvec(cv)

        vals′, vecs′, info′ = eigsolve(x̃, howmany, which, alg;
                                       alg_rrule=alg_rrule) do y
            return Hermitian(Ã) * y + c̃ * dot(c̃, y)
        end
        info′.converged < howmany && @warn "eigsolve did not converge"
        for i in 1:howmany
            d = dot(vecs[i], vecs′[i])
            @assert abs(d) > sqrt(eps(real(eltype(A))))
            phasefix = abs(d) / d
            vecs′[i] = vecs′[i] * phasefix
        end
        catresults = vcat(vals′[1:howmany], vecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    return fun_example, fun_example_fd, Avec, xvec, cvec, vals, vecs, howmany
end

@timedtestset "Small eigsolve AD test for eltype=$T" for T in
                                                         (Float32, Float64, ComplexF32,
                                                          ComplexF64)
    if T <: Complex
        whichlist = (:LM, :SR, :LR, :SI, :LI)
    else
        whichlist = (:LM, :SR, :LR)
    end
    A = 2 * (rand(T, (n, n)) .- one(T) / 2)
    x = 2 * (rand(T, n) .- one(T) / 2)
    x /= norm(x)

    howmany = 3
    condA = cond(A)
    tol = n * condA * (T <: Real ? eps(T) : 4 * eps(real(T)))
    alg = Arnoldi(; tol=tol, krylovdim=n)
    alg_rrule1 = Arnoldi(; tol=tol, krylovdim=2n, verbosity=-1)
    alg_rrule2 = GMRES(; tol=tol, krylovdim=n + 1, verbosity=-1)
    config = Zygote.ZygoteRuleConfig()
    @testset for which in whichlist
        for alg_rrule in (alg_rrule1, alg_rrule2)
            # unfortunately, rrule does not seem type stable for function arguments, because the
            # `rrule_via_ad` call does not produce type stable `rrule`s for the function
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          which, alg; alg_rrule=alg_rrule)
            # NOTE: the following is not necessary here, as it is corrected for in the `eigsolve` rrule
            # if length(vals) > howmany && vals[howmany] == conj(vals[howmany + 1])
            #     howmany += 1
            # end
            @constinferred pb((ZeroTangent(), ZeroTangent(), NoTangent()))
            @constinferred pb((randn(T, howmany), ZeroTangent(), NoTangent()))
            @constinferred pb((randn(T, howmany), [randn(T, n)], NoTangent()))
            @constinferred pb((randn(T, howmany), [randn(T, n) for _ in 1:howmany],
                               NoTangent()))
        end

        for alg_rrule in (alg_rrule1, alg_rrule2)
            mat_example, mat_example_fun, mat_example_fd, Avec, xvec, vals, vecs, howmany = build_mat_example(A,
                                                                                                              x,
                                                                                                              howmany,
                                                                                                              which,
                                                                                                              alg,
                                                                                                              alg_rrule)

            (JA, Jx) = FiniteDifferences.jacobian(fdm, mat_example_fd, Avec, xvec)
            (JA1, Jx1) = Zygote.jacobian(mat_example, Avec, xvec)
            (JA2, Jx2) = Zygote.jacobian(mat_example_fun, Avec, xvec)

            # finite difference comparison using some kind of tolerance heuristic
            @test isapprox(JA, JA1; rtol=condA * sqrt(eps(real(T))))
            @test all(isapprox.(JA1, JA2; atol=n * eps(real(T))))
            @test norm(Jx, Inf) < condA * sqrt(eps(real(T)))
            @test all(iszero, Jx1)
            @test all(iszero, Jx2)

            # some analysis
            ∂vals = complex.(JA1[1:howmany, :], JA1[howmany * (n + 1) .+ (1:howmany), :])
            ∂vecs = map(1:howmany) do i
                return complex.(JA1[(howmany + (i - 1) * n) .+ (1:n), :],
                                JA1[(howmany * (n + 2) + (i - 1) * n) .+ (1:n), :])
            end
            if eltype(A) <: Complex # test holomorphicity / Cauchy-Riemann equations
                # for eigenvalues
                @test real(∂vals[:, 1:2:(2n^2)]) ≈ +imag(∂vals[:, 2:2:(2n^2)])
                @test imag(∂vals[:, 1:2:(2n^2)]) ≈ -real(∂vals[:, 2:2:(2n^2)])
                # and for eigenvectors
                for i in 1:howmany
                    @test real(∂vecs[i][:, 1:2:(2n^2)]) ≈ +imag(∂vecs[i][:, 2:2:(2n^2)])
                    @test imag(∂vecs[i][:, 1:2:(2n^2)]) ≈ -real(∂vecs[i][:, 2:2:(2n^2)])
                end
            end
            # test orthogonality of vecs and ∂vecs
            for i in 1:howmany
                @test all(isapprox.(abs.(vecs[i]' * ∂vecs[i]), 0; atol=sqrt(eps(real(T)))))
            end
        end
    end

    if T <: Complex
        @testset "test warnings and info" begin
            alg_rrule = Arnoldi(; tol=tol, krylovdim=n, verbosity=-1)
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          :LR, alg; alg_rrule=alg_rrule)
            @test_logs pb((ZeroTangent(), im .* vecs[1:2] .+ vecs[2:-1:1], NoTangent()))

            alg_rrule = Arnoldi(; tol=tol, krylovdim=n, verbosity=0)
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          :LR, alg; alg_rrule=alg_rrule)
            @test_logs (:warn,) pb((ZeroTangent(), im .* vecs[1:2] .+ vecs[2:-1:1],
                                    NoTangent()))
            pbs = @test_logs pb((ZeroTangent(), vecs[1:2], NoTangent()))
            @test norm(unthunk(pbs[1]), Inf) < condA * sqrt(eps(real(T)))

            alg_rrule = Arnoldi(; tol=tol, krylovdim=n, verbosity=1)
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          :LR, alg; alg_rrule=alg_rrule)
            @test_logs (:warn,) (:info,) pb((ZeroTangent(), im .* vecs[1:2] .+ vecs[2:-1:1],
                                             NoTangent()))
            pbs = @test_logs (:info,) pb((ZeroTangent(), vecs[1:2], NoTangent()))
            @test norm(unthunk(pbs[1]), Inf) < condA * sqrt(eps(real(T)))

            alg_rrule = GMRES(; tol=tol, krylovdim=n, verbosity=-1)
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          :LR, alg; alg_rrule=alg_rrule)
            @test_logs pb((ZeroTangent(), im .* vecs[1:2] .+ vecs[2:-1:1], NoTangent()))

            alg_rrule = GMRES(; tol=tol, krylovdim=n, verbosity=0)
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          :LR, alg; alg_rrule=alg_rrule)
            @test_logs (:warn,) (:warn,) pb((ZeroTangent(),
                                             im .* vecs[1:2] .+
                                             vecs[2:-1:1],
                                             NoTangent()))
            pbs = @test_logs pb((ZeroTangent(), vecs[1:2], NoTangent()))
            @test norm(unthunk(pbs[1]), Inf) < condA * sqrt(eps(real(T)))

            alg_rrule = GMRES(; tol=tol, krylovdim=n, verbosity=1)
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          :LR, alg; alg_rrule=alg_rrule)
            @test_logs (:warn,) (:info,) (:warn,) (:info,) pb((ZeroTangent(),
                                                               im .* vecs[1:2] .+
                                                               vecs[2:-1:1],
                                                               NoTangent()))
            pbs = @test_logs (:info,) (:info,) pb((ZeroTangent(), vecs[1:2], NoTangent()))
            @test norm(unthunk(pbs[1]), Inf) < condA * sqrt(eps(real(T)))
        end
    end
end
@timedtestset "Large eigsolve AD test with eltype=$T" for T in (Float64, ComplexF64)
    if T <: Complex
        whichlist = (:LM, :SI)
    else
        whichlist = (:LM, :SR)
    end
    @testset for which in whichlist
        A = rand(T, (N, N)) .- one(T) / 2
        A = I - (9 // 10) * A / maximum(abs, eigvals(A))
        x = 2 * (rand(T, N) .- one(T) / 2)
        x /= norm(x)
        c = 2 * (rand(T, N) .- one(T) / 2)
        d = 2 * (rand(T, N) .- one(T) / 2)

        howmany = 2
        tol = 2 * N^2 * eps(real(T))
        alg = Arnoldi(; tol=tol, krylovdim=2n)
        alg_rrule1 = Arnoldi(; tol=tol, krylovdim=2n, verbosity=-1)
        alg_rrule2 = GMRES(; tol=tol, krylovdim=2n, verbosity=-1)
        @testset for alg_rrule in (alg_rrule1, alg_rrule2)
            fun_example, fun_example_fd, Avec, xvec, cvec, dvec, vals, vecs, howmany = build_fun_example(A,
                                                                                                         x,
                                                                                                         c,
                                                                                                         d,
                                                                                                         howmany,
                                                                                                         which,
                                                                                                         alg,
                                                                                                         alg_rrule)

            (JA, Jx, Jc, Jd) = FiniteDifferences.jacobian(fdm, fun_example_fd, Avec, xvec,
                                                          cvec, dvec)
            (JA′, Jx′, Jc′, Jd′) = Zygote.jacobian(fun_example, Avec, xvec, cvec, dvec)
            @test JA ≈ JA′
            @test Jc ≈ Jc′
            @test Jd ≈ Jd′
        end
    end
end
@timedtestset "Large Hermitian eigsolve AD test with eltype=$T" for T in
                                                                    (Float64, ComplexF64)
    whichlist = (:LR, :SR)
    @testset for which in whichlist
        A = rand(T, (N, N)) .- one(T) / 2
        A = I - (9 // 10) * A / maximum(abs, eigvals(A))
        x = 2 * (rand(T, N) .- one(T) / 2)
        x /= norm(x)
        c = 2 * (rand(T, N) .- one(T) / 2)

        howmany = 2
        tol = 2 * N^2 * eps(real(T))
        alg = Lanczos(; tol=tol, krylovdim=2n)
        alg_rrule1 = Arnoldi(; tol=tol, krylovdim=2n, verbosity=-1)
        alg_rrule2 = GMRES(; tol=tol, krylovdim=2n, verbosity=-1)
        @testset for alg_rrule in (alg_rrule1, alg_rrule2)
            fun_example, fun_example_fd, Avec, xvec, cvec, vals, vecs, howmany = build_hermitianfun_example(A,
                                                                                                            x,
                                                                                                            c,
                                                                                                            howmany,
                                                                                                            which,
                                                                                                            alg,
                                                                                                            alg_rrule)

            (JA, Jx, Jc) = FiniteDifferences.jacobian(fdm, fun_example_fd, Avec, xvec,
                                                      cvec)
            (JA′, Jx′, Jc′) = Zygote.jacobian(fun_example, Avec, xvec, cvec)
            @test JA ≈ JA′
            @test Jc ≈ Jc′
        end
    end
end

end
