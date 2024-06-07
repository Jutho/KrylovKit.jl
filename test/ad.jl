module LinsolveAD
using KrylovKit, LinearAlgebra
using Random, Test, TestExtras
using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences

fdm = ChainRulesTestUtils._fdm
n = 10
N = 30

function build_mat_example(A, b, x, alg, alg_rrule)
    Avec, A_fromvec = to_vec(A)
    bvec, b_fromvec = to_vec(b)
    xvec, x_fromvec = to_vec(x)
    T = eltype(A)

    function mat_example(Av, bv, xv)
        Ã = A_fromvec(Av)
        b̃ = b_fromvec(bv)
        x̃ = x_fromvec(xv)
        x, info = linsolve(Ã, b̃, x̃, alg; alg_rrule=alg_rrule)
        if info.converged == 0
            @warn "linsolve did not converge:"
            println("normres = ", info.normres)
        end
        xv, = to_vec(x)
        return xv
    end
    function mat_example_fun(Av, bv, xv)
        Ã = A_fromvec(Av)
        b̃ = b_fromvec(bv)
        x̃ = x_fromvec(xv)
        f = x -> Ã * x
        x, info = linsolve(f, b̃, x̃, alg; alg_rrule=alg_rrule)
        if info.converged == 0
            @warn "linsolve did not converge:"
            println("normres = ", info.normres)
        end
        xv, = to_vec(x)
        return xv
    end
    return mat_example, mat_example_fun, Avec, bvec, xvec
end

function build_fun_example(A, b, c, d, e, f, alg, alg_rrule)
    Avec, matfromvec = to_vec(A)
    bvec, vecfromvec = to_vec(b)
    cvec, = to_vec(c)
    dvec, = to_vec(d)
    evec, scalarfromvec = to_vec(e)
    fvec, = to_vec(f)

    function fun_example(Av, bv, cv, dv, ev, fv)
        Ã = matfromvec(Av)
        b̃ = vecfromvec(bv)
        c̃ = vecfromvec(cv)
        d̃ = vecfromvec(dv)
        ẽ = scalarfromvec(ev)
        f̃ = scalarfromvec(fv)

        x, info = linsolve(b̃, zero(b̃), alg, ẽ, f̃; alg_rrule=alg_rrule) do y
            return Ã * y + c̃ * dot(d̃, y)
        end
        # info.converged > 0 || @warn "not converged"
        xv, = to_vec(x)
        return xv
    end
    return fun_example, Avec, bvec, cvec, dvec, evec, fvec
end

@testset "Small linsolve AD test with eltype=$T" for T in (Float32, Float64, ComplexF32,
                                                           ComplexF64)
    A = 2 * (rand(T, (n, n)) .- one(T) / 2)
    b = 2 * (rand(T, n) .- one(T) / 2)
    b /= norm(b)
    x = 2 * (rand(T, n) .- one(T) / 2)

    condA = cond(A)
    tol = condA * (T <: Real ? eps(T) : 4 * eps(real(T)))
    alg = GMRES(; tol=tol, krylovdim=n, maxiter=1)

    config = Zygote.ZygoteRuleConfig()
    _, pb = ChainRulesCore.rrule(config, linsolve, A, b, x, alg, 0, 1; alg_rrule=alg)
    @constinferred pb((ZeroTangent(), NoTangent()))
    @constinferred pb((rand(T, n), NoTangent()))

    mat_example, mat_example_fun, Avec, bvec, xvec = build_mat_example(A, b, x, alg, alg)
    (JA, Jb, Jx) = FiniteDifferences.jacobian(fdm, mat_example, Avec, bvec, xvec)
    (JA1, Jb1, Jx1) = Zygote.jacobian(mat_example, Avec, bvec, xvec)
    (JA2, Jb2, Jx2) = Zygote.jacobian(mat_example_fun, Avec, bvec, xvec)

    @test isapprox(JA, JA1; rtol=condA * sqrt(eps(real(T))))
    @test all(isapprox.(JA1, JA2; atol=n * eps(real(T))))
    # factor 2 is minimally necessary for complex case, but 3 is more robust
    @test norm(Jx, Inf) < condA * sqrt(eps(real(T)))
    @test all(iszero, Jx1)
end

@testset "Large linsolve AD test with eltype=$T" for T in (Float64, ComplexF64)
    A = rand(T, (N, N)) .- one(T) / 2
    A = I - (9 // 10) * A / maximum(abs, eigvals(A))
    b = 2 * (rand(T, N) .- one(T) / 2)
    c = 2 * (rand(T, N) .- one(T) / 2)
    d = 2 * (rand(T, N) .- one(T) / 2)
    e = rand(T)
    f = rand(T)

    # mix algorithms]
    tol = N^2 * eps(real(T))
    alg1 = GMRES(; tol=tol, krylovdim=20)
    alg2 = BiCGStab(; tol=tol, maxiter=100) # BiCGStab seems to require slightly smaller tolerance for tests to work
    for (alg, alg_rrule) in ((alg1, alg2), (alg2, alg1))
        fun_example, Avec, bvec, cvec, dvec, evec, fvec = build_fun_example(A, b, c, d, e,
                                                                            f, alg,
                                                                            alg_rrule)

        (JA, Jb, Jc, Jd, Je, Jf) = FiniteDifferences.jacobian(fdm, fun_example,
                                                              Avec, bvec, cvec, dvec, evec,
                                                              fvec)
        (JA′, Jb′, Jc′, Jd′, Je′, Jf′) = Zygote.jacobian(fun_example, Avec, bvec, cvec,
                                                         dvec, evec, fvec)
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
using Random, Test, TestExtras
using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences
Random.seed!(123456789)

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
    alg_rrule1 = Arnoldi(; tol=tol, krylovdim=2n)
    alg_rrule2 = GMRES(; tol=tol, krylovdim=n)
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
            alg_rrule = Arnoldi(; tol=tol, krylovdim=n, verbosity=0)
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          :LR, alg; alg_rrule=alg_rrule)
            @test_logs pb((ZeroTangent(), im .* vecs[1:2] .+ vecs[2:-1:1], NoTangent()))

            alg_rrule = Arnoldi(; tol=tol, krylovdim=n, verbosity=1)
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          :LR, alg; alg_rrule=alg_rrule)
            @test_logs (:warn,) (:info,) pb((ZeroTangent(), im .* vecs[1:2] .+ vecs[2:-1:1],
                                             NoTangent()))
            pbs = @test_logs (:info,) pb((ZeroTangent(), vecs[1:2], NoTangent()))
            @test norm(unthunk(pbs[1]), Inf) < condA * sqrt(eps(real(T)))

            alg_rrule = GMRES(; tol=tol, krylovdim=n)
            (vals, vecs, info), pb = ChainRulesCore.rrule(config, eigsolve, A, x, howmany,
                                                          :LR, alg; alg_rrule=alg_rrule)
            @test_logs pb((ZeroTangent(), im .* vecs[1:2] .+ vecs[2:-1:1], NoTangent()))

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
        alg_rrule1 = Arnoldi(; tol=tol, krylovdim=2n)
        alg_rrule2 = GMRES(; tol=tol, krylovdim=2n)
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
        alg_rrule1 = Arnoldi(; tol=tol, krylovdim=2n)
        alg_rrule2 = GMRES(; tol=tol, krylovdim=2n)
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

module SvdsolveAD
using KrylovKit, LinearAlgebra
using Random, Test, TestExtras
using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences
Random.seed!(123456789)

fdm = ChainRulesTestUtils._fdm
n = 10
N = 30

function build_mat_example(A, x, howmany::Int, alg, alg_rrule)
    Avec, A_fromvec = to_vec(A)
    xvec, x_fromvec = to_vec(x)

    vals, lvecs, rvecs, info = svdsolve(A, x, howmany, :LR, alg)
    info.converged < howmany && @warn "svdsolve did not converge"

    function mat_example_mat(Av, xv)
        Ã = A_fromvec(Av)
        x̃ = x_fromvec(xv)
        vals′, lvecs′, rvecs′, info′ = svdsolve(Ã, x̃, howmany, :LR, alg;
                                                alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "svdsolve did not converge"
        catresults = vcat(vals′[1:howmany], lvecs′[1:howmany]..., rvecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end
    function mat_example_fval(Av, xv)
        Ã = A_fromvec(Av)
        x̃ = x_fromvec(xv)
        f = (x, adj::Val) -> (adj isa Val{true}) ? adjoint(Ã) * x : Ã * x
        vals′, lvecs′, rvecs′, info′ = svdsolve(f, x̃, howmany, :LR, alg;
                                                alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "svdsolve did not converge"
        catresults = vcat(vals′[1:howmany], lvecs′[1:howmany]..., rvecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end
    function mat_example_ftuple(Av, xv)
        Ã = A_fromvec(Av)
        x̃ = x_fromvec(xv)
        (f, fᴴ) = (x -> Ã * x, x -> adjoint(Ã) * x)
        vals′, lvecs′, rvecs′, info′ = svdsolve((f, fᴴ), x̃, howmany, :LR, alg;
                                                alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "svdsolve did not converge"
        catresults = vcat(vals′[1:howmany], lvecs′[1:howmany]..., rvecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    function mat_example_fd(Av, xv)
        Ã = A_fromvec(Av)
        x̃ = x_fromvec(xv)
        vals′, lvecs′, rvecs′, info′ = svdsolve(Ã, x̃, howmany, :LR, alg;
                                                alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "svdsolve did not converge"
        for i in 1:howmany
            dl = dot(lvecs[i], lvecs′[i])
            dr = dot(rvecs[i], rvecs′[i])
            @assert abs(dl) > sqrt(eps(real(eltype(A))))
            @assert abs(dr) > sqrt(eps(real(eltype(A))))
            phasefix = sqrt(abs(dl * dr) / (dl * dr))
            lvecs′[i] = lvecs′[i] * phasefix
            rvecs′[i] = rvecs′[i] * phasefix
        end
        catresults = vcat(vals′[1:howmany], lvecs′[1:howmany]..., rvecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    return mat_example_mat, mat_example_ftuple, mat_example_fval, mat_example_fd, Avec,
           xvec, vals, lvecs, rvecs
end

function build_fun_example(A, x, c, d, howmany::Int, alg, alg_rrule)
    Avec, matfromvec = to_vec(A)
    xvec, xvecfromvec = to_vec(x)
    cvec, cvecfromvec = to_vec(c)
    dvec, dvecfromvec = to_vec(d)

    f = y -> A * y + c * dot(d, y)
    fᴴ = y -> adjoint(A) * y + d * dot(c, y)
    vals, lvecs, rvecs, info = svdsolve((f, fᴴ), x, howmany, :LR, alg)
    info.converged < howmany && @warn "svdsolve did not converge"

    function fun_example_ad(Av, xv, cv, dv)
        Ã = matfromvec(Av)
        x̃ = xvecfromvec(xv)
        c̃ = cvecfromvec(cv)
        d̃ = dvecfromvec(dv)

        f = y -> Ã * y + c̃ * dot(d̃, y)
        fᴴ = y -> adjoint(Ã) * y + d̃ * dot(c̃, y)
        vals′, lvecs′, rvecs′, info′ = svdsolve((f, fᴴ), x̃, howmany, :LR, alg;
                                                alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "svdsolve did not converge"
        catresults = vcat(vals′[1:howmany], lvecs′[1:howmany]..., rvecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end
    function fun_example_fd(Av, xv, cv, dv)
        Ã = matfromvec(Av)
        x̃ = xvecfromvec(xv)
        c̃ = cvecfromvec(cv)
        d̃ = dvecfromvec(dv)

        f = y -> Ã * y + c̃ * dot(d̃, y)
        fᴴ = y -> adjoint(Ã) * y + d̃ * dot(c̃, y)
        vals′, lvecs′, rvecs′, info′ = svdsolve((f, fᴴ), x̃, howmany, :LR, alg;
                                                alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "svdsolve did not converge"
        for i in 1:howmany
            dl = dot(lvecs[i], lvecs′[i])
            dr = dot(rvecs[i], rvecs′[i])
            @assert abs(dl) > sqrt(eps(real(eltype(A))))
            @assert abs(dr) > sqrt(eps(real(eltype(A))))
            phasefix = sqrt(abs(dl * dr) / (dl * dr))
            lvecs′[i] = lvecs′[i] * phasefix
            rvecs′[i] = rvecs′[i] * phasefix
        end
        catresults = vcat(vals′[1:howmany], lvecs′[1:howmany]..., rvecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end
    return fun_example_ad, fun_example_fd, Avec, xvec, cvec, dvec, vals, lvecs, rvecs
end

@timedtestset "Small svdsolve AD test with eltype=$T" for T in
                                                          (Float32, Float64, ComplexF32,
                                                           ComplexF64)
    A = 2 * (rand(T, (n, 2 * n)) .- one(T) / 2)
    x = 2 * (rand(T, n) .- one(T) / 2)
    x /= norm(x)
    condA = cond(A)

    howmany = 3
    tol = 3 * n * condA * (T <: Real ? eps(T) : 4 * eps(real(T)))
    alg = GKL(; krylovdim=2n, tol=tol)
    alg_rrule1 = Arnoldi(; tol=tol, krylovdim=4n)
    alg_rrule2 = GMRES(; tol=tol, krylovdim=3n)
    config = Zygote.ZygoteRuleConfig()
    for alg_rrule in (alg_rrule1, alg_rrule2)
        # unfortunately, rrule does not seem type stable for function arguments, because the
        # `rrule_via_ad` call does not produce type stable `rrule`s for the function
        _, pb = ChainRulesCore.rrule(config, svdsolve, A, x, howmany, :LR, alg;
                                     alg_rrule=alg_rrule)
        @constinferred pb((ZeroTangent(), ZeroTangent(), ZeroTangent(), NoTangent()))
        @constinferred pb((randn(real(T), howmany), ZeroTangent(), ZeroTangent(),
                           NoTangent()))
        @constinferred pb((randn(real(T), howmany), [randn(T, n)], ZeroTangent(),
                           NoTangent()))
        @constinferred pb((randn(real(T), howmany), [randn(T, n) for _ in 1:howmany],
                           [randn(T, 2 * n) for _ in 1:howmany], NoTangent()))
    end
    for alg_rrule in (alg_rrule1, alg_rrule2)
        (mat_example_mat, mat_example_ftuple, mat_example_fval, mat_example_fd,
        Avec, xvec, vals, lvecs, rvecs) = build_mat_example(A, x, howmany, alg, alg_rrule)

        (JA, Jx) = FiniteDifferences.jacobian(fdm, mat_example_fd, Avec, xvec)
        (JA1, Jx1) = Zygote.jacobian(mat_example_mat, Avec, xvec)
        (JA2, Jx2) = Zygote.jacobian(mat_example_fval, Avec, xvec)
        (JA3, Jx3) = Zygote.jacobian(mat_example_ftuple, Avec, xvec)

        # finite difference comparison using some kind of tolerance heuristic
        @test isapprox(JA, JA1; rtol=3 * n * n * condA * sqrt(eps(real(T))))
        @test all(isapprox.(JA1, JA2; atol=n * eps(real(T))))
        @test all(isapprox.(JA1, JA3; atol=n * eps(real(T))))
        @test norm(Jx, Inf) < 5 * condA * sqrt(eps(real(T)))
        @test all(iszero, Jx1)
        @test all(iszero, Jx2)
        @test all(iszero, Jx3)

        # some analysis
        if eltype(A) <: Complex # test holomorphicity / Cauchy-Riemann equations
            ∂vals = complex.(JA1[1:howmany, :],
                             JA1[howmany * (3 * n + 1) .+ (1:howmany), :])
            ∂lvecs = map(1:howmany) do i
                return complex.(JA1[(howmany + (i - 1) * n) .+ (1:n), :],
                                JA1[(howmany * (3 * n + 2) + (i - 1) * n) .+ (1:n), :])
            end
            ∂rvecs = map(1:howmany) do i
                return complex.(JA1[(howmany * (n + 1) + (i - 1) * (2 * n)) .+ (1:(2n)), :],
                                JA1[(howmany * (4 * n + 2) + (i - 1) * 2n) .+ (1:(2n)), :])
            end
        else
            ∂vals = JA1[1:howmany, :]
            ∂lvecs = map(1:howmany) do i
                return JA1[(howmany + (i - 1) * n) .+ (1:n), :]
            end
            ∂rvecs = map(1:howmany) do i
                return JA1[(howmany * (n + 1) + (i - 1) * (2 * n)) .+ (1:(2n)), :]
            end
        end
        # test orthogonality of vecs and ∂vecs
        for i in 1:howmany
            prec = 4 * cond(A) * sqrt(eps(real(T)))
            @test all(<(prec), real.(lvecs[i]' * ∂lvecs[i]))
            @test all(<(prec), real.(rvecs[i]' * ∂rvecs[i]))
            @test all(<(prec), abs.(lvecs[i]' * ∂lvecs[i] + rvecs[i]' * ∂rvecs[i]))
        end
    end
    if T <: Complex
        @testset "test warnings and info" begin
            alg_rrule = Arnoldi(; tol=tol, krylovdim=4n, verbosity=0)
            (vals, lvecs, rvecs, info), pb = ChainRulesCore.rrule(config, svdsolve, A, x,
                                                                  howmany, :LR, alg;
                                                                  alg_rrule=alg_rrule)
            @test_logs pb((ZeroTangent(), im .* lvecs[1:2] .+ lvecs[2:-1:1], ZeroTangent(),
                           NoTangent()))

            alg_rrule = Arnoldi(; tol=tol, krylovdim=4n, verbosity=1)
            (vals, lvecs, rvecs, info), pb = ChainRulesCore.rrule(config, svdsolve, A, x,
                                                                  howmany, :LR, alg;
                                                                  alg_rrule=alg_rrule)
            @test_logs (:warn,) (:info,) pb((ZeroTangent(),
                                             im .* lvecs[1:2] .+ lvecs[2:-1:1],
                                             ZeroTangent(),
                                             NoTangent()))
            @test_logs (:warn,) (:info,) pb((ZeroTangent(), lvecs[2:-1:1],
                                             im .* rvecs[1:2] .+ rvecs[2:-1:1],
                                             ZeroTangent(),
                                             NoTangent()))
            @test_logs (:info,) pb((ZeroTangent(), lvecs[1:2] .+ lvecs[2:-1:1],
                                    ZeroTangent(),
                                    NoTangent()))
            @test_logs (:warn,) (:info,) pb((ZeroTangent(),
                                             im .* lvecs[1:2] .+ lvecs[2:-1:1],
                                             +im .* rvecs[1:2] + rvecs[2:-1:1],
                                             NoTangent()))
            @test_logs (:info,) pb((ZeroTangent(), (1 + im) .* lvecs[1:2] .+ lvecs[2:-1:1],
                                    (1 - im) .* rvecs[1:2] + rvecs[2:-1:1],
                                    NoTangent()))

            alg_rrule = GMRES(; tol=tol, krylovdim=3n, verbosity=0)
            (vals, lvecs, rvecs, info), pb = ChainRulesCore.rrule(config, svdsolve, A, x,
                                                                  howmany, :LR, alg;
                                                                  alg_rrule=alg_rrule)
            @test_logs pb((ZeroTangent(), im .* lvecs[1:2] .+ lvecs[2:-1:1], ZeroTangent(),
                           NoTangent()))

            alg_rrule = GMRES(; tol=tol, krylovdim=3n, verbosity=1)
            (vals, lvecs, rvecs, info), pb = ChainRulesCore.rrule(config, svdsolve, A, x,
                                                                  howmany, :LR, alg;
                                                                  alg_rrule=alg_rrule)
            @test_logs (:warn,) (:info,) (:warn,) (:info,) pb((ZeroTangent(),
                                                               im .* lvecs[1:2] .+
                                                               lvecs[2:-1:1], ZeroTangent(),
                                                               NoTangent()))
            @test_logs (:warn,) (:info,) (:warn,) (:info,) pb((ZeroTangent(), lvecs[2:-1:1],
                                                               im .* rvecs[1:2] .+
                                                               rvecs[2:-1:1], ZeroTangent(),
                                                               NoTangent()))
            @test_logs (:info,) (:info,) pb((ZeroTangent(), lvecs[1:2] .+ lvecs[2:-1:1],
                                             ZeroTangent(),
                                             NoTangent()))
            @test_logs (:warn,) (:info,) (:warn,) (:info,) pb((ZeroTangent(),
                                                               im .* lvecs[1:2] .+
                                                               lvecs[2:-1:1],
                                                               +im .* rvecs[1:2] +
                                                               rvecs[2:-1:1],
                                                               NoTangent()))
            @test_logs (:info,) (:info,) pb((ZeroTangent(),
                                             (1 + im) .* lvecs[1:2] .+ lvecs[2:-1:1],
                                             (1 - im) .* rvecs[1:2] + rvecs[2:-1:1],
                                             NoTangent()))
        end
    end
end
@timedtestset "Large svdsolve AD test with eltype=$T" for T in (Float64, ComplexF64)
    which = :LR
    A = rand(T, (N, N + n)) .- one(T) / 2
    A = I[1:N, 1:(N + n)] - (9 // 10) * A / maximum(svdvals(A))
    x = 2 * (rand(T, N) .- one(T) / 2)
    x /= norm(x)
    c = 2 * (rand(T, N) .- one(T) / 2)
    d = 2 * (rand(T, N + n) .- one(T) / 2)

    howmany = 2
    tol = 2 * N^2 * eps(real(T))
    alg = GKL(; tol=tol, krylovdim=2n)
    alg_rrule1 = Arnoldi(; tol=tol, krylovdim=2n)
    alg_rrule2 = GMRES(; tol=tol, krylovdim=2n)
    for alg_rrule in (alg_rrule1, alg_rrule2)
        fun_example_ad, fun_example_fd, Avec, xvec, cvec, dvec, vals, lvecs, rvecs = build_fun_example(A,
                                                                                                       x,
                                                                                                       c,
                                                                                                       d,
                                                                                                       howmany,
                                                                                                       alg,
                                                                                                       alg_rrule)

        (JA, Jx, Jc, Jd) = FiniteDifferences.jacobian(fdm, fun_example_fd, Avec, xvec,
                                                      cvec, dvec)
        (JA′, Jx′, Jc′, Jd′) = Zygote.jacobian(fun_example_ad, Avec, xvec, cvec, dvec)
        @test JA ≈ JA′
        @test Jc ≈ Jc′
        @test Jd ≈ Jd′
        @test norm(Jx, Inf) < (T <: Complex ? 4n : n) * sqrt(eps(real(T)))
    end
end
end
