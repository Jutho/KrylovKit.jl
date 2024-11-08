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
    alg_rrule1 = Arnoldi(; tol=tol, krylovdim=4n, verbosity=-1)
    alg_rrule2 = GMRES(; tol=tol, krylovdim=3n, verbosity=-1)
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
            alg_rrule = Arnoldi(; tol=tol, krylovdim=4n, verbosity=-1)
            (vals, lvecs, rvecs, info), pb = ChainRulesCore.rrule(config, svdsolve, A, x,
                                                                  howmany, :LR, alg;
                                                                  alg_rrule=alg_rrule)
            @test_logs pb((ZeroTangent(), im .* lvecs[1:2] .+ lvecs[2:-1:1], ZeroTangent(),
                           NoTangent()))

            alg_rrule = Arnoldi(; tol=tol, krylovdim=4n, verbosity=0)
            (vals, lvecs, rvecs, info), pb = ChainRulesCore.rrule(config, svdsolve, A, x,
                                                                  howmany, :LR, alg;
                                                                  alg_rrule=alg_rrule)
            @test_logs (:warn,) pb((ZeroTangent(),
                                    im .* lvecs[1:2] .+ lvecs[2:-1:1],
                                    ZeroTangent(),
                                    NoTangent()))
            @test_logs (:warn,) pb((ZeroTangent(), lvecs[2:-1:1],
                                    im .* rvecs[1:2] .+ rvecs[2:-1:1],
                                    ZeroTangent(),
                                    NoTangent()))
            @test_logs pb((ZeroTangent(), lvecs[1:2] .+ lvecs[2:-1:1],
                           ZeroTangent(),
                           NoTangent()))
            @test_logs (:warn,) pb((ZeroTangent(),
                                    im .* lvecs[1:2] .+ lvecs[2:-1:1],
                                    +im .* rvecs[1:2] + rvecs[2:-1:1],
                                    NoTangent()))
            @test_logs pb((ZeroTangent(), (1 + im) .* lvecs[1:2] .+ lvecs[2:-1:1],
                           (1 - im) .* rvecs[1:2] + rvecs[2:-1:1],
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

            alg_rrule = GMRES(; tol=tol, krylovdim=3n, verbosity=-1)
            (vals, lvecs, rvecs, info), pb = ChainRulesCore.rrule(config, svdsolve, A, x,
                                                                  howmany, :LR, alg;
                                                                  alg_rrule=alg_rrule)
            @test_logs pb((ZeroTangent(), im .* lvecs[1:2] .+ lvecs[2:-1:1], ZeroTangent(),
                           NoTangent()))

            alg_rrule = GMRES(; tol=tol, krylovdim=3n, verbosity=0)
            (vals, lvecs, rvecs, info), pb = ChainRulesCore.rrule(config, svdsolve, A, x,
                                                                  howmany, :LR, alg;
                                                                  alg_rrule=alg_rrule)
            @test_logs (:warn,) (:warn,) pb((ZeroTangent(),
                                             im .* lvecs[1:2] .+
                                             lvecs[2:-1:1], ZeroTangent(),
                                             NoTangent()))
            @test_logs (:warn,) (:warn,) pb((ZeroTangent(), lvecs[2:-1:1],
                                             im .* rvecs[1:2] .+
                                             rvecs[2:-1:1], ZeroTangent(),
                                             NoTangent()))
            @test_logs pb((ZeroTangent(), lvecs[1:2] .+ lvecs[2:-1:1],
                           ZeroTangent(),
                           NoTangent()))
            @test_logs (:warn,) (:warn,) pb((ZeroTangent(),
                                             im .* lvecs[1:2] .+
                                             lvecs[2:-1:1],
                                             +im .* rvecs[1:2] +
                                             rvecs[2:-1:1],
                                             NoTangent()))
            @test_logs pb((ZeroTangent(),
                           (1 + im) .* lvecs[1:2] .+ lvecs[2:-1:1],
                           (1 - im) .* rvecs[1:2] + rvecs[2:-1:1],
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
    alg_rrule1 = Arnoldi(; tol=tol, krylovdim=2n, verbosity=-1)
    alg_rrule2 = GMRES(; tol=tol, krylovdim=2n, verbosity=-1)
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
