module LinsolveAD
using KrylovKit, LinearAlgebra
using Random, Test
using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences

fdm = ChainRulesTestUtils._fdm
tolerance(T::Type{<:Number}) = eps(real(T))^(2 / 3)
n = 10
N = 30

function build_mat_example(A, b, alg, alg_rrule)
    Avec, A_fromvec = to_vec(A)
    bvec, b_fromvec = to_vec(b)
    T = eltype(A)

    function mat_example(Av, bv)
        A′ = A_fromvec(Av)
        b′ = b_fromvec(bv)
        x, info = linsolve(A′, b′, zero(b′), alg; alg_rrule=alg_rrule)
        info.converged == 0 && @warn "linsolve did not converge"
        xv, = to_vec(x)
        return xv
    end
    return mat_example, Avec, bvec
end

function build_fun_example(A, b, c, d, e, f, alg, alg_rrule)
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

        x, info = linsolve(b′, zero(b′), alg, e′, f′; alg_rrule=alg_rrule) do y
            return A′ * y + c′ * dot(d′, y)
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

    alg = GMRES(; tol=cond(A) * eps(real(T)), krylovdim=n, maxiter=1)
    mat_example, Avec, bvec = build_mat_example(A, b, alg, alg)
    (JA, Jb) = FiniteDifferences.jacobian(fdm, mat_example, Avec, bvec)
    (JA′, Jb′) = Zygote.jacobian(mat_example, Avec, bvec)
    @test JA ≈ JA′ rtol = cond(A) * tolerance(T)
    @test Jb ≈ Jb′ rtol = cond(A) * tolerance(T)
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
    tol = tolerance(T)
    alg1 = GMRES(; tol=tol, krylovdim=20)
    alg2 = BiCGStab(; tol=tol / 10, maxiter=100) # BiCGStab seems to require slightly smaller tolerance for tests to work
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
tolerance(T::Type{<:Number}) = eps(real(T))^(2 / 3)
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

    function mat_example_ad(Av, xv)
        A′ = A_fromvec(Av)
        x′ = x_fromvec(xv)
        vals′, vecs′, info′ = eigsolve(A′, x′, howmany, which, alg; alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "eigsolve did not converge"
        catresults = vcat(vals′[1:howmany], vecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    function mat_example_fd(Av, xv)
        A′ = A_fromvec(Av)
        x′ = x_fromvec(xv)
        vals′, vecs′, info′ = eigsolve(A′, x′, howmany, which, alg)
        info′.converged < howmany && @warn "eigsolve did not converge"
        for i in 1:howmany
            d = dot(vecs[i], vecs′[i])
            @assert abs(d) > tolerance(eltype(A))
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

    return mat_example_ad, mat_example_fd, Avec, xvec, vals, vecs, howmany
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
            A′ = matfromvec(Av)
            x′ = vecfromvec(xv)
            c′ = vecfromvec(cv)
            d′ = vecfromvec(dv)

            vals′, vecs′, info′ = eigsolve(x′, howmany′, which, alg;
                                           alg_rrule=alg_rrule) do y
                return A′ * y + c′ * dot(d′, y)
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
            A′ = matfromvec(Av)
            x′ = vecfromvec(xv)
            c′ = vecfromvec(cv)
            d′ = vecfromvec(dv)

            vals′, vecs′, info′ = eigsolve(x′, howmany′, which, alg) do y
                return A′ * y + c′ * dot(d′, y)
            end
            info′.converged < howmany′ && @warn "eigsolve did not converge"
            for i in 1:howmany′
                d = dot(vecs[i], vecs′[i])
                @assert abs(d) > tolerance(eltype(A))
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

@timedtestset "Small eigsolve AD test for eltype=$T" for T in
                                                         (Float32, Float64, ComplexF32,
                                                          ComplexF64)
    if T <: Complex
        whichlist = (:LM, :SR, :LR, :SI, :LI)
    else
        whichlist = (:LM, :SR, :LR)
    end
    @testset for which in whichlist
        A = 2 * (rand(T, (n, n)) .- one(T) / 2)
        x = 2 * (rand(T, n) .- one(T) / 2)
        x /= norm(x)

        howmany = 3
        tol = tolerance(T)
        alg = Arnoldi(; tol=tol, krylovdim=n)
        alg_rrule1 = alg
        alg_rrule2 = GMRES(; tol=tol, krylovdim=n)
        for alg_rrule in (alg_rrule1, alg_rrule2)
            mat_example_ad, mat_example_fd, Avec, xvec, vals, vecs, howmany = build_mat_example(A,
                                                                                                x,
                                                                                                howmany,
                                                                                                which,
                                                                                                alg,
                                                                                                alg_rrule)

            (JA, Jx) = FiniteDifferences.jacobian(fdm, mat_example_fd, Avec, xvec)
            (JA′, Jx′) = Zygote.jacobian(mat_example_ad, Avec, xvec)

            # finite difference comparison using some kind of tolerance heuristic
            @test JA ≈ JA′ rtol = (T <: Complex ? 4n : n) * cond(A) * tolerance(T)
            @test norm(Jx, Inf) < (T <: Complex ? 4n : n) * cond(A) * tolerance(T)
            @test Jx′ == zero(Jx)

            # some analysis
            ∂vals = complex.(JA′[1:howmany, :], JA′[howmany * (n + 1) .+ (1:howmany), :])
            ∂vecs = map(1:howmany) do i
                return complex.(JA′[(howmany + (i - 1) * n) .+ (1:n), :],
                                JA′[(howmany * (n + 2) + (i - 1) * n) .+ (1:n), :])
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
                @test all(<(n * tolerance(T)), abs.(vecs[i]' * ∂vecs[i]))
            end
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
        alg = Arnoldi(; tol=tolerance(T), krylovdim=2n)
        alg_rrule1 = Arnoldi(; tol=tolerance(T), krylovdim=2n)
        alg_rrule2 = GMRES(; tol=tolerance(T), krylovdim=2n)
        @testset for alg_rrule in (alg_rrule1, alg_rrule2)
            fun_example_ad, fun_example_fd, Avec, xvec, cvec, dvec, vals, vecs, howmany = build_fun_example(A,
                                                                                                            x,
                                                                                                            c,
                                                                                                            d,
                                                                                                            howmany,
                                                                                                            which,
                                                                                                            alg,
                                                                                                            alg_rrule)

            (JA, Jx, Jc, Jd) = FiniteDifferences.jacobian(fdm, fun_example_fd, Avec, xvec,
                                                          cvec, dvec)
            (JA′, Jx′, Jc′, Jd′) = Zygote.jacobian(fun_example_ad, Avec, xvec, cvec, dvec)
            @test JA ≈ JA′
            @test Jc ≈ Jc′
            @test Jd ≈ Jd′
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
tolerance(T::Type{<:Number}) = eps(real(T))^(2 / 3)
n = 10
N = 30

function build_mat_example(A, x, howmany::Int, alg, alg_rrule)
    Avec, A_fromvec = to_vec(A)
    xvec, x_fromvec = to_vec(x)

    vals, lvecs, rvecs, info = svdsolve(A, x, howmany, :LR, alg)
    info.converged < howmany && @warn "svdsolve did not converge"

    function mat_example_ad(Av, xv)
        A′ = A_fromvec(Av)
        x′ = x_fromvec(xv)
        vals′, lvecs′, rvecs′, info′ = svdsolve(A′, x′, howmany, :LR, alg;
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
        A′ = A_fromvec(Av)
        x′ = x_fromvec(xv)
        f = function (x, adj::Val)
            if adj isa Val{true}
                return adjoint(A′) * x
            else
                return A′ * x
            end
        end
        vals′, lvecs′, rvecs′, info′ = svdsolve(f, x′, howmany, :LR, alg;
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
        A′ = A_fromvec(Av)
        x′ = x_fromvec(xv)
        f = function (x)
            return A′ * x
        end
        fᴴ = function (x)
            return adjoint(A′) * x
        end
        vals′, lvecs′, rvecs′, info′ = svdsolve((f, fᴴ), x′, howmany, :LR, alg;
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
        A′ = A_fromvec(Av)
        x′ = x_fromvec(xv)
        vals′, lvecs′, rvecs′, info′ = svdsolve(A′, x′, howmany, :LR, alg)
        info′.converged < howmany && @warn "svdsolve did not converge"
        for i in 1:howmany
            dl = dot(lvecs[i], lvecs′[i])
            dr = dot(rvecs[i], rvecs′[i])
            @assert abs(dl) > tolerance(eltype(A))
            @assert abs(dr) > tolerance(eltype(A))
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

    return mat_example_ad, mat_example_ftuple, mat_example_fval, mat_example_fd, Avec, xvec,
           vals, lvecs, rvecs
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
        A′ = matfromvec(Av)
        x′ = xvecfromvec(xv)
        c′ = cvecfromvec(cv)
        d′ = dvecfromvec(dv)

        f = y -> A′ * y + c′ * dot(d′, y)
        fᴴ = y -> adjoint(A′) * y + d′ * dot(c′, y)
        vals′, lvecs′, rvecs′, info′ = svdsolve((f, fᴴ), x′, howmany, :LR, alg;
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
        A′ = matfromvec(Av)
        x′ = xvecfromvec(xv)
        c′ = cvecfromvec(cv)
        d′ = dvecfromvec(dv)

        f = let A′ = A′, c′ = c′, d′ = d′
            y -> A′ * y + c′ * dot(d′, y)
        end
        fᴴ = let A′ = A′, c′ = c′, d′ = d′
            y -> adjoint(A′) * y + d′ * dot(c′, y)
        end
        vals′, lvecs′, rvecs′, info′ = svdsolve((f, fᴴ), x′, howmany, :LR, alg)
        info′.converged < howmany && @warn "svdsolve did not converge"
        for i in 1:howmany
            dl = dot(lvecs[i], lvecs′[i])
            dr = dot(rvecs[i], rvecs′[i])
            @assert abs(dl) > tolerance(eltype(A))
            @assert abs(dr) > tolerance(eltype(A))
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

    howmany = 3
    tol = tolerance(T)
    alg = GKL(; krylovdim=2n, tol=tol)
    alg_rrule1 = Arnoldi(; tol=tol, krylovdim=(3n + howmany))
    alg_rrule2 = GMRES(; tol=tol, krylovdim=(3n + 1))
    for alg_rrule in (alg_rrule1, alg_rrule2)
        (mat_example_ad, mat_example_ftuple, mat_example_fval, mat_example_fd,
        Avec, xvec, vals, lvecs, rvecs) = build_mat_example(A, x, howmany, alg, alg_rrule)

        (JA, Jx) = FiniteDifferences.jacobian(fdm, mat_example_fd, Avec, xvec)
        (JA1, Jx1) = Zygote.jacobian(mat_example_ad, Avec, xvec)
        (JA2, Jx2) = Zygote.jacobian(mat_example_fval, Avec, xvec)
        (JA3, Jx3) = Zygote.jacobian(mat_example_ftuple, Avec, xvec)

        condA = vals[1] / vals[n]

        # finite difference comparison using some kind of tolerance heuristic
        @test all(isapprox.(JA, JA1;
                            atol=(T <: Complex ? 4 : 2) * n * n * condA * tolerance(T)))
        @test all(isapprox.(JA1, JA2; atol=eps(real(T))))
        @test all(isapprox.(JA1, JA3; atol=eps(real(T))))
        @test norm(Jx, Inf) < (T <: Complex ? 4 : 2) * n * n * condA * tolerance(T)
        @test Jx1 == zero(Jx)

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
            # @show norm(lvecs[i]' * ∂lvecs[i] + rvecs[i]' * ∂rvecs[i])
            @test all(<(tolerance(T)), real.(lvecs[i]' * ∂lvecs[i]))
            @test all(<(tolerance(T)), real.(rvecs[i]' * ∂rvecs[i]))
            @test all(<(tolerance(T)), abs.(lvecs[i]' * ∂lvecs[i] + rvecs[i]' * ∂rvecs[i]))
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
    tol = tolerance(T)
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
        @test norm(Jx, Inf) < (T <: Complex ? 4n : n) * tolerance(T)
    end
end
end
