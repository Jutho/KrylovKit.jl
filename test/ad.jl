module LinsolveAD
using KrylovKit, LinearAlgebra
using Random, Test
using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences

fdm = ChainRulesTestUtils._fdm
tolerance(T::Type{<:Number}) = eps(real(T))^(2 / 3)
n = 10
N = 30

function build_mat_example(A, b; tol=tolerance(eltype(A)), kwargs...)
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

function build_fun_example(A, b, c, d, e, f; tol=tolerance(eltype(A)), kwargs...)
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
            return A′ * y + c′ * dot(d′, y)
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

        mat_example, Avec, bvec = build_mat_example(A, b; tol=cond(A) * eps(real(T)),
                                                    krylovdim=n, maxiter=1)

        (JA, Jb) = FiniteDifferences.jacobian(fdm, mat_example, Avec, bvec)
        (JA′, Jb′) = Zygote.jacobian(mat_example, Avec, bvec)
        @test JA ≈ JA′ rtol = cond(A) * tolerance(T)
        @test Jb ≈ Jb′ rtol = cond(A) * tolerance(T)
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

        fun_example, Avec, bvec, cvec, dvec, evec, fvec = build_fun_example(A, b, c, d, e,
                                                                            f;
                                                                            tol=tolerance(T),
                                                                            krylovdim=20)

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

function build_mat_example(A, x, howmany::Int, which, alg)
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
        vals′, vecs′, info′ = eigsolve(A′, x′, howmany, which, alg)
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
            vecs′[i] = vecs′[i] / d
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

function build_fun_example(A, x, c, d, howmany::Int, which, alg)
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

            vals′, vecs′, info′ = eigsolve(x′, howmany′, which, alg) do y
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
                normfix = dot(vecs[i], vecs′[i])
                @assert abs(normfix) > tolerance(eltype(A))
                vecs′[i] = vecs′[i] / normfix
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

@timedtestset "Small eigsolve AD test" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
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
            alg = Arnoldi(; tol=2 * cond(A) * eps(real(T)), krylovdim=n)
            mat_example_ad, mat_example_fd, Avec, xvec, vals, vecs, howmany = build_mat_example(A,
                                                                                                x,
                                                                                                howmany,
                                                                                                which,
                                                                                                alg)

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
                @test all(<(tolerance(T)), abs.(vecs[i]' * ∂vecs[i]))
            end
        end
    end
end
@timedtestset "Large eigsolve AD test" begin
    @testset for T in (ComplexF64,) # (Float64, ComplexF64,)
        # disable real case untill ChainRules.jl/issues/625 is fixed
        if T <: Complex
            whichlist = (:LM, :SI)
        else
            whichlist = (:SR,)
        end
        @testset for which in whichlist
            A = rand(T, (N, N)) .- one(T) / 2
            A = I - (9 // 10) * A / maximum(abs, eigvals(A))
            x = 2 * (rand(T, N) .- one(T) / 2)
            x /= norm(x)
            c = 2 * (rand(T, N) .- one(T) / 2)
            d = 2 * (rand(T, N) .- one(T) / 2)

            howmany = 2
            alg = Arnoldi(; tol=N * N * eps(real(T)), krylovdim=2n)
            fun_example_ad, fun_example_fd, Avec, xvec, cvec, dvec, vals, vecs, howmany = build_fun_example(A,
                                                                                                            x,
                                                                                                            c,
                                                                                                            d,
                                                                                                            howmany,
                                                                                                            which,
                                                                                                            alg)

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
