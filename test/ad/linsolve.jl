module LinsolveAD
using KrylovKit, LinearAlgebra
using Random, Test, TestExtras
using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences
using ..TestSetup

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
        x, info = linsolve(Ã, b̃, x̃, alg; alg_rrule = alg_rrule)
        info.converged == 0 &&
            println("linsolve did not converge: normres = ", info.normres)
        xv, = to_vec(x)
        return xv
    end
    function mat_example_fun(Av, bv, xv)
        Ã = A_fromvec(Av)
        b̃ = b_fromvec(bv)
        x̃ = x_fromvec(xv)
        f = x -> Ã * x
        x, info = linsolve(f, b̃, x̃, alg; alg_rrule = alg_rrule)
        info.converged == 0 &&
            println("linsolve did not converge: normres = ", info.normres)
        xv, = to_vec(x)
        return xv
    end
    return mat_example, mat_example_fun, Avec, bvec, xvec
end

function testfun(A, x, c, d)
    return A * x + c * dot(d, x)
end
testfunthunk(A, x, c, d) = testfun(A, x, c, d)
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(testfunthunk),
        args...
    )
    y = testfunthunk(args...)
    function thunkedpb(dy)
        pb = rrule_via_ad(config, testfun, args...)[2]
        return map(z -> @thunk(z), pb(dy))
    end
    return y, thunkedpb
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

        x, info = linsolve(b̃, zero(b̃), alg, ẽ, f̃; alg_rrule = alg_rrule) do y
            return testfunthunk(Ã, y, c̃, d̃)
        end
        # info.converged > 0 || @warn "not converged"
        xv, = to_vec(x)
        return xv
    end
    return fun_example, Avec, bvec, cvec, dvec, evec, fvec
end

@testset "Small linsolve AD test with eltype=$T" for T in (
        Float32, Float64, ComplexF32,
        ComplexF64,
    )
    A = 2 * (rand(T, (n, n)) .- one(T) / 2)
    b = 2 * (rand(T, n) .- one(T) / 2)
    b /= norm(b)
    x = 2 * (rand(T, n) .- one(T) / 2)

    condA = cond(A)
    tol = tolerance(T) #condA * (T <: Real ? eps(T) : 4 * eps(real(T)))
    alg = GMRES(; tol = tol, krylovdim = n, maxiter = 1)

    config = Zygote.ZygoteRuleConfig()
    _, pb = ChainRulesCore.rrule(config, linsolve, A, b, x, alg, 0, 1; alg_rrule = alg)
    @constinferred pb((ZeroTangent(), NoTangent()))
    @constinferred pb((rand(T, n), NoTangent()))

    mat_example, mat_example_fun, Avec, bvec, xvec = build_mat_example(A, b, x, alg, alg)
    (JA, Jb, Jx) = FiniteDifferences.jacobian(fdm, mat_example, Avec, bvec, xvec)
    (JA1, Jb1, Jx1) = Zygote.jacobian(mat_example, Avec, bvec, xvec)
    (JA2, Jb2, Jx2) = Zygote.jacobian(mat_example_fun, Avec, bvec, xvec)

    @test isapprox(JA, JA1; rtol = condA * sqrt(eps(real(T))))
    @test all(isapprox.(JA1, JA2; atol = n * eps(real(T))))
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
    tol = tolerance(T) # N^2 * eps(real(T))
    alg1 = GMRES(; tol = tol, krylovdim = 20)
    alg2 = BiCGStab(; tol = tol, maxiter = 100) # BiCGStab seems to require slightly smaller tolerance for tests to work
    for (alg, alg_rrule) in ((alg1, alg2), (alg2, alg1))
        #! format: off
        fun_example, Avec, bvec, cvec, dvec, evec, fvec =
            build_fun_example(A, b, c, d, e, f, alg, alg_rrule)
        #! format: on

        (JA, Jb, Jc, Jd, Je, Jf) = FiniteDifferences.jacobian(
            fdm, fun_example,
            Avec, bvec, cvec, dvec, evec,
            fvec
        )
        (JA′, Jb′, Jc′, Jd′, Je′, Jf′) = Zygote.jacobian(
            fun_example, Avec, bvec, cvec,
            dvec, evec, fvec
        )
        rtol = 2 * cond(A + c * d') * sqrt(eps(real(T)))
        @test isapprox(JA, JA′; rtol = rtol)
        @test isapprox(Jb, Jb′; rtol = rtol)
        @test isapprox(Jc, Jc′; rtol = rtol)
        @test isapprox(Jd, Jd′; rtol = rtol)
        @test isapprox(Je, Je′; rtol = rtol)
        @test isapprox(Jf, Jf′; rtol = rtol)
    end
end
end
