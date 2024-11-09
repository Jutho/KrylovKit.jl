module DegenerateEigsolveAD

using KrylovKit, LinearAlgebra
using Random, Test, TestExtras
using ChainRulesCore, ChainRulesTestUtils, Zygote, FiniteDifferences
Random.seed!(987654321)

fdm = ChainRulesTestUtils._fdm
n = 10
N = 30

function build_mat_example(A, B, C, x, alg, alg_rrule)
    howmany = 1
    which = :LM

    Avec, A_fromvec = to_vec(A)
    Bvec, B_fromvec = to_vec(B)
    Cvec, C_fromvec = to_vec(C)
    xvec, x_fromvec = to_vec(x)

    M = [zero(A) zero(A) C; A zero(A) zero(A); zero(A) B zero(A)]
    vals, vecs, info = eigsolve(M, x, howmany, which, alg)
    info.converged < howmany && @warn "eigsolve did not converge"

    function mat_example(Av, Bv, Cv, xv)
        Ã = A_fromvec(Av)
        B̃ = B_fromvec(Bv)
        C̃ = C_fromvec(Cv)
        x̃ = x_fromvec(xv)
        M̃ = [zero(Ã) zero(Ã) C̃; Ã zero(Ã) zero(Ã); zero(Ã) B̃ zero(Ã)]
        vals′, vecs′, info′ = eigsolve(M̃, x̃, howmany, which, alg; alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "eigsolve did not converge"
        catresults = vcat(vals′[1:howmany], vecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    function mat_example_fun(Av, Bv, Cv, xv)
        Ã = A_fromvec(Av)
        B̃ = B_fromvec(Bv)
        C̃ = C_fromvec(Cv)
        x̃ = x_fromvec(xv)
        M̃ = [zero(Ã) zero(Ã) C̃; Ã zero(Ã) zero(Ã); zero(Ã) B̃ zero(Ã)]
        f = x -> M̃ * x
        vals′, vecs′, info′ = eigsolve(f, x̃, howmany, which, alg; alg_rrule=alg_rrule)
        info′.converged < howmany && @warn "eigsolve did not converge"
        catresults = vcat(vals′[1:howmany], vecs′[1:howmany]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    function mat_example_fd(Av, Bv, Cv, xv)
        Ã = A_fromvec(Av)
        B̃ = B_fromvec(Bv)
        C̃ = C_fromvec(Cv)
        x̃ = x_fromvec(xv)
        M̃ = [zero(Ã) zero(Ã) C̃; Ã zero(Ã) zero(Ã); zero(Ã) B̃ zero(Ã)]
        howmany′ = (eltype(Av) <: Complex ? 3 : 6) * howmany
        vals′, vecs′, info′ = eigsolve(M̃, x̃, howmany′, which, alg; alg_rrule=alg_rrule)
        _, i = findmin(abs.(vals′ .- vals[1]))
        info′.converged < i && @warn "eigsolve did not converge"
        d = dot(vecs[1], vecs′[i])
        @assert abs(d) > sqrt(eps(real(eltype(A))))
        phasefix = abs(d) / d
        vecs′[i] = vecs′[i] * phasefix
        catresults = vcat(vals′[i:i], vecs′[i:i]...)
        if eltype(catresults) <: Complex
            return vcat(real(catresults), imag(catresults))
        else
            return catresults
        end
    end

    return mat_example, mat_example_fun, mat_example_fd, Avec, Bvec, Cvec, xvec, vals,
           vecs
end

@timedtestset "Degenerate eigsolve AD test with eltype=$T" for T in (Float64, ComplexF64)
    n = 10
    N = 3n

    A = randn(T, n, n)
    B = randn(T, n, n)
    C = randn(T, n, n)

    M = [zeros(T, n, 2n) A; B zeros(T, n, 2n); zeros(T, n, n) C zeros(T, n, n)]
    x = randn(T, N)

    tol = 2 * N^2 * eps(real(T))
    alg = Arnoldi(; tol=tol, krylovdim=2n)
    alg_rrule1 = Arnoldi(; tol=tol, krylovdim=2n, verbosity=-1)
    alg_rrule2 = GMRES(; tol=tol, krylovdim=2n, verbosity=-1)
    mat_example1, mat_example_fun1, mat_example_fd, Avec, Bvec, Cvec, xvec, vals, vecs = build_mat_example(A,
                                                                                                           B,
                                                                                                           C,
                                                                                                           x,
                                                                                                           alg,
                                                                                                           alg_rrule1)
    mat_example2, mat_example_fun2, mat_example_fd, Avec, Bvec, Cvec, xvec, vals, vecs = build_mat_example(A,
                                                                                                           B,
                                                                                                           C,
                                                                                                           x,
                                                                                                           alg,
                                                                                                           alg_rrule2)
    (JA, JB, JC, Jx) = FiniteDifferences.jacobian(fdm, mat_example_fd, Avec, Bvec,
                                                  Cvec, xvec)
    (JA1, JB1, JC1, Jx1) = Zygote.jacobian(mat_example1, Avec, Bvec, Cvec, xvec)
    (JA2, JB2, JC2, Jx2) = Zygote.jacobian(mat_example_fun1, Avec, Bvec, Cvec, xvec)
    (JA3, JB3, JC3, Jx3) = Zygote.jacobian(mat_example2, Avec, Bvec, Cvec, xvec)
    (JA4, JB4, JC4, Jx4) = Zygote.jacobian(mat_example_fun2, Avec, Bvec, Cvec, xvec)

    @test isapprox(JA, JA1; rtol=N * sqrt(eps(real(T))))
    @test isapprox(JB, JB1; rtol=N * sqrt(eps(real(T))))
    @test isapprox(JC, JC1; rtol=N * sqrt(eps(real(T))))

    @test all(isapprox.(JA1, JA2; atol=n * eps(real(T))))
    @test all(isapprox.(JB1, JB2; atol=n * eps(real(T))))
    @test all(isapprox.(JC1, JC2; atol=n * eps(real(T))))

    @test all(isapprox.(JA1, JA3; atol=tol))
    @test all(isapprox.(JB1, JB3; atol=tol))
    @test all(isapprox.(JC1, JC3; atol=tol))

    @test all(isapprox.(JA1, JA4; atol=tol))
    @test all(isapprox.(JB1, JB4; atol=tol))
    @test all(isapprox.(JC1, JC4; atol=tol))

    @test norm(Jx, Inf) < N * sqrt(eps(real(T)))
    @test all(iszero, Jx1)
    @test all(iszero, Jx2)
    @test all(iszero, Jx3)
    @test all(iszero, Jx4)

    # some analysis
    ∂valsA = complex.(JA1[1, :], JA1[N + 2, :])
    ∂valsB = complex.(JB1[1, :], JB1[N + 2, :])
    ∂valsC = complex.(JC1[1, :], JC1[N + 2, :])
    ∂vecsA = complex.(JA1[1 .+ (1:N), :], JA1[N + 2 .+ (1:N), :])
    ∂vecsB = complex.(JB1[1 .+ (1:N), :], JB1[N + 2 .+ (1:N), :])
    ∂vecsC = complex.(JC1[1 .+ (1:N), :], JC1[N + 2 .+ (1:N), :])
    if T <: Complex # test holomorphicity / Cauchy-Riemann equations
        # for eigenvalues
        @test real(∂valsA[1:2:(2n^2)]) ≈ +imag(∂valsA[2:2:(2n^2)])
        @test imag(∂valsA[1:2:(2n^2)]) ≈ -real(∂valsA[2:2:(2n^2)])
        @test real(∂valsB[1:2:(2n^2)]) ≈ +imag(∂valsB[2:2:(2n^2)])
        @test imag(∂valsB[1:2:(2n^2)]) ≈ -real(∂valsB[2:2:(2n^2)])
        @test real(∂valsC[1:2:(2n^2)]) ≈ +imag(∂valsC[2:2:(2n^2)])
        @test imag(∂valsC[1:2:(2n^2)]) ≈ -real(∂valsC[2:2:(2n^2)])
        # and for eigenvectors
        @test real(∂vecsA[:, 1:2:(2n^2)]) ≈ +imag(∂vecsA[:, 2:2:(2n^2)])
        @test imag(∂vecsA[:, 1:2:(2n^2)]) ≈ -real(∂vecsA[:, 2:2:(2n^2)])
        @test real(∂vecsB[:, 1:2:(2n^2)]) ≈ +imag(∂vecsB[:, 2:2:(2n^2)])
        @test imag(∂vecsB[:, 1:2:(2n^2)]) ≈ -real(∂vecsB[:, 2:2:(2n^2)])
        @test real(∂vecsC[:, 1:2:(2n^2)]) ≈ +imag(∂vecsC[:, 2:2:(2n^2)])
        @test imag(∂vecsC[:, 1:2:(2n^2)]) ≈ -real(∂vecsC[:, 2:2:(2n^2)])
    end
    # test orthogonality of vecs and ∂vecs
    @test all(isapprox.(abs.(vecs[1]' * ∂vecsA), 0; atol=sqrt(eps(real(T)))))
    @test all(isapprox.(abs.(vecs[1]' * ∂vecsB), 0; atol=sqrt(eps(real(T)))))
    @test all(isapprox.(abs.(vecs[1]' * ∂vecsC), 0; atol=sqrt(eps(real(T)))))
end

end
