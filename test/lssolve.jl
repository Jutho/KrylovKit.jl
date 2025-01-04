# Test LSMR complete
@testset "LSMR small problem ($mode)" for mode in (:vector, :inplace, :outplace, :mixed)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (2 * n, n))
        U, S, V = svd(A)
        invS = 1 ./ S
        S[end] = 0 # make rank deficient
        invS[end] = 0 # choose minimal norm solution
        A = U * Diagonal(S) * V'

        b = rand(T, 2 * n)
        tol = tol = 10 * n * eps(real(T))
        alg = LSMR(; maxiter=3, tol=tol, verbosity=2)
        x, info = @constinferred lssolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), alg)
        r = b - A * unwrapvec(x)
        @test unwrapvec(info.residual) ≈ r
        @test info.normres ≈ norm(A' * r)

        alg = LSMR(; maxiter=2 * n, tol=tol, verbosity=1)
        # maxiter = 2 * n because of loss of orthogonality for single precision
        x, info = @constinferred lssolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), alg)

        @test info.converged > 0
        @test abs(inner(V[:, end], unwrapvec(x))) < alg.tol
        @test unwrapvec(x) ≈ V * Diagonal(invS) * U' * b

        λ = rand(real(T))
        alg = LSMR(; maxiter=2 * n, tol=tol, verbosity=0)
        x, info = @constinferred lssolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode)), alg,
                                         λ)

        r = b - A * unwrapvec(x)
        @test info.converged > 0
        @test A' * r ≈ λ^2 * unwrapvec(x)
    end
end
@testset "LSMR large problem ($mode)" for mode in (:vector, :inplace, :outplace, :mixed)
    scalartypes = mode === :vector ? (Float64, ComplexF64) : (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (2 * N, N)) .- (one(T) / 2)
        b = rand(T, 2 * N) .- (one(T) / 2)

        tol = 10 * N * eps(real(T))
        x, info = @constinferred lssolve(wrapop(A, Val(mode)), wrapvec(b, Val(mode));
                                         maxiter=N, tol=tol, verbosity=0)

        r = b - A * unwrapvec(x)
        @test info.converged > 0
        @test norm(A' * r) < 2 * tol
    end
end