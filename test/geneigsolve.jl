@testset "GolubYe - geneigsolve full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            A = (A + A') / 2
            B = rand(T, (n, n)) .- one(T) / 2
            B = sqrt(B * B')
            v = rand(T, (n,))
            alg = GolubYe(;
                          orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=WARN_LEVEL)
            n1 = div(n, 2)
            D1, V1, info = @constinferred geneigsolve((wrapop(A, Val(mode)),
                                                       wrapop(B, Val(mode))),
                                                      wrapvec(v, Val(mode)),
                                                      n1, :SR; orth=orth, krylovdim=n,
                                                      maxiter=1, tol=tolerance(T),
                                                      ishermitian=true, isposdef=true,
                                                      verbosity=SILENT_LEVEL)

            if info.converged < n1
                @test_logs geneigsolve((wrapop(A, Val(mode)),
                                        wrapop(B, Val(mode))),
                                       wrapvec(v, Val(mode)),
                                       n1, :SR; orth=orth, krylovdim=n,
                                       maxiter=1, tol=tolerance(T),
                                       ishermitian=true, isposdef=true,
                                       verbosity=SILENT_LEVEL)
                @test_logs geneigsolve((wrapop(A, Val(mode)),
                                        wrapop(B, Val(mode))),
                                       wrapvec(v, Val(mode)),
                                       n1, :SR; orth=orth, krylovdim=n,
                                       maxiter=1, tol=tolerance(T),
                                       ishermitian=true, isposdef=true,
                                       verbosity=WARN_LEVEL)
                @test_logs (:warn,) geneigsolve((wrapop(A, Val(mode)),
                                                 wrapop(B, Val(mode))),
                                                wrapvec(v, Val(mode)),
                                                n1, :SR; orth=orth, krylovdim=n1 + 1,
                                                maxiter=1, tol=tolerance(T),
                                                ishermitian=true, isposdef=true,
                                                verbosity=WARN_LEVEL)
                @test_logs (:info,) geneigsolve((wrapop(A, Val(mode)),
                                                 wrapop(B, Val(mode))),
                                                wrapvec(v, Val(mode)),
                                                n1, :SR; orth=orth, krylovdim=n,
                                                maxiter=1, tol=tolerance(T),
                                                ishermitian=true, isposdef=true,
                                                verbosity=STARTSTOP_LEVEL)
                alg = GolubYe(;
                              orth=orth, krylovdim=n1, maxiter=3, tol=tolerance(T),
                              verbosity=EACHITERATION_LEVEL)
                @test_logs((:info,), (:info,), (:info,), (:warn,),
                           geneigsolve((wrapop(A, Val(mode)), wrapop(B, Val(mode))),
                                       wrapvec(v, Val(mode)), 1, :SR, alg))
                alg = GolubYe(;
                              orth=orth, krylovdim=3, maxiter=2, tol=tolerance(T),
                              verbosity=EACHITERATION_LEVEL + 1)
                @test_logs((:info,), (:info,), (:info,), (:info,),
                           (:info,), (:info,), (:info,), (:info,), (:warn,),
                           geneigsolve((wrapop(A, Val(mode)), wrapop(B, Val(mode))),
                                       wrapvec(v, Val(mode)), 1, :SR, alg))
            end
            @test KrylovKit.geneigselector((wrapop(A, Val(mode)), wrapop(B, Val(mode))),
                                           scalartype(v); orth=orth, krylovdim=n,
                                           maxiter=1, tol=tolerance(T), ishermitian=true,
                                           isposdef=true) isa GolubYe
            n2 = n - n1
            D2, V2, info = @constinferred geneigsolve((wrapop(A, Val(mode)),
                                                       wrapop(B, Val(mode))),
                                                      wrapvec(v, Val(mode)),
                                                      n2, :LR, alg)
            @test vcat(D1[1:n1], reverse(D2[1:n2])) ≈ eigvals(A, B)

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test U1' * B * U1 ≈ I
            @test U2' * B * U2 ≈ I

            @test A * U1 ≈ B * U1 * Diagonal(D1)
            @test A * U2 ≈ B * U2 * Diagonal(D2)
        end
    end
end

@testset "GolubYe - geneigsolve iteratively ($mode)" for mode in
                                                         (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float64, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (N, N)) .- one(T) / 2
            A = (A + A') / 2
            B = rand(T, (N, N)) .- one(T) / 2
            B = sqrt(B * B')
            v = rand(T, (N,))
            alg = GolubYe(;
                          orth=orth, krylovdim=3 * n, maxiter=100,
                          tol=cond(B) * tolerance(T), verbosity=SILENT_LEVEL)
            D1, V1, info1 = @constinferred geneigsolve((wrapop(A, Val(mode)),
                                                        wrapop(B, Val(mode))),
                                                       wrapvec(v, Val(mode)),
                                                       n, :SR, alg)
            D2, V2, info2 = geneigsolve((wrapop(A, Val(mode)), wrapop(B, Val(mode))),
                                        wrapvec(v, Val(mode)), n, :LR, alg)

            l1 = info1.converged
            l2 = info2.converged
            @test l1 > 0
            @test l2 > 0
            @test D1[1:l1] ≊ eigvals(A, B)[1:l1]
            @test D2[1:l2] ≊ eigvals(A, B)[N:-1:(N - l2 + 1)]

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test U1' * B * U1 ≈ I
            @test U2' * B * U2 ≈ I

            R1 = stack(unwrapvec, info1.residual)
            R2 = stack(unwrapvec, info2.residual)
            @test A * U1 ≈ B * U1 * Diagonal(D1) + R1
            @test A * U2 ≈ B * U2 * Diagonal(D2) + R2
        end
    end
end
