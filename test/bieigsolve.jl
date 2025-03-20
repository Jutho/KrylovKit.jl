@testset "BiArnoldi - eigsolve full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            v = rand(T, (n,))
            w = rand(T, (n,))
            n1 = div(n, 2)

            alg = BiArnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T))
            (D1v, D1w), (V1, W1), (info1v, info1w) = @constinferred bieigsolve(wrapop(A,
                                                                                      Val(mode)),
                                                                               wrapvec(v,
                                                                                       Val(mode)),
                                                                               wrapvec(w,
                                                                                       Val(mode)),
                                                                               n1, :SR, alg)

            # Some of these still fail
            # alg = BiArnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
            #                 verbosity=0)
            # @test_logs bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
            #                       wrapvec(w, Val(mode)), n1, :SR, alg)
            # alg = BiArnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
            #                 verbosity=1)
            # @test_logs bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
            #                       wrapvec(w, Val(mode)), n1, :SR, alg)
            # alg = BiArnoldi(; orth=orth, krylovdim=n1 + 2, maxiter=1, tol=tolerance(T),
            #                 verbosity=1)
            # @test_logs (:warn,) bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
            #                                wrapvec(w, Val(mode)), n1,
            #                                :SR, alg)
            # alg = BiArnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
            #                 verbosity=2)
            # @test_logs (:info,) bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
            #                                wrapvec(w, Val(mode)), n1,
            #                                :SR, alg)
            # alg = BiArnoldi(; orth=orth, krylovdim=n1, maxiter=3, tol=tolerance(T),
            #                 verbosity=3)
            # @test_logs((:info,), (:info,), (:info,), (:warn,),
            #            bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
            #                       wrapvec(w, Val(mode)), 1, :SR, alg))
            # alg = BiArnoldi(; orth=orth, krylovdim=4, maxiter=1, tol=tolerance(T),
            #                 verbosity=4)
            # since it is impossible to know exactly the size of the Krylov subspace after shrinking,
            # we only know the output for a sigle iteration
            # @test_logs((:info,), (:info,), (:info,), (:info,), (:info,), (:warn,),
            #            bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
            #                       wrapvec(w, Val(mode)), 1, :SR, alg))

            # @test KrylovKit.eigselector(wrapop(A, Val(mode)), eltype(v); orth=orth,
            #                             krylovdim=n, maxiter=1,
            #                             tol=tolerance(T)) isa BiArnoldi
            n2 = n - n1
            alg = BiArnoldi(; orth=orth, krylovdim=2 * n, maxiter=1, tol=tolerance(T))
            (D2v, D2w), (V2, W2), (info2v, info2w) = @constinferred bieigsolve(wrapop(A,
                                                                                      Val(mode)),
                                                                               wrapvec(v,
                                                                                       Val(mode)),
                                                                               wrapvec(w,
                                                                                       Val(mode)),
                                                                               n2, :LR, alg)
            D = sort(sort(eigvals(A); by=imag, rev=true); alg=MergeSort, by=real)
            D2v′ = sort(sort(D2v; by=imag, rev=true); alg=MergeSort, by=real)
            @test vcat(D1v[1:n1], D2v′[(end - n2 + 1):end]) ≊ D
            
            Dw = sort(sort(eigvals(adjoint(A)); by=imag, rev=true); alg=MergeSort, by=real)
            D2w′ = sort(sort(D2w; by=imag, rev=true); alg=MergeSort, by=real)
            @test vcat(D1w[1:n1], D2w′[(end - n2 + 1):end]) ≊ Dw

            UV1 = stack(unwrapvec, V1)
            UV2 = stack(unwrapvec, V2)
            @test A * UV1 ≈ UV1 * Diagonal(D1v)
            @test A * UV2 ≈ UV2 * Diagonal(D2v)


            UW1 = stack(unwrapvec, W1)
            UW2 = stack(unwrapvec, W2)
            @test A'UW1 ≈ UW1 * Diagonal(D1w)
            @test A'UW2 ≈ UW2 * Diagonal(D2w)

            if T <: Complex
                n1 = div(n, 2)
                (D1v, D1w), (V1, W1), info = bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), wrapvec(w, Val(mode)), n1,
                                          :SI,
                                          alg)
                n2 = n - n1
                (D2v, D2w), (V2, W2), info = bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), wrapvec(w, Val(mode)), n2,
                                          :LI,
                                          alg)
                Dv = sort(eigvals(A); by=imag)
                Dw = sort(eigvals(adjoint(A)); by=imag)

                @test vcat(D1v[1:n1], reverse(D2v[1:n2])) ≊ Dv
                @test vcat(D1w[1:n1], reverse(D2w[1:n2])) ≊ Dw

                UV1 = stack(unwrapvec, V1)
                UV2 = stack(unwrapvec, V2)
                @test A * UV1 ≈ UV1 * Diagonal(D1v)
                @test A * UV2 ≈ UV2 * Diagonal(D2v)
                
                UW1 = stack(unwrapvec, W1)
                UW2 = stack(unwrapvec, W2)
                @test A'UW1 ≈ UW1 * Diagonal(D1w)
                @test A'UW2 ≈ UW2 * Diagonal(D2w)
            end

            # alg = Arnoldi(; orth=orth, krylovdim=2n, maxiter=1, tol=tolerance(T),
            #               verbosity=1)
            # @test_logs (:warn,) (:warn,) eigsolve(wrapop(A, Val(mode)),
            #                                       wrapvec(v, Val(mode)), n + 1, :LM, alg)
        end
    end
end
