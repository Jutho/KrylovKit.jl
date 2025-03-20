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
            D1, V1, W1, info1 = @constinferred bieigsolve(wrapop(A,
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
            D2, V2, W2, info2 = @constinferred bieigsolve(wrapop(A,
                                                                  Val(mode)),
                                                           wrapvec(v,
                                                                   Val(mode)),
                                                           wrapvec(w,
                                                                   Val(mode)),
                                                           n2, :LR, alg)
            D = sort(sort(eigvals(A); by=imag, rev=true); alg=MergeSort, by=real)
            D2′ = sort(sort(D2; by=imag, rev=true); alg=MergeSort, by=real)
            @test vcat(D1[1:n1], D2′[(end - n2 + 1):end]) ≊ D

            UV1 = stack(unwrapvec, V1)
            UV2 = stack(unwrapvec, V2)
            @test A * UV1 ≈ UV1 * Diagonal(D1)
            @test A * UV2 ≈ UV2 * Diagonal(D2)

            UW1 = stack(unwrapvec, W1)
            UW2 = stack(unwrapvec, W2)
            if T <: Real 
                @test A'UW1 ≈ UW1 * Diagonal(D1)
                @test A'UW2 ≈ UW2 * Diagonal(D2)
            else
                @test A'UW1 ≈ UW1 * Diagonal(conj.(D1))
                @test A'UW2 ≈ UW2 * Diagonal(conj.(D2))
            end

            # if T <: Complex
            #     n1 = div(n, 2)
            #     D1, V1, W1, info = bieigsolve(wrapop(A, Val(mode)),
            #                                             wrapvec(v, Val(mode)),
            #                                             wrapvec(w, Val(mode)), n1,
            #                                             :SI,
            #                                             alg)
            #     n2 = n - n1
            #     D2, V2, W2, info = bieigsolve(wrapop(A, Val(mode)),
            #                                             wrapvec(v, Val(mode)),
            #                                             wrapvec(w, Val(mode)), n2,
            #                                             :LI,
            #                                             alg)
            #     D = sort(eigvals(A); by=imag)
            #     @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ D

            #     UV1 = stack(unwrapvec, V1)
            #     UV2 = stack(unwrapvec, V2)
            #     @test A * UV1 ≈ UV1 * Diagonal(D1)
            #     @test A * UV2 ≈ UV2 * Diagonal(D2)

            #     UW1 = stack(unwrapvec, W1)
            #     UW2 = stack(unwrapvec, W2)
            #     @test A'UW1 ≈ UW1 * Diagonal(conj.(D1))
            #     @test A'UW2 ≈ UW2 * Diagonal(conj.(D2))
            # end

            # alg = Arnoldi(; orth=orth, krylovdim=2n, maxiter=1, tol=tolerance(T),
            #               verbosity=1)
            # @test_logs (:warn,) (:warn,) eigsolve(wrapop(A, Val(mode)),
            #                                       wrapvec(v, Val(mode)), n + 1, :LM, alg)
        end
    end
end
