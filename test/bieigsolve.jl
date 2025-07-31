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
            alg = BiArnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                            verbosity=STARTSTOP_LEVEL)
            #! format: off
            D1, (V1, W1), (infoV1, infoV2) =
                @constinferred bieigsolve(wrapop(A, Val(mode)),
                                          wrapvec(v, Val(mode)), wrapvec(w, Val(mode)),
                                          n1, :SR, alg)
            #! format: on

            # Some of these still fail
            alg = BiArnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                            verbosity=SILENT_LEVEL)
            @test_logs bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                  wrapvec(w, Val(mode)), n1, :SR, alg)
            alg = BiArnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                            verbosity=WARN_LEVEL)
            @test_logs bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                  wrapvec(w, Val(mode)), n1, :SR, alg)
            alg = BiArnoldi(; orth=orth, krylovdim=n1 + 2, maxiter=1, tol=tolerance(T),
                            verbosity=WARN_LEVEL)
            @test_logs (:warn,) bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                           wrapvec(w, Val(mode)), n1,
                                           :SR, alg)
            alg = BiArnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                            verbosity=STARTSTOP_LEVEL)
            @test_logs (:info,) bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                           wrapvec(w, Val(mode)), n1,
                                           :SR, alg)
            alg = BiArnoldi(; orth=orth, krylovdim=n1, maxiter=3, tol=tolerance(T),
                            verbosity=EACHITERATION_LEVEL)
            @test_logs((:info,), (:info,), (:info,), (:warn,),
                       bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                  wrapvec(w, Val(mode)), 1, :SR, alg))
            alg = BiArnoldi(; orth=orth, krylovdim=4, maxiter=1, tol=tolerance(T),
                            verbosity=EACHITERATION_LEVEL + 1)
            # since it is impossible to know exactly the size of the Krylov subspace after shrinking,
            # we only know the output for a sigle iteration
            @test_logs((:info,), (:info,), (:info,), (:info,),
                       (:info,), (:info,), (:info,), (:info,), (:info,), (:warn,),
                       bieigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                  wrapvec(w, Val(mode)), 1, :SR, alg))

            n2 = n - n1
            alg = BiArnoldi(; orth=orth, krylovdim=2 * n, maxiter=1, tol=tolerance(T))
            #! format: off
            D2, (V2, W2), (infoV2, infoW2) =
                @constinferred bieigsolve(wrapop(A, Val(mode)), 
                                          wrapvec(v, Val(mode)), wrapvec(w, Val(mode)),
                                          n2, :LR, alg)
            #! format: on
            D = sort(sort(eigvals(A); by=imag, rev=true); alg=MergeSort, by=real)
            D2′ = sort(sort(D2; by=imag, rev=true); alg=MergeSort, by=real)
            @test vcat(D1[1:n1], D2′[(end - n2 + 1):end]) ≊ D

            UV1 = stack(unwrapvec, V1)
            UV2 = stack(unwrapvec, V2)
            @test A * UV1 ≈ UV1 * Diagonal(D1)
            @test A * UV2 ≈ UV2 * Diagonal(D2)

            UW1 = stack(unwrapvec, W1)
            UW2 = stack(unwrapvec, W2)
            @test A' * UW1 ≈ UW1 * Diagonal(conj.(D1))
            @test A' * UW2 ≈ UW2 * Diagonal(conj.(D2))

            # check biorthogonality
            @test UW1' * UV1 ≈ Matrix(I, size(UW1, 2), size(UV1, 2))
            @test UW2' * UV2 ≈ Matrix(I, size(UW2, 2), size(UV2, 2))
            @test norm(UW1[:, 1:n1]' * UV2[:, 1:n2], Inf) < tolerance(T)
            @test norm(UW2[:, 1:n2]' * UV1[:, 1:n1], Inf) < tolerance(T)

            if T <: Complex
                n1 = div(n, 2)
                D1, (V1, W1), (infoV1, infoW1) = bieigsolve(wrapop(A, Val(mode)),
                                                            wrapvec(v, Val(mode)),
                                                            wrapvec(w, Val(mode)),
                                                            n1, :SI, alg)
                n2 = n - n1
                D2, (V2, W2), (infoV2, infoW2) = bieigsolve(wrapop(A, Val(mode)),
                                                            wrapvec(v, Val(mode)),
                                                            wrapvec(w, Val(mode)),
                                                            n2, :LI, alg)
                D = sort(eigvals(A); by=imag)
                @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ D

                UV1 = stack(unwrapvec, V1)
                UV2 = stack(unwrapvec, V2)
                @test A * UV1 ≈ UV1 * Diagonal(D1)
                @test A * UV2 ≈ UV2 * Diagonal(D2)

                UW1 = stack(unwrapvec, W1)
                UW2 = stack(unwrapvec, W2)
                @test A' * UW1 ≈ UW1 * Diagonal(conj.(D1))
                @test A' * UW2 ≈ UW2 * Diagonal(conj.(D2))

                # check biorthogonality
                @test UW1' * UV1 ≈ Matrix(I, size(UW1, 2), size(UV1, 2))
                @test UW2' * UV2 ≈ Matrix(I, size(UW2, 2), size(UV2, 2))
                @test norm(UW1[:, 1:n1]' * UV2[:, 1:n2], Inf) < tolerance(T)
                @test norm(UW2[:, 1:n2]' * UV1[:, 1:n1], Inf) < tolerance(T)
            end

            # alg = Arnoldi(; orth=orth, krylovdim=2n, maxiter=1, tol=tolerance(T),
            #               verbosity=1)
            # @test_logs (:warn,) (:warn,) eigsolve(wrapop(A, Val(mode)),
            #                                       wrapvec(v, Val(mode)), n + 1, :LM, alg)
        end
    end
end

# @testset "BiArnoldi - eigsolve iteratively ($mode)" for mode in
#                                                         (:vector, :inplace, :outplace)
#     scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
#                   (ComplexF64,)
#     orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
#     @testset for T in scalartypes
#         @testset for orth in orths
#             A = rand(T, (N, N)) .- one(T) / 2
#             v = rand(T, (N,))
#             w = rand(T, (N,))
#             alg = BiArnoldi(; krylovdim=3 * n, maxiter=20, orth = orth,
#                             tol=tolerance(T), eager=true, verbosity=EACHITERATION_LEVEL)
#             #! format: off
#             D1, (V1, W1), (infoV1, infoW1) =
#                 @constinferred bieigsolve(wrapop(A, Val(mode)),
#                                                  wrapvec(v, Val(mode)), wrapvec(w, Val(mode)),
#                                                  n, :SR, alg)
#             #! format: on
#             D2, (V2, W2), (infoV2, infoW2) = bieigsolve(wrapop(A, Val(mode)),
#                                                         wrapvec(v, Val(mode)),
#                                                         wrapvec(w, Val(mode)),
#                                                         n, :LR, alg)
#             D3, (V3, W3), (infoV3, infoW3) = bieigsolve(wrapop(A, Val(mode)),
#                                                        wrapvec(v, Val(mode)),
#                                                        wrapvec(w, Val(mode)),
#                                                        n, :LM, alg)
#             D = sort(eigvals(A); by=imag, rev=true)

#             l1 = infoV1.converged
#             l2 = infoV2.converged
#             l3 = infoV3.converged
#             @test l1 > 0
#             @test l2 > 0
#             @test l3 > 0
#             @test D1[1:l1] ≊ sort(D; alg=MergeSort, by=real)[1:l1]
#             @test D2[1:l2] ≊ sort(D; alg=MergeSort, by=real, rev=true)[1:l2]
#             # sorting by abs does not seem very reliable if two distinct eigenvalues are close
#             # in absolute value, so we perform a second sort afterwards using the real part
#             @test D3[1:l3] ≊ sort(D; by=abs, rev=true)[1:l3]

#             U1 = stack(unwrapvec, V1)
#             U2 = stack(unwrapvec, V2)
#             U3 = stack(unwrapvec, V3)
#             R1 = stack(unwrapvec, infoV1.residual)
#             R2 = stack(unwrapvec, infoV2.residual)
#             R3 = stack(unwrapvec, infoV3.residual)
#             @test A * U1 ≈ U1 * Diagonal(D1) + R1
#             @test A * U2 ≈ U2 * Diagonal(D2) + R2
#             @test A * U3 ≈ U3 * Diagonal(D3) + R3

#             U1 = stack(unwrapvec, W1)
#             U2 = stack(unwrapvec, W2)
#             U3 = stack(unwrapvec, W3)
#             R1 = stack(unwrapvec, infoW1.residual)
#             R2 = stack(unwrapvec, infoW2.residual)
#             R3 = stack(unwrapvec, infoW3.residual)
#             @test A' * U1 ≈ U1 * Diagonal(D1) + R1
#             @test A' * U2 ≈ U2 * Diagonal(D2) + R2
#             @test A' * U3 ≈ U3 * Diagonal(D3) + R3

#             if T <: Complex
#                 D1, V1, info1 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
#                                          :SI, alg)
#                 D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
#                                          :LI, alg)
#                 D = eigvals(A)

#                 l1 = info1.converged
#                 l2 = info2.converged
#                 @test l1 > 0
#                 @test l2 > 0
#                 @test D1[1:l1] ≈ sort(D; by=imag)[1:l1]
#                 @test D2[1:l2] ≈ sort(D; by=imag, rev=true)[1:l2]

#                 U1 = stack(unwrapvec, V1)
#                 U2 = stack(unwrapvec, V2)
#                 R1 = stack(unwrapvec, info1.residual)
#                 R2 = stack(unwrapvec, info2.residual)
#                 @test A * U1 ≈ U1 * Diagonal(D1) + R1
#                 @test A * U2 ≈ U2 * Diagonal(D2) + R2
#             end
#         end
#     end
# end