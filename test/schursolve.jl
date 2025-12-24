@testset "Arnoldi - schursolve full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            v = rand(T, (n,))
            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T))
            n1 = div(n, 2)
            T1, V1, D1, info1 = @constinferred schursolve(wrapop(A, Val(mode)),
                                                          wrapvec(v, Val(mode)), n1, :SR,
                                                          alg)
            @test_logs schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1, :SR,
                                  alg)
            alg = Arnoldi(;
                          orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=WARN_LEVEL)
            @test_logs schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1, :SR,
                                  alg)
            alg = Arnoldi(;
                          orth=orth, krylovdim=n1 + 1, maxiter=1, tol=tolerance(T),
                          verbosity=WARN_LEVEL)
            @test_logs (:warn,) schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                                           :SR,
                                           alg)
            alg = Arnoldi(;
                          orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=STARTSTOP_LEVEL)
            @test_logs (:info,) schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                                           :SR,
                                           alg)

            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T))
            n2 = n - n1
            T2, V2, D2, info2 = schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n2,
                                           :LR, alg)
            D = sort(sort(eigvals(A); by=imag, rev=true); alg=MergeSort, by=real)
            D2′ = sort(sort(D2; by=imag, rev=true); alg=MergeSort, by=real)

            @test vcat(D1[1:n1], D2′[(end - n2 + 1):end]) ≈ D

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I

            @test A * U1 ≈ U1 * T1
            @test A * U2 ≈ U2 * T2

            if T <: Complex
                T1, V1, D1, info = schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                              n1, :SI, alg)
                T2, V2, D2, info = schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                              n2, :LI, alg)
                D = sort(eigvals(A); by=imag)

                @test vcat(D1[1:n1], reverse(D2[1:n2])) ≈ D

                U1 = stack(unwrapvec, V1)
                U2 = stack(unwrapvec, V2)
                @test U1' * U1 ≈ I
                @test U2' * U2 ≈ I

                @test A * U1 ≈ U1 * T1
                @test A * U2 ≈ U2 * T2
            end
        end
    end
end

@testset "Arnoldi - schursolve iteratively ($mode)" for mode in
                                                        (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (N, N)) .- one(T) / 2
            v = rand(T, (N,))
            alg = Arnoldi(;
                          orth=orth, krylovdim=3 * n, maxiter=10, tol=tolerance(T),
                          verbosity=SILENT_LEVEL)
            T1, V1, D1, info1 = @constinferred schursolve(wrapop(A, Val(mode)),
                                                          wrapvec(v, Val(mode)), n, :SR,
                                                          alg)
            T2, V2, D2, info2 = schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                           :LR, alg)
            T3, V3, D3, info3 = schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                           :LM, alg)
            D = sort(eigvals(A); by=imag, rev=true)

            l1 = info1.converged
            l2 = info2.converged
            l3 = info3.converged
            @test D1[1:l1] ≊ sort(D; alg=MergeSort, by=real)[1:l1]
            @test D2[1:l2] ≊ sort(D; alg=MergeSort, by=real, rev=true)[1:l2]
            @test D3[1:l3] ≊ sort(D; alg=MergeSort, by=abs, rev=true)[1:l3]

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            U3 = stack(unwrapvec, V3)
            @test U1' * U1 ≈ one(U1' * U1)
            @test U2' * U2 ≈ one(U2' * U2)
            @test U3' * U3 ≈ one(U3' * U3)

            R1 = stack(unwrapvec, info1.residual)
            R2 = stack(unwrapvec, info2.residual)
            R3 = stack(unwrapvec, info3.residual)
            @test A * U1 ≈ U1 * T1 + R1
            @test A * U2 ≈ U2 * T2 + R2
            @test A * U3 ≈ U3 * T3 + R3

            if T <: Complex
                T1, V1, D1, info1 = schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                               n, :SI, alg)
                T2, V2, D2, info2 = schursolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)),
                                               n, :LI, alg)
                D = eigvals(A)

                l1 = info1.converged
                l2 = info2.converged
                @test D1[1:l1] ≊ sort(D; by=imag)[1:l1]
                @test D2[1:l2] ≊ sort(D; by=imag, rev=true)[1:l2]

                U1 = stack(unwrapvec, V1)
                U2 = stack(unwrapvec, V2)
                @test U1[:, 1:l1]' * U1[:, 1:l1] ≈ I
                @test U2[:, 1:l2]' * U2[:, 1:l2] ≈ I

                R1 = stack(unwrapvec, info1.residual)
                R2 = stack(unwrapvec, info2.residual)
                @test A * U1 ≈ U1 * T1 + R1
                @test A * U2 ≈ U2 * T2 + R2
            end
        end
    end
end
