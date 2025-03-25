@testset "Lanczos - eigsolve full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            A = (A + A') / 2
            v = rand(T, (n,))
            n1 = div(n, 2)
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=2)
            D1, V1, info = @test_logs (:info,) eigsolve(wrapop(A, Val(mode)),
                                                        wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs eigsolve(wrapop(A, Val(mode)),
                                wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Lanczos(; orth=orth, krylovdim=n1 + 1, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) eigsolve(wrapop(A, Val(mode)),
                                         wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=2)
            @test_logs (:info,) eigsolve(wrapop(A, Val(mode)),
                                         wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Lanczos(; orth=orth, krylovdim=n1, maxiter=3, tol=tolerance(T),
                          verbosity=3)
            @test_logs((:info,), (:info,), (:info,), (:warn,),
                       eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg))
            alg = Lanczos(; orth=orth, krylovdim=4, maxiter=1, tol=tolerance(T),
                          verbosity=4)
            # since it is impossible to know exactly the size of the Krylov subspace after shrinking,
            # we only know the output for a sigle iteration
            @test_logs((:info,), (:info,), (:info,), (:info,), (:info,), (:warn,),
                       eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg))

            @test KrylovKit.eigselector(wrapop(A, Val(mode)), scalartype(v); krylovdim=n,
                                        maxiter=1,
                                        tol=tolerance(T), ishermitian=true) isa Lanczos
            n2 = n - n1
            alg = Lanczos(; krylovdim=2 * n, maxiter=1, tol=tolerance(T))
            D2, V2, info = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                   wrapvec(v, Val(mode)),
                                                   n2, :LR, alg)
            @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ eigvals(A)

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I

            @test A * U1 ≈ U1 * Diagonal(D1)
            @test A * U2 ≈ U2 * Diagonal(D2)

            alg = Lanczos(; orth=orth, krylovdim=2n, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) (:warn,) eigsolve(wrapop(A, Val(mode)),
                                                  wrapvec(v, Val(mode)), n + 1, :LM, alg)
        end
    end
end

@testset "Lanczos - eigsolve iteratively ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (N, N)) .- one(T) / 2
            A = (A + A') / 2
            v = rand(T, (N,))
            alg = Lanczos(; krylovdim=2 * n, maxiter=10,
                          tol=tolerance(T), eager=true, verbosity=0)
            D1, V1, info1 = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                    wrapvec(v, Val(mode)), n, :SR, alg)
            D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n, :LR,
                                     alg)

            l1 = info1.converged
            l2 = info2.converged
            @test l1 > 0
            @test l2 > 0
            @test D1[1:l1] ≈ eigvals(A)[1:l1]
            @test D2[1:l2] ≈ eigvals(A)[N:-1:(N - l2 + 1)]

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I

            R1 = stack(unwrapvec, info1.residual)
            R2 = stack(unwrapvec, info2.residual)
            @test A * U1 ≈ U1 * Diagonal(D1) + R1
            @test A * U2 ≈ U2 * Diagonal(D2) + R2
        end
    end
end

@testset "Arnoldi - eigsolve full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            v = rand(T, (n,))
            n1 = div(n, 2)
            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T))
            D1, V1, info1 = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                    wrapvec(v, Val(mode)), n1, :SR, alg)

            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=0)
            @test_logs eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Arnoldi(; orth=orth, krylovdim=n1 + 2, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                                         :SR, alg)
            alg = Arnoldi(; orth=orth, krylovdim=n, maxiter=1, tol=tolerance(T),
                          verbosity=2)
            @test_logs (:info,) eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                                         :SR, alg)
            alg = Arnoldi(; orth=orth, krylovdim=n1, maxiter=3, tol=tolerance(T),
                          verbosity=3)
            @test_logs((:info,), (:info,), (:info,), (:warn,),
                       eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg))
            alg = Arnoldi(; orth=orth, krylovdim=4, maxiter=1, tol=tolerance(T),
                          verbosity=4)
            # since it is impossible to know exactly the size of the Krylov subspace after shrinking,
            # we only know the output for a sigle iteration
            @test_logs((:info,), (:info,), (:info,), (:info,), (:info,), (:warn,),
                       eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg))

            @test KrylovKit.eigselector(wrapop(A, Val(mode)), eltype(v); orth=orth,
                                        krylovdim=n, maxiter=1,
                                        tol=tolerance(T)) isa Arnoldi
            n2 = n - n1
            alg = Arnoldi(; orth=orth, krylovdim=2 * n, maxiter=1, tol=tolerance(T))
            D2, V2, info2 = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                    wrapvec(v, Val(mode)), n2, :LR, alg)
            D = sort(sort(eigvals(A); by=imag, rev=true); alg=MergeSort, by=real)
            D2′ = sort(sort(D2; by=imag, rev=true); alg=MergeSort, by=real)
            @test vcat(D1[1:n1], D2′[(end - n2 + 1):end]) ≈ D

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test A * U1 ≈ U1 * Diagonal(D1)
            @test A * U2 ≈ U2 * Diagonal(D2)

            if T <: Complex
                n1 = div(n, 2)
                D1, V1, info = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                                        :SI,
                                        alg)
                n2 = n - n1
                D2, V2, info = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n2,
                                        :LI,
                                        alg)
                D = sort(eigvals(A); by=imag)

                @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ D

                U1 = stack(unwrapvec, V1)
                U2 = stack(unwrapvec, V2)
                @test A * U1 ≈ U1 * Diagonal(D1)
                @test A * U2 ≈ U2 * Diagonal(D2)
            end

            alg = Arnoldi(; orth=orth, krylovdim=2n, maxiter=1, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) (:warn,) eigsolve(wrapop(A, Val(mode)),
                                                  wrapvec(v, Val(mode)), n + 1, :LM, alg)
        end
    end
end

@testset "Arnoldi - eigsolve iteratively ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (N, N)) .- one(T) / 2
            v = rand(T, (N,))
            alg = Arnoldi(; krylovdim=3 * n, maxiter=20,
                          tol=tolerance(T), eager=true, verbosity=0)
            D1, V1, info1 = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                    wrapvec(v, Val(mode)), n, :SR, alg)
            D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n, :LR,
                                     alg)
            D3, V3, info3 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n, :LM,
                                     alg)
            D = sort(eigvals(A); by=imag, rev=true)

            l1 = info1.converged
            l2 = info2.converged
            l3 = info3.converged
            @test l1 > 0
            @test l2 > 0
            @test l3 > 0
            @test D1[1:l1] ≊ sort(D; alg=MergeSort, by=real)[1:l1]
            @test D2[1:l2] ≊ sort(D; alg=MergeSort, by=real, rev=true)[1:l2]
            # sorting by abs does not seem very reliable if two distinct eigenvalues are close
            # in absolute value, so we perform a second sort afterwards using the real part
            @test D3[1:l3] ≊ sort(D; by=abs, rev=true)[1:l3]

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            U3 = stack(unwrapvec, V3)
            R1 = stack(unwrapvec, info1.residual)
            R2 = stack(unwrapvec, info2.residual)
            R3 = stack(unwrapvec, info3.residual)
            @test A * U1 ≈ U1 * Diagonal(D1) + R1
            @test A * U2 ≈ U2 * Diagonal(D2) + R2
            @test A * U3 ≈ U3 * Diagonal(D3) + R3

            if T <: Complex
                D1, V1, info1 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                         :SI, alg)
                D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                         :LI, alg)
                D = eigvals(A)

                l1 = info1.converged
                l2 = info2.converged
                @test l1 > 0
                @test l2 > 0
                @test D1[1:l1] ≈ sort(D; by=imag)[1:l1]
                @test D2[1:l2] ≈ sort(D; by=imag, rev=true)[1:l2]

                U1 = stack(unwrapvec, V1)
                U2 = stack(unwrapvec, V2)
                R1 = stack(unwrapvec, info1.residual)
                R2 = stack(unwrapvec, info2.residual)
                @test A * U1 ≈ U1 * Diagonal(D1) + R1
                @test A * U2 ≈ U2 * Diagonal(D2) + R2
            end
        end
    end
end

@testset "Arnoldi - realeigsolve iteratively ($mode)" for mode in
                                                          (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64) : (Float64,)
    orths = mode === :vector ? (cgs2, mgs2, cgsr, mgsr) : (mgsr,)
    @testset for T in scalartypes
        @testset for orth in orths
            V = exp(randn(T, (N, N)) / 10)
            D = randn(T, N)
            A = V * Diagonal(D) / V
            v = rand(T, (N,))
            alg = Arnoldi(; krylovdim=3 * n, maxiter=20,
                          tol=tolerance(T), eager=true, verbosity=0)
            D1, V1, info1 = @constinferred realeigsolve(wrapop(A, Val(mode)),
                                                        wrapvec(v, Val(mode)), n, :SR, alg)
            D2, V2, info2 = realeigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                         :LR,
                                         alg)
            D3, V3, info3 = realeigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                                         :LM,
                                         alg)
            l1 = info1.converged
            l2 = info2.converged
            l3 = info3.converged
            @test l1 > 0
            @test l2 > 0
            @test l3 > 0
            @test D1[1:l1] ≊ sort(D; alg=MergeSort)[1:l1]
            @test D2[1:l2] ≊ sort(D; alg=MergeSort, rev=true)[1:l2]
            # sorting by abs does not seem very reliable if two distinct eigenvalues are close
            # in absolute value, so we perform a second sort afterwards using the real part
            @test D3[1:l3] ≊ sort(D; by=abs, rev=true)[1:l3]

            @test eltype(D1) == T
            @test eltype(D2) == T
            @test eltype(D3) == T

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            U3 = stack(unwrapvec, V3)
            R1 = stack(unwrapvec, info1.residual)
            R2 = stack(unwrapvec, info2.residual)
            R3 = stack(unwrapvec, info3.residual)
            @test A * U1 ≈ U1 * Diagonal(D1) + R1
            @test A * U2 ≈ U2 * Diagonal(D2) + R2
            @test A * U3 ≈ U3 * Diagonal(D3) + R3

            if mode == :vector # solve eigenvalue problem as complex problem with real linear operator
                V = exp(randn(T, (2N, 2N)) / 10)
                D = randn(T, 2N)
                Ar = V * Diagonal(D) / V
                Z = zeros(T, N, N)
                J = [Z -I; I Z]
                Ar1 = (Ar - J * Ar * J) / 2
                Ar2 = (Ar + J * Ar * J) / 2
                A = complex.(Ar1[1:N, 1:N], -Ar1[1:N, (N + 1):end])
                B = complex.(Ar2[1:N, 1:N], +Ar2[1:N, (N + 1):end])
                f = buildrealmap(A, B)
                v = rand(complex(T), (N,))
                alg = Arnoldi(; krylovdim=3 * n, maxiter=20,
                              tol=tolerance(T), eager=true, verbosity=0)
                D1, V1, info1 = @constinferred realeigsolve(f, v, n, :SR, alg)
                D2, V2, info2 = realeigsolve(f, v, n, :LR, alg)
                D3, V3, info3 = realeigsolve(f, v, n, :LM, alg)

                l1 = info1.converged
                l2 = info2.converged
                l3 = info3.converged
                @test l1 > 0
                @test l2 > 0
                @test l3 > 0
                @test D1[1:l1] ≊ sort(D; alg=MergeSort)[1:l1]
                @test D2[1:l2] ≊ sort(D; alg=MergeSort, rev=true)[1:l2]
                # sorting by abs does not seem very reliable if two distinct eigenvalues are close
                # in absolute value, so we perform a second sort afterwards using the real part
                @test D3[1:l3] ≊ sort(D; by=abs, rev=true)[1:l3]

                @test eltype(D1) == T
                @test eltype(D2) == T
                @test eltype(D3) == T

                U1 = stack(V1)
                U2 = stack(V2)
                U3 = stack(V3)
                R1 = stack(info1.residual)
                R2 = stack(info2.residual)
                R3 = stack(info3.residual)
                @test A * U1 + B * conj(U1) ≈ U1 * Diagonal(D1) + R1
                @test A * U2 + B * conj(U2) ≈ U2 * Diagonal(D2) + R2
                @test A * U3 + B * conj(U3) ≈ U3 * Diagonal(D3) + R3
            end
        end
    end
end

@testset "Arnoldi - realeigsolve imaginary eigenvalue warning" begin
    A = diagm(vcat(1, 1, exp.(-(0.1:0.02:2))))
    A[2, 1] = 1e-9
    A[1, 2] = -1e-9
    v = ones(Float64, size(A, 1))
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-8, verbosity=0))
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-8, verbosity=1))
    @test_logs (:info,) realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-8, verbosity=2))
    @test_logs (:warn,) realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-10, verbosity=1))
    @test_logs (:warn,) (:info,) realeigsolve(A, v, 1, :LM,
                                              Arnoldi(; tol=1e-10, verbosity=2))

    # this should not trigger a warning
    A[1, 2] = A[2, 1] = 0
    A[1, 1] = 1
    A[2, 2] = A[3, 3] = 0.99
    A[3, 2] = 1e-6
    A[2, 3] = -1e-6
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-12, verbosity=0))
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-12, verbosity=1))
    @test_logs (:info,) realeigsolve(A, v, 1, :LM, Arnoldi(; tol=1e-12, verbosity=2))
end

using LinearAlgebra,KrylovKit,Test,Random,Yao
@testset "Block Lanczos - eigsolve " begin
	function toric_code_strings(m::Int, n::Int)
		li = LinearIndices((m, n))
		bottom(i, j) = li[mod1(i, m), mod1(j, n)] + m * n
		right(i, j) = li[mod1(i, m), mod1(j, n)]
		xstrings = Vector{Int}[]
		zstrings = Vector{Int}[]
		for i ∈ 1:m, j ∈ 1:n
			# face center
			push!(xstrings, [bottom(i, j - 1), right(i, j), bottom(i, j), right(i - 1, j)])
			# cross
			push!(zstrings, [right(i, j), bottom(i, j), right(i, j + 1), bottom(i + 1, j)])
		end
		return xstrings, zstrings
	end
	function toric_code_hamiltonian(m::Int, n::Int)
		xstrings, zstrings = toric_code_strings(m, n)
		sum([kron(2m * n, [i => X for i in xs]...) for xs in xstrings[1:end-1]]) + sum([kron(2m * n, [i => Z for i in zs]...) for zs in zstrings[1:end-1]])
	end
	h = toric_code_hamiltonian(3, 3)

	Random.seed!(4)
	p = 8 # block size
	X1 = Matrix(qr(rand(2^18, p)).Q)
	D, U, info = eigsolve(-mat(h), X1, 10, :SR, BlockLanczos(p))
    @show D[1:10]
    @test count(x -> abs(x+16.0) < 1.9, D[1:10]) == 4
    @test count(x -> abs(x+16.0) < 1e-8, D[1:10]) == 4
end