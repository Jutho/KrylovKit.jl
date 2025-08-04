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
            alg = Lanczos(;
                orth = orth, krylovdim = n, maxiter = 1, tol = tolerance(T),
                verbosity = STARTSTOP_LEVEL
            )
            D1, V1, info = @test_logs (:info,) eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n1, :SR, alg
            )
            alg = Lanczos(;
                orth = orth, krylovdim = n, maxiter = 1, tol = tolerance(T),
                verbosity = WARN_LEVEL
            )
            @test_logs eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n1, :SR, alg
            )
            alg = Lanczos(;
                orth = orth, krylovdim = n1 + 1, maxiter = 1, tol = tolerance(T),
                verbosity = WARN_LEVEL
            )
            @test_logs (:warn,) eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n1, :SR, alg
            )
            alg = Lanczos(;
                orth = orth, krylovdim = n, maxiter = 1, tol = tolerance(T),
                verbosity = STARTSTOP_LEVEL
            )
            @test_logs (:info,) eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n1, :SR, alg
            )
            alg = Lanczos(;
                orth = orth, krylovdim = n1, maxiter = 3, tol = tolerance(T),
                verbosity = EACHITERATION_LEVEL
            )
            @test_logs(
                (:info,), (:info,), (:info,), (:warn,),
                eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg)
            )
            alg = Lanczos(;
                orth = orth, krylovdim = 4, maxiter = 1, tol = tolerance(T),
                verbosity = EACHITERATION_LEVEL + 1
            )
            # since it is impossible to know exactly the size of the Krylov subspace after shrinking,
            # we only know the output for a sigle iteration
            @test_logs(
                (:info,), (:info,), (:info,), (:info,), (:info,), (:warn,),
                eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg)
            )

            @test KrylovKit.eigselector(
                wrapop(A, Val(mode)), scalartype(v); krylovdim = n,
                maxiter = 1,
                tol = tolerance(T), ishermitian = true
            ) isa Lanczos
            n2 = n - n1
            alg = Lanczos(; krylovdim = 2 * n, maxiter = 1, tol = tolerance(T))
            D2, V2, info = @constinferred eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)),
                n2, :LR, alg
            )
            @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ eigvals(A)

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I

            @test A * U1 ≈ U1 * Diagonal(D1)
            @test A * U2 ≈ U2 * Diagonal(D2)

            alg = Lanczos(;
                orth = orth, krylovdim = 2n, maxiter = 1, tol = tolerance(T),
                verbosity = WARN_LEVEL
            )
            @test_logs (:warn,) (:warn,) eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n + 1, :LM, alg
            )
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
            alg = Lanczos(;
                krylovdim = 2 * n, maxiter = 10,
                tol = tolerance(T), eager = true, verbosity = SILENT_LEVEL
            )
            D1, V1, info1 = @constinferred eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n, :SR, alg
            )
            D2, V2, info2 = eigsolve(
                wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n, :LR,
                alg
            )

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
            alg = Arnoldi(; orth = orth, krylovdim = n, maxiter = 1, tol = tolerance(T))
            D1, V1, info1 = @constinferred eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n1, :SR, alg
            )

            alg = Arnoldi(;
                orth = orth, krylovdim = n, maxiter = 1, tol = tolerance(T),
                verbosity = SILENT_LEVEL
            )
            @test_logs eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Arnoldi(;
                orth = orth, krylovdim = n, maxiter = 1, tol = tolerance(T),
                verbosity = WARN_LEVEL
            )
            @test_logs eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1, :SR, alg)
            alg = Arnoldi(;
                orth = orth, krylovdim = n1 + 2, maxiter = 1, tol = tolerance(T),
                verbosity = WARN_LEVEL
            )
            @test_logs (:warn,) eigsolve(
                wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                :SR, alg
            )
            alg = Arnoldi(;
                orth = orth, krylovdim = n, maxiter = 1, tol = tolerance(T),
                verbosity = STARTSTOP_LEVEL
            )
            @test_logs (:info,) eigsolve(
                wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                :SR, alg
            )
            alg = Arnoldi(;
                orth = orth, krylovdim = n1, maxiter = 3, tol = tolerance(T),
                verbosity = EACHITERATION_LEVEL
            )
            @test_logs(
                (:info,), (:info,), (:info,), (:warn,),
                eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg)
            )
            alg = Arnoldi(;
                orth = orth, krylovdim = 4, maxiter = 1, tol = tolerance(T),
                verbosity = EACHITERATION_LEVEL + 1
            )
            # since it is impossible to know exactly the size of the Krylov subspace after shrinking,
            # we only know the output for a sigle iteration
            @test_logs(
                (:info,), (:info,), (:info,), (:info,), (:info,), (:warn,),
                eigsolve(wrapop(A, Val(mode)), wrapvec(v, Val(mode)), 1, :SR, alg)
            )

            @test KrylovKit.eigselector(
                wrapop(A, Val(mode)), eltype(v); orth = orth,
                krylovdim = n, maxiter = 1,
                tol = tolerance(T)
            ) isa Arnoldi
            n2 = n - n1
            alg = Arnoldi(; orth = orth, krylovdim = 2 * n, maxiter = 1, tol = tolerance(T))
            D2, V2, info2 = @constinferred eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n2, :LR, alg
            )
            D = sort(sort(eigvals(A); by = imag, rev = true); alg = MergeSort, by = real)
            D2′ = sort(sort(D2; by = imag, rev = true); alg = MergeSort, by = real)
            @test vcat(D1[1:n1], D2′[(end - n2 + 1):end]) ≈ D

            U1 = stack(unwrapvec, V1)
            U2 = stack(unwrapvec, V2)
            @test A * U1 ≈ U1 * Diagonal(D1)
            @test A * U2 ≈ U2 * Diagonal(D2)

            if T <: Complex
                n1 = div(n, 2)
                D1, V1, info = eigsolve(
                    wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n1,
                    :SI,
                    alg
                )
                n2 = n - n1
                D2, V2, info = eigsolve(
                    wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n2,
                    :LI,
                    alg
                )
                D = sort(eigvals(A); by = imag)

                @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ D

                U1 = stack(unwrapvec, V1)
                U2 = stack(unwrapvec, V2)
                @test A * U1 ≈ U1 * Diagonal(D1)
                @test A * U2 ≈ U2 * Diagonal(D2)
            end

            alg = Arnoldi(;
                orth = orth, krylovdim = 2n, maxiter = 1, tol = tolerance(T),
                verbosity = WARN_LEVEL
            )
            @test_logs (:warn,) (:warn,) eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n + 1, :LM, alg
            )
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
            alg = Arnoldi(;
                krylovdim = 3 * n, maxiter = 20,
                tol = tolerance(T), eager = true, verbosity = SILENT_LEVEL
            )
            D1, V1, info1 = @constinferred eigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n, :SR, alg
            )
            D2, V2, info2 = eigsolve(
                wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n, :LR,
                alg
            )
            D3, V3, info3 = eigsolve(
                wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n, :LM,
                alg
            )
            D = sort(eigvals(A); by = imag, rev = true)

            l1 = info1.converged
            l2 = info2.converged
            l3 = info3.converged
            @test l1 > 0
            @test l2 > 0
            @test l3 > 0
            @test D1[1:l1] ≊ sort(D; alg = MergeSort, by = real)[1:l1]
            @test D2[1:l2] ≊ sort(D; alg = MergeSort, by = real, rev = true)[1:l2]
            # sorting by abs does not seem very reliable if two distinct eigenvalues are close
            # in absolute value, so we perform a second sort afterwards using the real part
            @test D3[1:l3] ≊ sort(D; by = abs, rev = true)[1:l3]

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
                D1, V1, info1 = eigsolve(
                    wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                    :SI, alg
                )
                D2, V2, info2 = eigsolve(
                    wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                    :LI, alg
                )
                D = eigvals(A)

                l1 = info1.converged
                l2 = info2.converged
                @test l1 > 0
                @test l2 > 0
                @test D1[1:l1] ≈ sort(D; by = imag)[1:l1]
                @test D2[1:l2] ≈ sort(D; by = imag, rev = true)[1:l2]

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
            alg = Arnoldi(;
                krylovdim = 3 * n, maxiter = 20,
                tol = tolerance(T), eager = true, verbosity = SILENT_LEVEL
            )
            D1, V1, info1 = @constinferred realeigsolve(
                wrapop(A, Val(mode)),
                wrapvec(v, Val(mode)), n, :SR, alg
            )
            D2, V2, info2 = realeigsolve(
                wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                :LR,
                alg
            )
            D3, V3, info3 = realeigsolve(
                wrapop(A, Val(mode)), wrapvec(v, Val(mode)), n,
                :LM,
                alg
            )
            l1 = info1.converged
            l2 = info2.converged
            l3 = info3.converged
            @test l1 > 0
            @test l2 > 0
            @test l3 > 0
            @test D1[1:l1] ≊ sort(D; alg = MergeSort)[1:l1]
            @test D2[1:l2] ≊ sort(D; alg = MergeSort, rev = true)[1:l2]
            # sorting by abs does not seem very reliable if two distinct eigenvalues are close
            # in absolute value, so we perform a second sort afterwards using the real part
            @test D3[1:l3] ≊ sort(D; by = abs, rev = true)[1:l3]

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
                alg = Arnoldi(;
                    krylovdim = 3 * n, maxiter = 20,
                    tol = tolerance(T), eager = true, verbosity = SILENT_LEVEL
                )
                D1, V1, info1 = @constinferred realeigsolve(f, v, n, :SR, alg)
                D2, V2, info2 = realeigsolve(f, v, n, :LR, alg)
                D3, V3, info3 = realeigsolve(f, v, n, :LM, alg)

                l1 = info1.converged
                l2 = info2.converged
                l3 = info3.converged
                @test l1 > 0
                @test l2 > 0
                @test l3 > 0
                @test D1[1:l1] ≊ sort(D; alg = MergeSort)[1:l1]
                @test D2[1:l2] ≊ sort(D; alg = MergeSort, rev = true)[1:l2]
                # sorting by abs does not seem very reliable if two distinct eigenvalues are close
                # in absolute value, so we perform a second sort afterwards using the real part
                @test D3[1:l3] ≊ sort(D; by = abs, rev = true)[1:l3]

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
    A[2, 1] = 1.0e-9
    A[1, 2] = -1.0e-9
    v = ones(Float64, size(A, 1))
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol = 1.0e-8, verbosity = SILENT_LEVEL))
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol = 1.0e-8, verbosity = WARN_LEVEL))
    @test_logs (:info,) realeigsolve(
        A, v, 1, :LM,
        Arnoldi(; tol = 1.0e-8, verbosity = STARTSTOP_LEVEL)
    )
    @test_logs (:warn,) realeigsolve(
        A, v, 1, :LM,
        Arnoldi(; tol = 1.0e-10, verbosity = WARN_LEVEL)
    )
    @test_logs (:warn,) (:info,) realeigsolve(
        A, v, 1, :LM,
        Arnoldi(;
            tol = 1.0e-10,
            verbosity = STARTSTOP_LEVEL
        )
    )

    # this should not trigger a warning
    A[1, 2] = A[2, 1] = 0
    A[1, 1] = 1
    A[2, 2] = A[3, 3] = 0.99
    A[3, 2] = 1.0e-6
    A[2, 3] = -1.0e-6
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol = 1.0e-12, verbosity = SILENT_LEVEL))
    @test_logs realeigsolve(A, v, 1, :LM, Arnoldi(; tol = 1.0e-12, verbosity = WARN_LEVEL))
    @test_logs (:info,) realeigsolve(
        A, v, 1, :LM,
        Arnoldi(; tol = 1.0e-12, verbosity = STARTSTOP_LEVEL)
    )
end

@testset "BlockLanczos - eigsolve for large sparse matrix and map input" begin
    # There are 2 * m * n qubits and I choose to enumerate all the horizontal ones from 1:m*n
    # and all the vertical ones as (m*n+1):(2*m*n)
    function toric_code_strings(m::Int, n::Int)
        li = LinearIndices((m, n))
        bottom(i, j) = li[mod1(i, m), mod1(j, n)] + m * n
        right(i, j) = li[mod1(i, m), mod1(j, n)]
        xstrings = NTuple{4, Int}[]
        zstrings = NTuple{4, Int}[]
        for i in 1:m, j in 1:n
            # plaquette
            push!(xstrings, (bottom(i, j + 1), right(i, j), bottom(i, j), right(i - 1, j)))
            # vertex
            push!(zstrings, (right(i, j), bottom(i, j), right(i, j - 1), bottom(i + 1, j)))
        end
        return xstrings, zstrings
    end

    function pauli_kron(n::Int, ops::Pair{Int, Char}...)
        mat = sparse(1.0I, 2^n, 2^n)
        for (pos, op) in ops
            if op == 'X'
                σ = sparse([0 1; 1 0])
            elseif op == 'Y'
                σ = sparse([0 -im; im 0])
            elseif op == 'Z'
                σ = sparse([1 0; 0 -1])
            elseif op == 'I'
                σ = sparse(1.0I, 2, 2)
            else
                error("Unknown Pauli operator $op")
            end

            left = sparse(1.0I, 2^(pos - 1), 2^(pos - 1))
            right = sparse(1.0I, 2^(n - pos), 2^(n - pos))
            mat = kron(left, kron(σ, right)) * mat
        end
        return mat
    end

    # define the function to construct the Hamiltonian matrix
    function toric_code_hamiltonian_matrix(m::Int, n::Int)
        xstrings, zstrings = toric_code_strings(m, n)
        N = 2 * m * n  # total number of qubits

        # initialize the Hamiltonian matrix as a zero matrix
        H = spzeros(2^N, 2^N)

        # add the X-type operator terms
        for xs in xstrings[1:(end - 1)]
            H += pauli_kron(N, (i => 'X' for i in xs)...)
        end

        for zs in zstrings[1:(end - 1)]
            H += pauli_kron(N, (i => 'Z' for i in zs)...)
        end

        return H
    end

    sites_num = 3
    p = 5 # block size
    M = 2^(2 * sites_num^2)
    x₀ = Block([rand(M) for _ in 1:p])
    get_value_num = 10
    tol = 1.0e-6
    h_mat = toric_code_hamiltonian_matrix(sites_num, sites_num)

    # matrix input
    alg = BlockLanczos(; tol = tol, maxiter = 1)
    D, U, info = eigsolve(-h_mat, x₀, get_value_num, :SR, alg)
    @test count(x -> abs(x + 16.0) < 2.0 - tol, D[1:get_value_num]) == 4
    @test count(x -> abs(x + 16.0) < tol, D[1:get_value_num]) == 4

    # map input
    D, U, info = eigsolve(x -> -h_mat * x, x₀, get_value_num, :SR, alg)
    @test count(x -> abs(x + 16.0) < 1.9, D[1:get_value_num]) == 4
    @test count(x -> abs(x + 16.0) < 1.0e-8, D[1:get_value_num]) == 4
end

# For user interface, input is a block.
@testset "BlockLanczos - eigsolve full $mode" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        block_size = 2
        A = mat_with_eigrepition(T, n, block_size)
        x₀ = Block([wrapvec(rand(T, n), Val(mode)) for _ in 1:block_size])
        n1 = div(n, 2)  # eigenvalues to solve
        eigvalsA = eigvals(A)
        alg = BlockLanczos(;
            krylovdim = n, maxiter = 1, tol = tolerance(T),
            verbosity = STARTSTOP_LEVEL
        )
        D1, V1, info = @test_logs (:info,) eigsolve(
            wrapop(A, Val(mode)),
            x₀, n1, :SR, alg
        )
        alg = BlockLanczos(;
            krylovdim = n, maxiter = 1, tol = tolerance(T),
            verbosity = WARN_LEVEL
        )
        @test_logs eigsolve(wrapop(A, Val(mode)), x₀, n1, :SR, alg)
        alg = BlockLanczos(;
            krylovdim = n1 + 1, maxiter = 1, tol = tolerance(T),
            verbosity = WARN_LEVEL
        )
        @test_logs (:warn,) eigsolve(wrapop(A, Val(mode)), x₀, n1, :SR, alg)
        alg = BlockLanczos(;
            krylovdim = n, maxiter = 1, tol = tolerance(T),
            verbosity = STARTSTOP_LEVEL
        )
        @test_logs (:info,) eigsolve(wrapop(A, Val(mode)), x₀, n1, :SR, alg)
        alg = BlockLanczos(;
            krylovdim = 3, maxiter = 3, tol = tolerance(T),
            verbosity = EACHITERATION_LEVEL
        )
        @test_logs(
            (:info,), (:info,), (:info,), (:warn,),
            eigsolve(wrapop(A, Val(mode)), x₀, 1, :SR, alg)
        )
        alg = BlockLanczos(;
            krylovdim = 4, maxiter = 1, tol = tolerance(T),
            verbosity = EACHITERATION_LEVEL + 1
        )
        @test_logs(
            (:info,), (:info,), (:info,), (:warn,),
            eigsolve(wrapop(A, Val(mode)), x₀, 1, :SR, alg)
        )
        n2 = n - n1
        alg = BlockLanczos(; krylovdim = 2 * n, maxiter = 4, tol = tolerance(T))
        D2, V2, info = eigsolve(wrapop(A, Val(mode)), x₀, n2, :LR, alg)
        @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ eigvalsA

        U1 = hcat(unwrapvec.(V1)...)
        U2 = hcat(unwrapvec.(V2)...)

        @test U1' * U1 ≈ I
        @test U2' * U2 ≈ I

        @test (x -> KrylovKit.apply(A, x)).(unwrapvec.(V1)) ≈ D1 .* unwrapvec.(V1)
        @test (x -> KrylovKit.apply(A, x)).(unwrapvec.(V2)) ≈ D2 .* unwrapvec.(V2)

        alg = BlockLanczos(;
            krylovdim = 2n, maxiter = 1, tol = tolerance(T),
            verbosity = WARN_LEVEL
        )
        @test_logs (:warn,) (:warn,) eigsolve(wrapop(A, Val(mode)), x₀, n + 1, :LM, alg)
    end
end

@testset "BlockLanczos - eigsolve iteratively $mode" for mode in
    (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        block_size = 4
        A = mat_with_eigrepition(T, N, block_size)
        x₀ = Block([wrapvec(rand(T, N), Val(mode)) for _ in 1:block_size])
        eigvalsA = eigvals(A)

        alg = BlockLanczos(;
            krylovdim = N, maxiter = 10, tol = tolerance(T),
            eager = true, verbosity = SILENT_LEVEL
        )
        D1, V1, info1 = eigsolve(wrapop(A, Val(mode)), x₀, n, :SR, alg)
        D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), x₀, n, :LR, alg)

        l1 = info1.converged
        l2 = info2.converged

        @test l1 > 0
        @test l2 > 0
        @test D1[1:n] ≈ eigvalsA[1:n]
        @test D2[1:n] ≈ eigvalsA[N:-1:(N - n + 1)]

        U1 = hcat(unwrapvec.(V1[1:l1])...)
        U2 = hcat(unwrapvec.(V2[1:l2])...)
        R1 = hcat(unwrapvec.(info1.residual[1:l1])...)
        R2 = hcat(unwrapvec.(info2.residual[1:l2])...)

        @test U1' * U1 ≈ I
        @test U2' * U2 ≈ I
        @test hcat([KrylovKit.apply(A, U1[:, i]) for i in 1:l1]...) ≈
            U1 * Diagonal(D1) + R1
        @test hcat([KrylovKit.apply(A, U2[:, i]) for i in 1:l2]...) ≈
            U2 * Diagonal(D2) + R2
    end
end

@testset "BlockLanczos - eigsolve for abstract type" begin
    T = ComplexF64
    block_size = 2
    H = mat_with_eigrepition(T, n, block_size)
    H = H' * H + I
    eig_num = 2
    Hip(x::Vector, y::Vector) = x' * H * y
    x₀ = Block([InnerProductVec(rand(T, n), Hip) for _ in 1:block_size])
    Aip(x::InnerProductVec) = InnerProductVec(H * x.vec, Hip)
    D, V, info = eigsolve(
        Aip, x₀, eig_num, :SR,
        BlockLanczos(;
            krylovdim = n, maxiter = 1, tol = tolerance(T),
            verbosity = SILENT_LEVEL
        )
    )
    D_true = eigvals(H)
    BlockV = KrylovKit.Block(V)
    @test D[1:eig_num] ≈ D_true[1:eig_num]
    @test KrylovKit.block_inner(BlockV, BlockV) ≈ I
    @test findmax([norm(Aip(V[i]) - D[i] * V[i]) for i in 1:eig_num])[1] < tolerance(T)
end

# with the same krylovdim, BlockLanczos has lower accuracy with the size of block larger than 1.
@testset "Compare Lanczos and BlockLanczos $mode" for mode in
    (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        A = rand(T, (2N, 2N)) .- one(T) / 2
        A = (A + A') / 2
        block_size = 1
        x₀_block = Block([wrapvec(rand(T, 2N), Val(mode)) for _ in 1:block_size])
        x₀_lanczos = x₀_block[1]
        alg1 = Lanczos(;
            krylovdim = 2n, maxiter = 10, tol = tolerance(T),
            verbosity = SILENT_LEVEL
        )
        alg2 = BlockLanczos(;
            krylovdim = 2n, maxiter = 10, tol = tolerance(T),
            verbosity = SILENT_LEVEL
        )
        evals1, _, info1 = eigsolve(wrapop(A, Val(mode)), x₀_lanczos, n, :SR, alg1)
        evals2, _, info2 = eigsolve(wrapop(A, Val(mode)), x₀_block, n, :SR, alg2)
        @test info1.converged == info2.converged
        @test info1.numiter == info2.numiter
        @test info1.numops + 1 == info2.numops # one extra operation for the BlockLanczos initialization
        @test isapprox(info1.normres, info2.normres, atol = tolerance(T))
        @test isapprox(
            unwrapvec.(info1.residual[1:(info1.converged)])[2],
            unwrapvec.(info2.residual[1:(info2.converged)])[2];
            atol = tolerance(T)
        )
        @test isapprox(
            evals1[1:(info1.converged)], evals2[1:(info2.converged)];
            atol = tolerance(T)
        )
    end
    @testset for T in (Float64, ComplexF64)
        A = rand(T, (2N, 2N)) .- one(T) / 2
        A = (A + A') / 2
        block_size = 2
        x₀_block = Block([wrapvec(rand(T, 2N), Val(mode)) for _ in 1:block_size])
        x₀ = rand(T, 2N) # Lanczos input
        x₁ = A^n * x₀ # effectively build the same Krylov subspace in `BlockLanczos`
        x₀_lanczos = wrapvec(x₀, Val(mode))
        x₀_block = Block([wrapvec(x₀, Val(mode)), wrapvec(x₁, Val(mode))])
        alg1 = Lanczos(;
            krylovdim = 2n, maxiter = 1, tol = tolerance(T),
            verbosity = SILENT_LEVEL
        )
        alg2 = BlockLanczos(;
            krylovdim = 2n, maxiter = 1, tol = tolerance(T),
            verbosity = SILENT_LEVEL
        )
        vals1, vecs1, info1 = eigsolve(wrapop(A, Val(mode)), x₀_lanczos, n, :SR, alg1)
        vals2, vecs2, info2 = eigsolve(wrapop(A, Val(mode)), x₀_block, n, :SR, alg2)
        @test vals1 ≈ vals2
        @test all(abs.(inner.(vecs1, vecs2)) .≈ 1)
        @test info1.normres ≈ info2.normres
    end
end

# Test effectiveness of shrink!() in BlockLanczos
@testset "Test effectiveness of shrink!() in BlockLanczos $mode" for mode in
    (
        :vector, :inplace,
        :outplace,
    )
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        block_size = 5
        A = mat_with_eigrepition(T, N, block_size)
        x₀ = Block([wrapvec(rand(T, N), Val(mode)) for _ in 1:block_size])
        values0 = eigvals(A)[1:n]
        n1 = n ÷ 2
        alg = BlockLanczos(;
            krylovdim = 3 * n ÷ 2, maxiter = 1, tol = 1.0e-12,
            verbosity = SILENT_LEVEL
        )
        values, _, _ = eigsolve(wrapop(A, Val(mode)), x₀, n, :SR, alg)
        error1 = norm(values[1:n1] - values0[1:n1])
        alg_shrink = BlockLanczos(;
            krylovdim = 3 * n ÷ 2, maxiter = 2, tol = 1.0e-12,
            verbosity = SILENT_LEVEL
        )
        values_shrink, _, _ = eigsolve(wrapop(A, Val(mode)), x₀, n, :SR, alg_shrink)
        error2 = norm(values_shrink[1:n1] - values0[1:n1])
        @test error2 < error1
    end
end

@testset "BlockLanczos - eigsolve without alg $mode" for mode in
    (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
        (ComplexF64,)
    @testset for T in scalartypes
        block_size = 2
        A = mat_with_eigrepition(T, n, block_size)
        x₀ = Block([wrapvec(rand(T, n), Val(mode)) for _ in 1:block_size])
        if mode === :vector
            D1, V1, info1 = eigsolve(wrapop(A, Val(mode)), x₀, 1, :SR)
            @test info1.converged >= 1
            eigA = eigvals(A)
            @test D1[1] ≈ eigA[1]
            D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), x₀)
            @test info1.converged >= 1
            @test D2[1] ≈ (abs(eigA[1]) > abs(eigA[end]) ? eigA[1] : eigA[end])
            @test_throws ErrorException eigsolve(wrapop(A, Val(mode)), x₀, 1, :LI)
            B = copy(A)
            B[1, 2] += T(1) # error for non-symmetric/hermitian operator
            @test_throws ErrorException eigsolve(wrapop(B, Val(mode)), x₀, 1, :SR)
        else
            @test_throws ErrorException eigsolve(wrapop(A, Val(mode)), x₀, 1, :SR)
        end
    end
end
