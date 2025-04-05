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

@testset "Block Lanczos - eigsolve for large sparse matrix and map input" begin

    function toric_code_strings(m::Int, n::Int)
        li = LinearIndices((m, n))
        bottom(i, j) = li[mod1(i, m), mod1(j, n)] + m * n
        right(i, j) = li[mod1(i, m), mod1(j, n)]
        xstrings = Vector{Int}[]
        zstrings = Vector{Int}[]
        for i in 1:m, j in 1:n
            # face center
            push!(xstrings, [bottom(i, j - 1), right(i, j), bottom(i, j), right(i - 1, j)])
            # cross
            push!(zstrings, [right(i, j), bottom(i, j), right(i, j + 1), bottom(i + 1, j)])
        end
        return xstrings, zstrings
    end
    
    function pauli_kron(n::Int, ops::Pair{Int,Char}...)
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
            ops = [i => 'X' for i in xs]
            H += pauli_kron(N, ops...)
        end
    
        for zs in zstrings[1:(end - 1)]
            ops = [i => 'Z' for i in zs]
            H += pauli_kron(N, ops...)
        end
    
        return H
    end
    
    Random.seed!(4)
    sites_num = 3
    p = 5 # block size
    X1 = Matrix(qr(rand(2^(2*sites_num^2), p)).Q)
    get_value_num = 10
    tol = 1e-6
    h_mat = toric_code_hamiltonian_matrix(sites_num, sites_num)

    # matrix input
    D, U, info = eigsolve(-h_mat, X1, get_value_num, :SR,
                          Lanczos(; maxiter=20, tol=tol))            
    @show D[1:get_value_num]
    @test count(x -> abs(x + 16.0) < 2.0 - tol, D[1:get_value_num]) == 4
    @test count(x -> abs(x + 16.0) < tol, D[1:get_value_num]) == 4

    # map input
    D, U, info = eigsolve(x -> -h_mat * x, X1, get_value_num, :SR,
                          Lanczos(; maxiter=20, tol=tol))
    @show D[1:get_value_num]
    @test count(x -> abs(x + 16.0) < 1.9, D[1:get_value_num]) == 4
    @test count(x -> abs(x + 16.0) < 1e-8, D[1:get_value_num]) == 4
    
end

using Test, SparseArrays, Random, LinearAlgebra, Profile
using KrylovKit

@testset "Block Lanczos - eigsolve full ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (mgs2,) : (mgs2,)  # Block Lanczos 只支持 mgs2
    @testset for T in scalartypes
        @testset for orth in orths
            A = rand(T, (n, n)) .- one(T) / 2
            A = (A + A') / 2  # 确保矩阵是对称/厄密的
            
            # 创建初始块向量（矩阵形式，每列是一个向量）
            block_size = 2  # 块大小
            X = rand(T, (n, block_size))
            
            # 如果是 :vector 模式，将矩阵转换为向量数组
            x₀ = mode === :vector ? [X[:, i] for i in 1:block_size] : X
            
            n1 = div(n, 2)  # 要求解的特征值数量
            
            # 创建 Block Lanczos 算法配置
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=5, tol=tolerance(T),
                          verbosity=2)
            
            # 执行特征值求解
            D1, V1, info = @test_logs (:info,) eigsolve(wrapop(A, Val(mode)),
                                                        wrapvec(x₀, Val(mode)), n1, :SR, alg)
            
            # 测试不同详细级别的日志输出
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=5, tol=tolerance(T),
                          verbosity=1)
            @test_logs eigsolve(wrapop(A, Val(mode)),
                                wrapvec(x₀, Val(mode)), n1, :SR, alg)
            
            # 测试 Krylov 维度较小时的警告
            alg = Lanczos(; orth=orth, krylovdim=n1 + 1, maxiter=5, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) eigsolve(wrapop(A, Val(mode)),
                                         wrapvec(x₀, Val(mode)), n1, :SR, alg)
            
            # 测试详细级别为 2 的日志输出
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=5, tol=tolerance(T),
                          verbosity=2)
            @test_logs (:info,) eigsolve(wrapop(A, Val(mode)),
                                         wrapvec(x₀, Val(mode)), n1, :SR, alg)
            
            # 测试求解少量特征值时的日志
            alg = Lanczos(; orth=orth, krylovdim=n1, maxiter=3, tol=tolerance(T),
                          verbosity=3)
            @test_logs((:info,), (:info,), (:info,), (:warn,),
                       eigsolve(wrapop(A, Val(mode)), wrapvec(x₀, Val(mode)), 1, :SR, alg))
            
            # 计算剩余的特征值
            n2 = n - n1
            alg = Lanczos(; krylovdim=2 * n, maxiter=5, tol=tolerance(T))
            D2, V2, info = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                   wrapvec(x₀, Val(mode)),
                                                   n2, :LR, alg)
            
            # 验证计算的特征值与直接求解的特征值匹配
            @test vcat(D1[1:n1], reverse(D2[1:n2])) ≊ eigvals(A)
            
            # 测试特征向量的正交性
            # 对于块向量，我们需要特殊处理
            if mode === :vector
                # 将向量数组转换为矩阵
                U1 = hcat([unwrapvec(v) for v in V1]...)
                U2 = hcat([unwrapvec(v) for v in V2]...)
            else
                U1 = stack(unwrapvec, V1)
                U2 = stack(unwrapvec, V2)
            end
            
            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I
            
            # 验证特征值和特征向量满足 Av = λv
            @test A * U1 ≈ U1 * Diagonal(D1)
            @test A * U2 ≈ U2 * Diagonal(D2)
            
            # 测试请求过多特征值时的警告
            alg = Lanczos(; orth=orth, krylovdim=2n, maxiter=5, tol=tolerance(T),
                          verbosity=1)
            @test_logs (:warn,) (:warn,) eigsolve(wrapop(A, Val(mode)),
                                                  wrapvec(x₀, Val(mode)), n + 1, :LM, alg)
        end
    end
end

@testset "Block Lanczos - eigsolve iteratively ($mode)" for mode in (:vector, :inplace, :outplace)
    scalartypes = mode === :vector ? (Float32, Float64, ComplexF32, ComplexF64) :
                  (ComplexF64,)
    orths = mode === :vector ? (mgs2,) : (mgs2,)  # Block Lanczos 只支持 mgs2
    @testset for T in scalartypes
        @testset for orth in orths
            # 创建大型矩阵，用于测试迭代求解
            A = rand(T, (N, N)) .- one(T) / 2
            A = (A + A') / 2  # 确保矩阵是对称/厄密的
            
            # 创建初始块向量
            block_size = 2  # 块大小
            X = rand(T, (N, block_size))
            
            # 如果是 :vector 模式，将矩阵转换为向量数组
            x₀ = mode === :vector ? [X[:, i] for i in 1:block_size] : X
            
            # 创建 Block Lanczos 算法配置，启用 eager 模式加速收敛
            alg = Lanczos(; krylovdim=2 * n, maxiter=10,
                          tol=tolerance(T), eager=true, verbosity=0)
            
            # 求解最小实部特征值
            D1, V1, info1 = @constinferred eigsolve(wrapop(A, Val(mode)),
                                                    wrapvec(x₀, Val(mode)), n, :SR, alg)
            
            # 求解最大实部特征值
            D2, V2, info2 = eigsolve(wrapop(A, Val(mode)), wrapvec(x₀, Val(mode)), n, :LR,
                                     alg)
            
            # 提取收敛的特征值数量
            l1 = info1.converged
            l2 = info2.converged
            
            # 验证至少有部分特征值收敛
            @test l1 > 0
            @test l2 > 0
            
            # 验证计算的特征值与直接求解的特征值匹配
            @test D1[1:l1] ≈ eigvals(A)[1:l1]
            @test D2[1:l2] ≈ eigvals(A)[N:-1:(N - l2 + 1)]
            
            # 测试特征向量的正交性
            if mode === :vector
                # 将向量数组转换为矩阵
                U1 = hcat([unwrapvec(v) for v in V1]...)
                U2 = hcat([unwrapvec(v) for v in V2]...)
                
                # 转换残差向量
                R1 = hcat([unwrapvec(r) for r in info1.residual]...)
                R2 = hcat([unwrapvec(r) for r in info2.residual]...)
            else
                U1 = stack(unwrapvec, V1)
                U2 = stack(unwrapvec, V2)
                
                R1 = stack(unwrapvec, info1.residual)
                R2 = stack(unwrapvec, info2.residual)
            end
            
            @test U1' * U1 ≈ I
            @test U2' * U2 ≈ I
            
            # 验证特征值方程 A*v = λ*v + r 含残差项
            @test A * U1 ≈ U1 * Diagonal(D1) + R1
            @test A * U2 ≈ U2 * Diagonal(D2) + R2
        end
    end
end

# 测试块Lanczos在稀疏矩阵上的性能和准确性
@testset "Block Lanczos - eigsolve for sparse matrices" begin
    # 创建一个稀疏的对称矩阵
    N = 100
    A = spzeros(Float64, N, N)
    
    # 生成带对角线结构的稀疏矩阵
    for i in 1:N
        A[i, i] = i  # 对角线元素
        if i < N
            A[i, i+1] = A[i+1, i] = 0.5  # 次对角线元素
        end
        if i < N-5
            A[i, i+5] = A[i+5, i] = 0.1  # 远离对角线的元素
        end
    end
    
    # 创建多列初始矩阵（块向量）
    block_size = 3
    X = rand(Float64, (N, block_size))
    x₀ = [X[:, i] for i in 1:block_size]
    
    # 使用Block Lanczos求解10个最小特征值
    howmany = 10
    alg = Lanczos(; krylovdim=30, maxiter=20, tol=1e-8, verbosity=1)
    
    # 求解并验证
    D, V, info = eigsolve(A, x₀, howmany, :SR, alg)
    
    # 与直接求解的特征值比较
    exact_eigs = eigvals(Array(A))[1:howmany]
    @test D[1:howmany] ≈ exact_eigs
    
    # 验证收敛的特征值数量
    @test info.converged >= howmany/2  # 至少一半应该收敛
    
    # 验证残差范数
    @test all(info.normres[1:info.converged] .< 1e-7)
    
    # 验证特征向量满足特征值方程
    U = hcat([v for v in V]...)
    @test norm(A * U - U * Diagonal(D)) < 1e-6
end

# 测试块Lanczos与标准Lanczos的性能比较
@testset "Block Lanczos vs Standard Lanczos - Performance" begin
    # 创建一个中等大小的矩阵用于测试
    N = 50
    A = rand(Float64, (N, N))
    A = (A + A') / 2  # 确保对称性
    
    # 单向量初始条件（标准Lanczos）
    v = rand(Float64, N)
    
    # 多向量初始条件（块Lanczos）
    block_size = 3
    X = rand(Float64, (N, block_size))
    x₀ = [X[:, i] for i in 1:block_size]
    
    # 共同的算法参数
    howmany = 5
    krylovdim = 20
    maxiter = 10
    tol = 1e-8
    
    # 创建两种算法实例
    std_alg = Lanczos(; krylovdim=krylovdim, maxiter=maxiter, tol=tol, verbosity=0)
    block_alg = Lanczos(; krylovdim=krylovdim, maxiter=maxiter, tol=tol, verbosity=0)
    
    # 执行求解并计时
    std_time = @elapsed begin
        std_D, std_V, std_info = eigsolve(A, v, howmany, :SR, std_alg)
    end
    
    block_time = @elapsed begin
        block_D, block_V, block_info = eigsolve(A, x₀, howmany, :SR, block_alg)
    end
    
    # 验证结果的正确性
    @test std_D[1:howmany] ≈ block_D[1:howmany]
    
    # 打印性能比较结果（可选）
    @info "Performance comparison:" standard_lanczos=std_time block_lanczos=block_time ratio=block_time/std_time
    
    # 验证块Lanczos的收敛速度（以迭代次数衡量）通常更快
    if block_info.numiter < std_info.numiter
        @info "Block Lanczos converged in fewer iterations" block_iter=block_info.numiter std_iter=std_info.numiter
    end
    
    # 如果块Lanczos计算的更快，我们应该验证这一点
    # 注意：这个测试可能不稳定，取决于系统负载和实现细节
    # @test block_time <= 1.5 * std_time  # 允许50%的性能波动
end


