# TODO: Remove RecursiveVec and simply use tuple when RecursiveVec is removed.
@testset "RecursiveVec - singular values full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (n, n))
            v = rand(T, (n,))
            v2 = RecursiveVec(v, zero(v))
            alg = Lanczos(; orth=orth, krylovdim=2 * n, maxiter=1, tol=tolerance(T))
            D, V, info = eigsolve(v2, n, :LR, alg) do x
                x1, x2 = x
                y1 = A * x2
                y2 = A' * x1
                return RecursiveVec(y1, y2)
            end
            @test info.converged >= n
            S = D[1:n]
            @test S ≈ svdvals(A)
            UU = hcat((sqrt(2 * one(T)) * v[1] for v in V[1:n])...)
            VV = hcat((sqrt(2 * one(T)) * v[2] for v in V[1:n])...)
            @test UU * Diagonal(S) * VV' ≈ A
        end
    end
end

@testset "RecursiveVec - singular values iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T, (N, 2 * N))
            v = rand(T, (N,))
            w = rand(T, (2 * N,))
            v2 = RecursiveVec(v, w)
            alg = Lanczos(; orth=orth, krylovdim=n, maxiter=300, tol=tolerance(T))
            n1 = div(n, 2)
            D, V, info = eigsolve(v2, n1, :LR, alg) do x
                x1, x2 = x
                y1 = A * x2
                y2 = A' * x1
                return RecursiveVec(y1, y2)
            end
            @test info.converged >= n1
            S = D[1:n1]
            @test S ≈ svdvals(A)[1:n1]
            UU = hcat((sqrt(2 * one(T)) * v[1] for v in V[1:n1])...)
            VV = hcat((sqrt(2 * one(T)) * v[2] for v in V[1:n1])...)
            @test Diagonal(S) ≈ UU' * A * VV
        end
    end
end
