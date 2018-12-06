@testset "Lanczos - exponentiate full" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            A = (A+A')/2
            V = one(A)
            W = zero(A)
            alg = Lanczos(orth = orth, krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            for k = 1:n
                W[:,k], =  @inferred exponentiate(A, 1, view(V,:,k), alg)
            end
            @test W ≈ exp(A)
        end
    end
end

@testset "Lanczos - exponentiate iteratively" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            A = (A+A')/2
            v = rand(T,N)
            t = rand(complex(T))
            alg = Lanczos(orth = orth, krylovdim = n, maxiter = 30, tol = 10*N*eps(real(T))*abs(t))
            w, info =  @inferred exponentiate(A, t, v, alg)
            @test info.converged > 0
            @test w ≈ exp(t*A)*v
        end
    end
end
