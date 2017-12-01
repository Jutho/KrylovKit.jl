# Test lanczos exponentiate full
@testset "Lanczos - Exponentiate full" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            A = (A+A')/2
            V = one(A)
            W = zero(A)
            alg = Lanczos(orth; krylovdim = n, maxiter = 1, tol = 10*n*eps(real(T)))
            for k = 1:n
                W[:,k], =  @inferred exponentiate(1, A, view(V,:,k), alg)
            end
            if VERSION <= v"0.6.99"
                @test W ≈ expm(A)
            else
                @test W ≈ exp(A)
            end
        end
    end
end

# Test exponentiate
@testset "Lanczos - Exponentiate iteratively" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            A = (A+A')/2
            v = rand(T,N)
            t = rand(complex(T))
            alg = Lanczos(orth; krylovdim = n, maxiter = 30, tol = 10*N*eps(real(T))*abs(t))
            w, info =  @inferred exponentiate(t, A, v, alg)
            @test info.converged > 0
            if VERSION <= v"0.6.99"
                @test w ≈ expm(t*A)*v
            else
                @test w ≈ exp(t*A)*v
            end
        end
    end
end
