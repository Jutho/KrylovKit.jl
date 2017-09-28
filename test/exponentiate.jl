# Test lanczos exponentiate full
@testset "Lanczos - Expontiate full" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(n,n)) .- one(T)/2
            A = (A+A')/2
            V = one(A)
            W = zero(A)
            alg = Lanczos(ExplicitRestart(1), orth; krylovdim=n, tol = 10*n*eps(real(T)))
            for k = 1:n
                W[:,k], = exponentiate(1, A, view(V,:,k), alg)
            end
            @test W ≈ expm(A)
        end
    end
end

# Test exponentiate
@testset "Lanczos Expontiate" begin
    @testset for T in (Float32, Float64, Complex64, Complex128)
        @testset for orth in (cgs2, mgs2, cgsr, mgsr)
            A = rand(T,(N,N)) .- one(T)/2
            A = (A+A')/2
            v = rand(T,N)
            t = rand(complex(T))
            alg = Lanczos(ExplicitRestart(30), orth; krylovdim=n, tol = 10*N*eps(real(T))*abs(t))
            w, info = exponentiate(t, A, v, alg)
            @test info.converged > 0
            @test w ≈ expm(t*A)*v
        end
    end
end
