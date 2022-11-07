
using Zygote, FiniteDifferences

@testset "Small linsolve AD test" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = 2 * (rand(T, (n, n)) .- one(T) / 2)
        Avec, A_fromvec = to_vec(A)

        b = 2 * (rand(T, n) .- one(T) / 2)
        b /= norm(b)
        bvec, b_fromvec = to_vec(b)
        tol = cond(A) * eps(real(T))

        function f(Av, bv)
            A′ = wrapop(A_fromvec(Av))
            b′ = wrapvec(b_fromvec(bv))
            x₀ = wrapvec(zero(b_fromvec(bv)))
            x, info = linsolve(
                A′,
                b′,
                x₀,
                GMRES(; tol = tol, krylovdim = n, maxiter = 1)
            )
            info.converged == 0 && @warn "linsolve did not converge"
            xv, = to_vec(unwrapvec(x))
            return xv
        end

        (JA, Jb) = FiniteDifferences.jacobian(
            central_fdm(20, 1; factor = cond(A)), f, Avec, bvec
        )
        (JA′, Jb′) = Zygote.jacobian(f, Avec, bvec)
        @test JA ≈ JA′ rtol = cond(A) * precision(T)
        @test Jb ≈ Jb′ rtol = cond(A) * precision(T)
    end
end

@testset "Large linsolve AD test" begin
    for T in (Float64, ComplexF64)
        A = rand(T, (N, N)) .- one(T) / 2
        A = I - (9//10) * A / maximum(abs, eigvals(A))
        Avec, matfromvec = to_vec(A)
        
        b = 2 * (rand(T, N) .- one(T) / 2)
        bvec, vecfromvec = to_vec(b)
        
        c = rand(T)
        d = rand(T)
        
        tol = precision(T)
        
        function f(Av, bv, a₀, a₁)
            A′ = wrapop(matfromvec(Av))
            b′ = wrapvec(vecfromvec(bv))
            x₀ = wrapvec(zero(vecfromvec(bv)))
            x, info = linsolve(A′, b′, x₀, GMRES(; tol = tol, krylovdim = 20), a₀, a₁)
            info.converged == 0 && @warn "linsolve did not converge"
            xv, = to_vec(unwrapvec(x))
            return xv
        end
        
        (JA, Jb, Jc, Jd) = FiniteDifferences.jacobian(central_fdm(20, 1; factor = precision(T) / eps(real(T)), f, Avec, bvec, c, d)
        (JA′, Jb′, Jc′, Jd′) = Zygote.jacobian(f, Avec, bvec, c, d)
        
        @test JA ≈ JA′
        @test Jb ≈ Jb′
        @test Jc ≈ Jc′
        @test Jd ≈ Jd′
    end
end

# module EigsolveAD
# using KrylovKit, LinearAlgebra
# using Random, Test
# using ChainRulesCore, ChainRulesTestUtils, Zygote

# precision(T::Type{<:Number}) = eps(real(T))^(2 / 3)
# @testset "eigsolve AD" begin
#     @testset for n in (2, 10, 20)

#         @testset "Lanczos - full" begin
#             @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
#                 A = rand(T, (n, n)) .- one(T) / 2
#                 A = Hermitian((A + A') / 2)
#                 v = rand(T, (n,))
#                 alg = Lanczos(; krylovdim=2 * n, maxiter=1, tol=precision(T))
#                 which = :LM
#                 function f(A)
#                     vals, vecs, info = eigsolve(A, v, n, which, alg)

#                     vecs_phased = map(vecs) do vec
#                         return vec ./ exp(angle(vec[1])im)
#                     end
#                     D = vcat(vals...)
#                     U = hcat(vecs_phased...)
#                     return D, U
#                 end

#                 function g(A)
#                     vals, vecs = eigen(A; sortby=x -> -abs(x))
#                     vecs_phased = map(1:size(vecs, 2)) do i
#                         return vecs[:, i] ./ exp(angle(vecs[1, i])im)
#                     end
#                     return vals, hcat(vecs_phased...)
#                 end

#                 function h(A)
#                     vals, vecs = eigsolve(v, n, which, alg) do x
#                         return A * x
#                     end
#                     vecs_phased = map(vecs) do vec
#                         return vec ./ exp(angle(vec[1])im)
#                     end
#                     return vcat(vals...), hcat(vecs_phased...)
#                 end

#                 y1, back1 = pullback(f, A)
#                 y2, back2 = pullback(g, A)
#                 y3, back3 = pullback(h, A)

#                 for i in 1:2
#                     @test y1[i] ≈ y2[i]
#                     @test y2[i] ≈ y3[i]
#                 end

#                 for i in 1:3
#                     Δvals = rand(T, (n,))
#                     Δvecs = rand(T, (n, n))
#                     @test first(back1((Δvals, Δvecs))) ≈ first(back2((Δvals, Δvecs)))
#                     @test first(back2((Δvals, Δvecs))) ≈ first(back3((Δvals, Δvecs)))
#                 end
#             end
#         end

#         @testset "Arnoldi - full" begin
#             @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
#                 A = rand(T, (n, n)) .- one(T) / 2
#                 v = rand(T, (n,))

#                 alg = Arnoldi(; krylovdim=2 * n, maxiter=1, tol=precision(T))
#                 which = :LM

#                 function f(A)
#                     vals, vecs, _ = eigsolve(A, v, n, which, alg)
#                     vecs_phased = map(vecs) do vec
#                         return vec ./ exp(angle(vec[1])im)
#                     end
#                     D = vcat(vals...)
#                     U = hcat(vecs_phased...)
#                     D, U
#                     return D, U
#                 end

#                 function g(A)
#                     vals, vecs = eigen(A; sortby=x -> -abs(x))
#                     vecs_phased = map(1:size(vecs, 2)) do i
#                         return vecs[:, i] ./ exp(angle(vecs[1, i])im)
#                     end
#                     vals, vecs_phased
#                     return vals, hcat(vecs_phased...)
#                 end

#                 function h(A)
#                     vals, vecs, _ = eigsolve(v, n, which, alg) do x
#                         return A * x
#                     end
#                     vecs_phased = map(vecs) do vec
#                         return vec ./ exp(angle(vec[1])im)
#                     end
#                     return vcat(vals...), hcat(vecs_phased...)
#                 end

#                 y1, back1 = pullback(f, A)
#                 y2, back2 = pullback(g, A)
#                 y3, back3 = pullback(h, A)

#                 for i in 1:2
#                     @test y1[i] ≈ y2[i]
#                     @test y2[i] ≈ y3[i]
#                 end

#                 for i in 1:1
#                     Δvals = rand(T, (n,))
#                     Δvecs = rand(T, (n, n))
#                     @test first(back1((Δvals, Δvecs))) ≈ first(back2((Δvals, Δvecs)))
#                     @test first(back2((Δvals, Δvecs))) ≈ first(back3((Δvals, Δvecs)))
#                 end
#             end
#         end
#     end
# end
# end