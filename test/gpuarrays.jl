using JLArrays
using Random: MersenneTwister
using KrylovKit: OrthonormalBasis, project!!, unproject!!, basistransform!,
    orthogonalize!!, reorthogonalize!!

# JLArrays 0.1 wraps host arrays without copying them, so copy to keep the reference intact
jl(x) = JLArray(copy(x))

if VERSION < v"1.10"
    # JLArrays 0.1, the last version supporting Julia < 1.10, reduces into `R.data`,
    # which fails when `R` is a view
    function JLArrays.GPUArrays.mapreducedim!(
            f, op, R::SubArray{<:Any, <:Any, <:JLArray},
            A::Union{AbstractArray, Broadcast.Broadcasted}; init = nothing
        )
        R′ = JLArray(Array(R))
        JLArrays.GPUArrays.mapreducedim!(f, op, R′, A; init = init)
        copyto!(R, R′)
        return R
    end
end

@testset "Orthogonalization against a GPU array basis ($T)" for T in (Float64, ComplexF64)
    rng = MersenneTwister(1234)
    m, k = 3N ÷ 2, 7
    bc = OrthonormalBasis{Vector{T}}()
    for _ in 1:k
        w, = orthonormalize(randn(rng, T, m), bc, mgs2)
        push!(bc, w)
    end
    bg = OrthonormalBasis(jl.(bc.basis))

    @testset "$alg" for alg in (cgs, mgs, cgs2, mgs2, cgsr, mgsr)
        v = randn(rng, T, m)
        wc, xc = orthogonalize(v, bc, alg)
        wg, xg = orthogonalize!!(jl(v), bg, alg)
        @test wg isa JLArray
        @test Array(wg) ≈ wc
        @test xg ≈ xc
    end
    @testset "reorthogonalize!! $alg" for alg in (cgs, mgs)
        v = randn(rng, T, m)
        x0 = randn(rng, T, k)
        wc, xc = reorthogonalize!!(copy(v), bc, copy(x0), alg)
        wg, xg = reorthogonalize!!(jl(v), bg, copy(x0), alg)
        @test Array(wg) ≈ wc
        @test xg ≈ xc
    end

    @testset "project!! and unproject!!" begin
        v = randn(rng, T, m)
        α, β = T(0.5), T(-2)
        r = 2:5
        y0 = randn(rng, T, length(r))
        @test project!!(copy(y0), bg, jl(v), α, β, r) ≈ project!!(copy(y0), bc, v, α, β, r)
        @test project!!(zeros(T, k), bg, jl(v)) ≈ project!!(zeros(T, k), bc, v)
        c = randn(rng, T, length(r))
        @test Array(unproject!!(jl(v), bg, c, α, β, r)) ≈ unproject!!(copy(v), bc, c, α, β, r)
        c = randn(rng, T, k)
        @test Array(bg * c) ≈ bc * c
        for r in ([1, 3, 6], 4:-1:2, 2:1)
            y0 = randn(rng, T, length(r))
            @test project!!(copy(y0), bg, jl(v), α, β, r) ≈ project!!(copy(y0), bc, v, α, β, r)
            c = randn(rng, T, length(r))
            @test Array(unproject!!(jl(v), bg, c, α, β, r)) ≈ unproject!!(copy(v), bc, c, α, β, r)
            @test Array(unproject!!(jl(v), bg, c, α, zero(T), r)) ≈ unproject!!(copy(v), bc, c, α, zero(T), r)
        end
    end

    @testset "empty basis, $alg" for alg in (cgs, mgs, cgs2, mgs2)
        v = randn(rng, T, m)
        b0 = OrthonormalBasis{typeof(jl(v))}()
        w, x = orthogonalize!!(jl(v), b0, alg)
        @test Array(w) ≈ v
        @test isempty(x)
    end

    @testset "basistransform!" begin
        U = Matrix(qr(randn(rng, T, k, k)).Q)[:, 1:4]
        bc′ = OrthonormalBasis(copy(bc.basis))
        bg′ = OrthonormalBasis(copy(bg.basis))
        olds = Array.(bg′.basis)
        basistransform!(bc′, U)
        basistransform!(bg′, U)
        @test all(Array(bg′[j]) ≈ bc′[j] for j in 1:k)
        @test all(Array(bg.basis[j]) == olds[j] for j in 1:k)
        v = randn(rng, T, m)
        @test project!!(zeros(T, k), bg′, jl(v)) ≈ project!!(zeros(T, k), bc′, v)
    end

    @testset "mutating the basis" begin
        bc′ = OrthonormalBasis(copy(bc.basis))
        bg′ = OrthonormalBasis(copy(bg.basis))
        v = randn(rng, T, m)
        project!!(zeros(T, k), bg′, jl(v))
        bc′[2] = scale!!(bc′[2], -1)
        bg′[2] = scale!!(bg′[2], -1)
        @test project!!(zeros(T, k), bg′, jl(v)) ≈ project!!(zeros(T, k), bc′, v)
        q = pop!(bg′)
        pop!(bc′)
        @test project!!(zeros(T, k - 1), bg′, jl(v)) ≈ project!!(zeros(T, k - 1), bc′, v)
        push!(bg′, scale!!(q, -1))
        push!(bc′, -bc[k])
        @test project!!(zeros(T, k), bg′, jl(v)) ≈ project!!(zeros(T, k), bc′, v)
        empty!(bg′)
        @test isempty(bg′)
    end
end

@testset "Krylov solvers with GPU arrays ($T)" for T in (Float64, ComplexF64)
    rng = MersenneTwister(5678)
    m = 3N ÷ 2
    A = rand(rng, T, (m, m)) .- one(T) / 2
    H = (A + A') / 2
    Ag = jl(A)
    Hg = jl(H)
    v = rand(rng, T, m)
    vg = jl(v)

    @testset "Lanczos eigsolve, $orth" for orth in (cgs2, mgs2, cgsr, mgsr)
        alg = Lanczos(; orth, krylovdim = 2n, maxiter = 300, tol = 10 * n * eps(real(T)))
        vals, vecs, info = eigsolve(x -> Hg * x, vg, n, :SR, alg)
        valsc, = eigsolve(x -> H * x, v, n, :SR, alg)
        @test info.converged >= n
        @test vals[1:n] ≈ valsc[1:n]
        for i in 1:n
            @test vecs[i] isa JLArray
            @test norm(Hg * vecs[i] - vals[i] * vecs[i]) < sqrt(eps(real(T))) * norm(H)
        end
    end

    @testset "Arnoldi eigsolve, $orth" for orth in (cgs2, mgs2, cgsr, mgsr)
        alg = Arnoldi(; orth, krylovdim = 2n, maxiter = 300, tol = 10 * n * eps(real(T)))
        vals, vecs, info = eigsolve(x -> Ag * x, vg, n, :LM, alg)
        valsc, = eigsolve(x -> A * x, v, n, :LM, alg)
        @test info.converged >= n
        @test vals[1:n] ≈ valsc[1:n]
        for i in 1:n
            @test norm(Ag * vecs[i] - vals[i] * vecs[i]) < sqrt(eps(real(T))) * norm(A)
        end
    end

    @testset "GMRES linsolve, $orth" for orth in (cgs2, mgs2, cgsr, mgsr)
        alg = GMRES(; orth, krylovdim = n, maxiter = 50, tol = 10 * m * eps(real(T)))
        b = rand(rng, T, m)
        a₁ = -one(T) / (4 * norm(A))
        x, info = linsolve(y -> Ag * y, jl(b), zerovector(jl(b)), alg, one(T), a₁)
        xc, infoc = linsolve(y -> A * y, b, zerovector(b), alg, one(T), a₁)
        @test infoc.converged == 1
        @test info.converged == 1
        @test x isa JLArray
        @test Array(x) ≈ xc
    end
end
