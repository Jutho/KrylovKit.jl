# lanczos.jl
#
# Lanczos iteration for constructing the orthonormal basis of a Krylov subspace.
struct LanczosIterator{F,T,O<:Orthogonalizer}
    operator::F
    v₀::T
    krylovdim::Int
    tol::Real
    orth::O
    keepvecs::Bool
end

struct LanczosFact{T, S<:Real} <: KrylovFactorization{T}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k+1
    αs::Vector{S}
    βs::Vector{S}
end

basis(F::LanczosFact) = length(F.V) == F.k+1 ? F.V : error("Not keeping vectors during Lanczos factorization")
matrix(F::LanczosFact) = SymTridiagonal(F.αs, F.βs)
@inbounds normres(F::LanczosFact) = F.βs[F.k]
residual(F::LanczosFact) = normres(F)*F.V[end]

function LanczosIterator(A, v₀, orth = Defaults.orth; krylovdim::Int = length(v₀), tol::Real = 0, keepvecs=true)
    @assert krylovdim > 0
    if !keepvecs && isa(orth, Reorthogonalizer)
        error("Cannot use reorthogonalization without keeping all Krylov vectors")
    end
    return LanczosIterator(A, v₀, min(krylovdim, length(v₀)), tol, orth, keepvecs)
end

function Base.start(iter::LanczosIterator)
    m = iter.krylovdim
    v = iter.v₀ / vecnorm(iter.v₀) # division might change eltype
    w₁ = apply(iter.operator, v) # applying the operator might change eltype
    w₀ = copy!(similar(w₁), v)
    w₁, β, α = orthonormalize!(w₁, w₀, iter.orth)
    n = hypot(α,β)
    imag(α) <= 10*n || error("operator does not appear to be hermitian: $(imag(α)) vs $n")

    V = OrthonormalBasis([w₀,w₁])
    iter.keepvecs && sizehint!(V, m+1)
    S = eltype(β)
    αs = sizehint!(S[real(α)], m)
    βs = sizehint!(S[β], m)

    return LanczosFact(1, V, αs, βs)
end
function start!(iter::LanczosIterator, state::LanczosFact)
    m = iter.krylovdim
    v₀ = iter.v₀
    V = state.V
    αs = empty!(state.αs)
    βs = empty!(state.βs)
    while length(V) > 1
        pop!(V)
    end

    v = scale!(V[1], v₀, 1/vecnorm(v₀))
    w = iter.operator(v)
    w, β, α = orthonormalize!(w, v, iter.orth)
    n = hypot(α,β)
    imag(α) <= 10*n || error("operator does not appear to be hermitian: $(imag(α)) vs $n")

    push!(αs, real(α))
    push!(βs, β)
    return LanczosFact(1, V, αs, βs)
end

# return type declatation required because iter.tol is Real
Base.done(iter::LanczosIterator, state::LanczosFact)::Bool = state.k >= iter.krylovdim || normres(state) < iter.tol

function Base.next(iter::LanczosIterator, state::LanczosFact)
    state = next!(iter, deepcopy(state))
    return state, state
end
function next!(iter::LanczosIterator, state::LanczosFact)
    k = state.k
    V = state.V

    βold = last(state.βs)
    w, α, β = lanczosrecurrence(iter.operator, V, βold, iter.orth)
    n = hypot(α, β, βold)
    imag(α) <= 10*eps(n) || error("operator does not appear to be hermitian: $(imag(α)) vs $n")

    αs = push!(state.αs, real(α))
    βs = push!(state.βs, β)

    !iter.keepvecs && shift!(V) # remove oldest V if not keepvecs
    push!(V, w)

    return LanczosFact(k+1, V, αs, βs)
end

# Exploit hermiticity to "simplify" orthonormalization process:
# Lanczos three-term recurrence relation
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::Union{ClassicalGramSchmidt,ModifiedGramSchmidt})
    v = V[end]
    w = apply(operator, v)
    w = axpy!( -β, V[end-1], w)

    w, α = orthogonalize!(w, v, orth)
    β = vecnorm(w)
    scale!(w, 1/β)

    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt2)
    v = V[end]
    w = apply(operator, v)
    w = axpy!( -β, V[end-1], w)

    w, α = orthogonalize!(w, v, ClassicalGramSchmidt())
    w, s = orthogonalize!(w, V, ClassicalGramSchmidt())
    α += s[end]
    β = vecnorm(w)
    scale!(w, 1/β)

    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt2)
    v = V[end]
    w = apply(operator, v)
    w = axpy!( -β, V[end-1], w)

    w, α = orthogonalize!(w, v, ModifiedGramSchmidt())

    s = α
    for q in V
        w, s = orthogonalize!(w, q, ModifiedGramSchmidt())
    end
    α += s
    β = vecnorm(w)
    scale!(w, 1/β)

    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ClassicalGramSchmidtIR)
    v = V[end]
    w = apply(operator, v)
    w = axpy!( -β, V[end-1], w)

    w, α = orthogonalize!(w, v, ClassicalGramSchmidt())
    ab2 = abs2(α) + abs2(β)
    β = vecnorm(w)
    nold = sqrt(abs2(β)+ab2)
    while β < orth.η * nold
        nold = β
        w, s = orthogonalize!(w, V, ClassicalGramSchmidt())
        α += s[end]
        β = vecnorm(w)
    end
    scale!(w, 1/β)

    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ModifiedGramSchmidtIR)
    v = V[end]
    w = apply(operator, v)
    w = axpy!( -β, V[end-1], w)

    w, α = orthogonalize!(w, v, ModifiedGramSchmidt())
    ab2 = abs2(α) + abs2(β)
    β = vecnorm(w)
    nold = sqrt(abs2(β)+ab2)
    while β < orth.η * nold
        nold = β
        s = zero(α)
        for q in V
            w, s = orthogonalize!(w, q, ModifiedGramSchmidt())
        end
        α += s
        β = vecnorm(w)
    end
    scale!(w, 1/β)

    return w, α, β
end
