# lanczos.jl
#
# Lanczos iteration for constructing the orthonormal basis of a Krylov subspace.
struct LanczosIterator{F,T,O<:Orthogonalizer}
    operator::F
    v₀::T
    orth::O
    keepvecs::Bool
    function LanczosIterator{F,T,O}(operator::F, v₀::T, orth::O, keepvecs::Bool) where {F,T,O<:Orthogonalizer}
        if !keepvecs && isa(orth, Reorthogonalizer)
            error("Cannot use reorthogonalization without keeping all Krylov vectors")
        end
        new{F,T,O}(operator, v₀, orth, keepvecs)
    end
end
LanczosIterator(operator::F, v₀::T, orth::O = Defaults.orth, keepvecs::Bool = true) where {F,T,O<:Orthogonalizer} = LanczosIterator{F,T,O}(operator, v₀, orth, keepvecs)

Base.iteratorsize(::Type{<:LanczosIterator}) = Base.HasLength()
Base.length(iter::LanczosIterator) = length(iter.v₀)

mutable struct LanczosFact{T, S<:Real} <: KrylovFactorization{T}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k+1
    αs::Vector{S}
    βs::Vector{S}
end

Base.length(F::LanczosFact) = F.k
Base.sizehint!(F::LanczosFact, n) = begin
    sizehint!(F.αs, n)
    sizehint!(F.βs, n)
    return F
end
Base.eltype(F::LanczosFact) = eltype(typeof(F))
Base.eltype(::Type{<:LanczosFact{<:Any,S}}) where {S} = S

basis(F::LanczosFact) = length(F.V) == F.k+1 ? F.V : error("Not keeping vectors during Lanczos factorization")
rayleighquotient(F::LanczosFact) = SymTridiagonal(F.αs, F.βs)
@inbounds normres(F::LanczosFact) = F.βs[F.k]
residual(F::LanczosFact) = normres(F)*F.V[end]

function Base.start(iter::LanczosIterator)
    v = iter.v₀ / vecnorm(iter.v₀) # division might change eltype
    w₁ = apply(iter.operator, v) # applying the operator might change eltype
    w₀ = copy!(similar(w₁), v)
    w₁, β, α = orthonormalize!(w₁, w₀, iter.orth)
    n = hypot(α,2*β)
    imag(α) <= sqrt(eps(n)) || error("operator does not appear to be hermitian: $(imag(α)) vs $n")

    V = OrthonormalBasis([w₀,w₁])
    S = eltype(β)
    αs = [real(α)]
    βs = [β]

    return LanczosFact(1, V, αs, βs)
end
function start!(iter::LanczosIterator, state::LanczosFact)
    v₀ = iter.v₀
    V = state.V
    while length(V) > 1
        pop!(V)
    end
    αs = empty!(state.αs)
    βs = empty!(state.βs)

    v = scale!(V[1], v₀, 1/vecnorm(v₀))
    w = apply(iter.operator, v)
    w, β, α = orthonormalize!(w, v, iter.orth)
    n = hypot(α,β)
    imag(α) <= 10*eps(n) || error("operator does not appear to be hermitian: $(imag(α)) vs $n")

    push!(V, w)
    push!(αs, real(α))
    push!(βs, β)
    state.k = 1
    return state
end

# return type declatation required because iter.tol is Real
Base.done(iter::LanczosIterator, state::LanczosFact) = length(state) == length(iter)

function Base.next(iter::LanczosIterator, state::LanczosFact)
    nr = normres(state)
    state = next!(iter, deepcopy(state))
    return nr, state
end
function next!(iter::LanczosIterator, state::LanczosFact)
    βold = normres(state)
    w, α, β = lanczosrecurrence(iter.operator, state.V, βold, iter.orth)
    n = hypot(α, β, βold)
    imag(α) <= 10*eps(n) || error("operator does not appear to be hermitian: $(imag(α)) vs $n")

    αs = push!(state.αs, real(α))
    βs = push!(state.βs, β)

    !iter.keepvecs && shift!(state.V) # remove oldest V if not keepvecs
    push!(state.V, w)

    state.k += 1

    return state
end

function shrink!(state::LanczosFact, k)
    length(state)+1 == length(state.V) || error("we cannot shrink LanczosFact without keeping Lanczos vectors")
    V = state.V
    while length(V) > k+1
        pop!(V)
    end
    return LanczosFact(k, V, resize!(state.αs, k), resize!(state.βs, k))
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
