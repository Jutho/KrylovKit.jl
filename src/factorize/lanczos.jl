# lanczos.jl
#
# Lanczos iteration for constructing the orthonormal basis of a Krylov subspace.
struct LanczosIterator{F,T,S,Sr<:Real,O<:Orthogonalizer}
    operator::F
    v₀::T
    eltype::Type{S}
    krylovdim::Int
    tol::Sr
    orth::O
    keepvecs::Bool
end

mutable struct LanczosFact{T, S<:Real} # S = real(eltype(T))
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k+1
    αs::Vector{S}
    βs::Vector{S}
end

basis(F::LanczosFact) = F.V
matrix(F::LanczosFact) = SymTridiagonal(F.αs, F.βs)
normres(F::LanczosFact) = F.βs[F.k]
residual(F::LanczosFact) = normres(F)*F.V[F.k+1]

function lanczos(A, v₀, orth = orthdefault, ::Type{S} = eltype(v); krylovdim = length(v), tol=abs(zero(S)), keepvecs=true) where {S}
    @assert krylovdim > 0
    if !keepvecs && isa(orth, Reorthogonalizer)
        error("Cannot use reorthogonalization without keeping all Krylov vectors")
    end
    tolerance::real(S) = tol
    krylovdimension::Int = min(krylovdim, length(v))

    LanczosIterator(A, v₀, S, krylovdimension, tolerance, orth, keepvecs)
end

function Base.start(iter::LanczosIterator)
    m = iter.krylovdim
    v₀ = iter.v₀
    S = iter.S
    v = scale!(similar(v₀, S), v₀, 1/vecnorm(v₀))
    V = sizehint!(OrthonormalBasis([v]), iter.keepvecs ? m+1 : 2)

    Sr = real(S)
    αs = sizehint!(Vector{Sr}(), m)
    βs = sizehint!(Vector{Sr}(), m)

    return LanczosFact{T,S}(0, V, αs, βs)
end
function start!(iter::LanczosIterator, state::LanczosFact)
    m = iter.krylovdim
    v₀ = iter.v₀
    S = iter.S
    @assert eltype(state.V[1]) == S
    @assert eltype(state.αs) == real(S)

    state.k = 0
    scale!(state.V[1], v₀, 1/vecnorm(v₀))
    while length(state.V) > 1
        pop!(state.V)
    end
    empty!(state.αs)
    empty!(state.βs)
end

function Base.done(iter::LanczosIterator, state::LanczosFact)
    k = state.k
    @inbounds return k >= iter.krylovdim || (k > 0 && normres(state) < iter.tol)
end

Base.next(iter::LanczosIterator, state::LanczosFact) = next!(iter, deepcopy(state))
function next!(iter::LanczosIterator, state::LanczosFact)
    k = (state.k += 1)
    V = state.V

    if k == 1
        βold = zero(eltype(state.βs))
        v = last(V)
        w = apply(iter.operator, v)
        w, α = orthogonalize!(w, v, iter.orth)
        β = vecnorm(w)
        scale!(w, 1/β)
    else
        βold = state.βs[end]
        w, α, β = lanczosrecurrence(iter.operator, V, βold, iter.orth)
    end

    n = sqrt(abs2(α)+abs2(β)+abs2(βold))
    imag(α) <= 10*eps(n) || error("operator does not appear to be hermitian: $(imag(α)) vs $n")
    push!(state.αs, real(α))
    push!(state.βs, β)

    !iter.keepvecs && k > 1 && shift!(V) # remove oldest V if not keepvecs
    push!(V, w)

    return state, state
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
