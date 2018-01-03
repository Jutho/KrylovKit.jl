# arnoldi.jl
#
# Arnoldi iteration for constructing the orthonormal basis of a Krylov subspace.
struct ArnoldiIterator{F,T,O<:Orthogonalizer}
    operator::F
    v₀::T
    orth::O
end
ArnoldiIterator(A, v₀) = ArnoldiIterator(A, v₀, Defaults.orth)

Base.iteratorsize(::Type{<:ArnoldiIterator}) = Base.HasLength()
Base.length(iter::ArnoldiIterator) = length(iter.v₀)

mutable struct ArnoldiFact{T,S} <: KrylovFactorization{T}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k+1
    H::Vector{S} # stores the Hessenberg matrix in packed form
end

Base.length(F::ArnoldiFact) = F.k
Base.sizehint!(F::ArnoldiFact, n) = begin
    sizehint!(F.H, (n*n + 3*n) >> 1)
    return F
end
Base.eltype(F::ArnoldiFact) = eltype(typeof(F))
Base.eltype(::Type{<:ArnoldiFact{<:Any,S}}) where {S} = S

basis(F::ArnoldiFact) = F.V
rayleighquotient(F::ArnoldiFact) = PackedHessenberg(F.H, F.k)
normres(F::ArnoldiFact) = abs(F.H[end])
residual(F::ArnoldiFact) = normres(F)*F.V[F.k+1]

function Base.start(iter::ArnoldiIterator)
    β₀ = vecnorm(iter.v₀)
    T = typeof(one(eltype(iter.v₀))/β₀) # division might change eltype
    v = scale!(similar(iter.v₀, T), iter.v₀, 1/β₀)
    w₁ = apply(iter.operator, v) # applying the operator might change eltype
    w₀ = copy!(similar(w₁), v)
    w₁, β, α = orthonormalize!(w₁, w₀, iter.orth)

    V = OrthonormalBasis([w₀, w₁])
    H = [α, β]
    return ArnoldiFact(1, V, H)
end
function start!(iter::ArnoldiIterator, state::ArnoldiFact) # recylcle existing state
    v₀ = iter.v₀
    V = state.V
    while length(V) > 1
        pop!(V)
    end
    H = empty!(state.H)

    v = scale!(V[1], v₀, 1/vecnorm(v₀))
    w = apply(iter.operator, v)
    w, β, α = orthonormalize!(w, v, iter.orth)

    push!(V, w)
    push!(H, α, β)

    state.k = 1
    return state
end

Base.done(iter::ArnoldiIterator, state::ArnoldiFact) = length(state) == length(iter)

function Base.next(iter::ArnoldiIterator, state::ArnoldiFact)
    nr = normres(state)
    state = next!(iter, deepcopy(state))
    return nr, state
end
function next!(iter::ArnoldiIterator, state::ArnoldiFact)
    state.k += 1
    k = state.k
    V = state.V
    H = state.H
    m = length(H)
    resize!(H, m+k+1)
    w, β = arnoldirecurrence!(iter.operator, V, view(H, (m+1):(m+k)), iter.orth)
    push!(V, w)
    H[m+k+1] = β
    return state
end

function shrink!(state::ArnoldiFact, k)
    V = state.V
    H = state.H
    while length(V) > k+1
        pop!(V)
    end
    resize!(H, (k*k + 3*k) >> 1)
    state.k = k
    return state
end

# Arnoldi recurrence: simply use provided orthonormalization routines
function arnoldirecurrence!(operator, V::OrthonormalBasis, h::AbstractVector, orth::Orthogonalizer)
    w = apply(operator, last(V))
    w, β, h = orthonormalize!(w, V, h, orth)
    return w, β
end
