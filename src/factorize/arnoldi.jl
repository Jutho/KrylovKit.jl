# arnoldi.jl
#
# Arnoldi iteration for constructing the orthonormal basis of a Krylov subspace.
struct ArnoldiIterator{F,T,O<:Orthogonalizer}
    operator::F
    v₀::T
    orth::O
end
ArnoldiIterator(A, v₀) = ArnoldiIterator(A, v₀, Defaults.orth)

struct ArnoldiFact{T,S} <: KrylovFactorization{T}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k+1
    H::Vector{S} # stores the Hessenberg matrix in packed form
end

basis(F::ArnoldiFact) = F.V
matrix(F::ArnoldiFact) = PackedHessenberg(F.H, F.k)
# matrix(F::ArnoldiFact{<:Any,S}) where {S} = copy!(Array{S}(F.k,F.k), PackedHessenberg(F.H, F.k))
# # TODO: make everything work with PackedHessenberg directly
normres(F::ArnoldiFact) = abs(F.H[end])
residual(F::ArnoldiFact) = normres(F)*F.V[F.k+1]

function Base.start(iter::ArnoldiIterator)
    v = iter.v₀ / vecnorm(iter.v₀) # division might change eltype
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
    return ArnoldiFact(1, V, H)
end

# return type declatation required because iter.tol is Real
Base.done(iter::ArnoldiIterator, state::ArnoldiFact)::Bool = state.k >= length(iter.v₀)

function Base.next(iter::ArnoldiIterator, state::ArnoldiFact)
    nr = normres(state)
    state = next!(iter, deepcopy(state))
    return nr, state
end
function next!(iter::ArnoldiIterator, state::ArnoldiFact)
    k = state.k + 1
    V = state.V
    H = state.H
    m = length(H)
    resize!(H, m+k+1)
    w, β = arnoldirecurrence!(iter.operator, V, view(H, (m+1):(m+k)), iter.orth)
    push!(V, w)
    H[m+k+1] = β
    return ArnoldiFact(k, V, H)
end

# Arnoldi recurrence: simply use provided orthonormalization routines
function arnoldirecurrence!(operator, V::OrthonormalBasis, h::AbstractVector, orth::Orthogonalizer)
    w = apply(operator, last(V))
    w, β, h = orthonormalize!(w, V, h, orth)
    return w, β
end
