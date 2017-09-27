# arnoldi.jl
#
# Arnoldi iteration for constructing the orthonormal basis of a Krylov subspace.
struct ArnoldiIterator{F,T,O<:Orthogonalizer}
    operator::F
    v₀::T
    krylovdim::Int
    tol::Real
    orth::O
end

struct ArnoldiFact{T,S} <: KrylovFactorization{T}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k+1
    H::Matrix{S} # matrix of Hessenberg form: (m+1) x m with m maximal krylov dimension
end

basis(F::ArnoldiFact) = F.V
matrix(F::ArnoldiFact) = view(F.H, 1:F.k, 1:F.k)
normres(F::ArnoldiFact) = abs(F.H[F.k+1,F.k])
residual(F::ArnoldiFact) = normres(F)*F.V[F.k+1]

function ArnoldiIterator(A, v₀, orth = Defaults.orth; krylovdim::Int = length(v₀), tol::Real = 0)
    @assert krylovdim > 0
    return ArnoldiIterator(A, v₀, min(krylovdim, length(v₀)), tol, orth)
end

function Base.start(iter::ArnoldiIterator)
    m = iter.krylovdim
    v = iter.v₀ / vecnorm(iter.v₀) # division might change eltype
    w₁ = apply(iter.operator, v) # applying the operator might change eltype
    w₀ = copy!(similar(w₁), v)
    w₁, β, α = orthonormalize!(w₁, w₀, iter.orth)

    V = sizehint!(OrthonormalBasis([w₀, w₁]), m+1)
    H = zeros(typeof(α), (m+1, m))
    H[1,1] = α
    H[2,1] = β
    return ArnoldiFact(1, V, H)
end
function start!(iter::ArnoldiIterator, state::ArnoldiFact) # recylcle existing fact
    m = iter.krylovdim
    v₀ = iter.v₀
    V = state.V
    H = state.H
    @assert size(H) == (m+1,m)
    while length(V) > 1
        pop!(V)
    end
    fill!(H, 0)

    v = scale!(V[1], v₀, 1/vecnorm(v₀))
    w = apply(iter.operator, v)
    w, β, α = orthonormalize!(w, v, iter.orth)
    push!(V, w)
    H[1,1] = α
    H[2,1] = β
    return ArnoldiFact(1, V, H)
end

# return type declatation required because iter.tol is Real
Base.done(iter::ArnoldiIterator, state::ArnoldiFact)::Bool = state.k >= iter.krylovdim || normres(state) < iter.tol

function Base.next(iter::ArnoldiIterator, state::ArnoldiFact)
    state = next!(iter, deepcopy(state))
    return state, state
end
function next!(iter::ArnoldiIterator, state::ArnoldiFact)
    k = state.k + 1
    V = state.V
    H = state.H
    w, β = arnoldirecurrence!(iter.operator, V, view(H, 1:k, k), iter.orth)
    push!(V, w)
    H[k+1,k] = β
    return ArnoldiFact(k, V, H)
end

# Arnoldi recurrence: simply use provided orthonormalization routines
function arnoldirecurrence!(operator, V::OrthonormalBasis, h::AbstractVector, orth::Orthogonalizer)
    w = apply(operator, last(V))
    w, β, h = orthonormalize!(w, V, h, orth)
    return w, β
end
