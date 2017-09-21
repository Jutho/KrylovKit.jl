# arnoldi.jl
#
# Arnoldi iteration for constructing the orthonormal basis of a Krylov subspace.
struct ArnoldiIterator{F,T,S,Sr<:Real,O<:Orthogonalizer}
    operator::F
    v₀::T
    eltype::Type{S}
    krylovdim::Int
    tol::Sr
    orth::O
end

mutable struct ArnoldiFact{T,S} # S = eltype(T)
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k+1
    H::Matrix{S} # matrix of Hessenberg form: (m+1) x m with m maximal krylov dimension
end

basis(F::ArnoldiFact) = F.V
matrix(F::ArnoldiFact) = view(F.H, 1:F.k, 1:F.k)
normres(F::ArnoldiFact) = abs(F.H[F.k+1,F.k])
residual(F::ArnoldiFact) = normres(F)*F.V[F.k+1]

function arnoldi(A, v₀, orth=orthdefault, ::Type{S} = eltype(v₀); krylovdim = length(v₀), tol = abs(zero(S))) where {S}
    @assert krylovdim > 0
    tolerance::real(S) = tol
    krylovdimension::Int = min(krylovdim, length(v₀))

    ArnoldiIterator(A, v₀, S, krylovdimension, tolerance, orth)
end

function Base.start(iter::ArnoldiIterator)
    m = iter.krylovdim
    v₀ = iter.v₀
    S = iter.eltype
    v = scale!(similar(v₀, S), v₀, 1/vecnorm(v₀))
    V = sizehint!(OrthonormalBasis([v]), m+1)
    H = zeros(S, (m+1,m))
    return ArnoldiFact(0, V, H)
end
function start!(iter::ArnoldiIterator, state::ArnoldiFact) # recylcle existing fact
    m = iter.krylovdim
    v₀ = iter.v₀
    S = iter.eltype
    @assert eltype(state.H) == S
    @assert eltype(state.V[1]) == S
    @assert size(state.H) == (m+1, m)

    state.k = 0
    scale!(state.V[1], v₀, 1/vecnorm(v₀))
    while length(state.V) > 1
        pop!(state.V)
    end
    fill!(state.H, 0)
    return state
end

function Base.done(iter::ArnoldiIterator, state::ArnoldiFact)
    k = state.k
    @inbounds return k >= iter.krylovdim || (k > 0 && normres(state) < iter.tol)
end

Base.next(iter::ArnoldiIterator, state::ArnoldiFact) = next!(iter, deepcopy(state))
function next!(iter::ArnoldiIterator, state::ArnoldiFact)
    V = state.V
    H = state.H
    k = (state.k += 1)
    w, β = arnoldirecurrence!(iter.operator, V, view(H,1:k,k), iter.orth)
    push!(V, w)
    H[k+1,k] = β
    return state, state
end

# Arnoldi recurrence: simply use provided orthonormalization routines
function arnoldirecurrence!(operator, V::OrthonormalBasis, h::AbstractVector, orth::Orthogonalizer)
    w = apply(operator, last(V))
    w, β, h = orthonormalize!(w, V, h, orth)
    return w, β
end
