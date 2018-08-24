# arnoldi.jl

mutable struct ArnoldiFactorization{T,S} <: KrylovFactorization{T}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k
    H::Vector{S} # stores the Hessenberg matrix in packed form
    r::T # residual
end

Base.length(F::ArnoldiFactorization) = F.k
Base.sizehint!(F::ArnoldiFactorization, n) = begin
    sizehint!(F.V, n)
    sizehint!(F.H, (n*n + 3*n) >> 1)
    return F
end
Base.eltype(F::ArnoldiFactorization) = eltype(typeof(F))
Base.eltype(::Type{<:ArnoldiFactorization{<:Any,S}}) where {S} = S

basis(F::ArnoldiFactorization) = F.V
rayleighquotient(F::ArnoldiFactorization) = PackedHessenberg(F.H, F.k)
residual(F::ArnoldiFactorization) = F.r
@inbounds normres(F::ArnoldiFactorization) = abs(F.H[end])
rayleighextension(F::ArnoldiFactorization) = SimpleBasisVector(F.k, F.k)

# Arnoldi iteration for constructing the orthonormal basis of a Krylov subspace.
struct ArnoldiIterator{F,T,O<:Orthogonalizer}
    operator::F
    v₀::T
    orth::O
end
ArnoldiIterator(A, v₀) = ArnoldiIterator(A, v₀, Defaults.orth)

Base.IteratorSize(::Type{<:ArnoldiIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:ArnoldiIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::ArnoldiIterator)
    state = initialize(iter)
    return state, state
end
function Base.iterate(iter::ArnoldiIterator, state)
    if normres(state) < eps(real(eltype(state)))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end

function initialize(iter::ArnoldiIterator)
    β₀ = norm(iter.v₀)
    T = typeof(one(eltype(iter.v₀))/β₀) # division might change eltype
    v₀ = mul!(similar(iter.v₀, T), iter.v₀, 1/β₀)
    w = apply(iter.operator, v₀) # applying the operator might change eltype
    v = eltype(v₀) == eltype(w) ? v₀ : copyto!(similar(w), v₀)
    r, α = orthogonalize!(w, v, iter.orth)
    β = norm(r)
    V = OrthonormalBasis([v])
    H = [α, β]
    state = ArnoldiFactorization(1, V, H, r)
end
function initialize!(iter::ArnoldiIterator, state::ArnoldiFactorization) # recylcle existing state
    v₀ = iter.v₀
    V = state.V
    while length(V) > 1
        pop!(V)
    end
    H = empty!(state.H)

    v = mul!(V[1], v₀, 1/norm(v₀))
    w = apply(iter.operator, v)
    r, α = orthogonalize!(w, v, iter.orth)
    β = norm(r)
    state.k = 1
    push!(H, α, β)
    state.r = r
    return state
end
function expand!(iter::ArnoldiIterator, state::ArnoldiFactorization)
    state.k += 1
    k = state.k
    V = state.V
    H = state.H
    r = state.r
    β = normres(state)
    push!(V, rmul!(r, 1/β))
    m = length(H)
    resize!(H, m+k+1)
    r, β = arnoldirecurrence!(iter.operator, V, view(H, (m+1):(m+k)), iter.orth)
    H[m+k+1] = β
    state.r = r
    return state
end
function shrink!(state::ArnoldiFactorization, k)
    length(state) <= k && return state
    V = state.V
    H = state.H
    while length(V) > k+1
        pop!(V)
    end
    r = pop!(V)
    resize!(H, (k*k + 3*k) >> 1)
    state.k = k
    state.r = rmul!(r, normres(state))
    return state
end

# Arnoldi recurrence: simply use provided orthonormalization routines
function arnoldirecurrence!(operator, V::OrthonormalBasis, h::AbstractVector, orth::Orthogonalizer)
    w = apply(operator, last(V))
    r, h = orthogonalize!(w, V, h, orth)
    return r, norm(r)
end
