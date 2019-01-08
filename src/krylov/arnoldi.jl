# arnoldi.jl

mutable struct ArnoldiFactorization{T,S} <: KrylovFactorization{T,S}
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
struct ArnoldiIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    operator::F
    x₀::T
    orth::O
end
ArnoldiIterator(A, x₀) = ArnoldiIterator(A, x₀, KrylovDefaults.orth)
ArnoldiIterator(A::AbstractMatrix, x₀::AbstractVector, orth::Orthogonalizer =
                KrylovDefaults.orth) = ArnoldiIterator(x->A*x, x₀, orth)

Base.IteratorSize(::Type{<:ArnoldiIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:ArnoldiIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::ArnoldiIterator)
    state = initialize(iter)
    return state, state
end
function Base.iterate(iter::ArnoldiIterator, state)
    nr = normres(state)
    if nr < eps(typeof(nr))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end

function initialize(iter::ArnoldiIterator; verbosity::Int = 0)
    β₀ = norm(iter.x₀)
    β₀ == 0 && throw(ArgumentError("initial vector should not have norm zero"))
    invβ₀ = one(eltype(iter.x₀))/β₀
    T = typeof(invβ₀) # division might change eltype
    x₀ = mul!(similar(iter.x₀, T), iter.x₀, invβ₀)
    w = iter.operator(x₀) # applying the operator might change eltype
    v = eltype(x₀) == eltype(w) ? x₀ : copyto!(similar(w), x₀)
    r, α = orthogonalize!(w, v, iter.orth)
    β = norm(r)
    V = OrthonormalBasis([v])
    H = [α, β]
    if verbosity > 0
        @info "Arnoldi iteration step 1: normres = $β"
    end

    state = ArnoldiFactorization(1, V, H, r)
end
function initialize!(iter::ArnoldiIterator, state::ArnoldiFactorization; verbosity::Int = 0)
    x₀ = iter.x₀
    V = state.V
    while length(V) > 1
        pop!(V)
    end
    H = empty!(state.H)

    v = mul!(V[1], x₀, 1/norm(x₀))
    w = iter.operator(v)
    r, α = orthogonalize!(w, v, iter.orth)
    β = norm(r)
    state.k = 1
    push!(H, α, β)
    state.r = r
    if verbosity > 0
        @info "Arnoldi iteration step 1: normres = $β"
    end
    return state
end
function expand!(iter::ArnoldiIterator, state::ArnoldiFactorization; verbosity::Int = 0)
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
    if verbosity > 0
        @info "Arnoldi iteration step $k: normres = $β"
    end
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
function arnoldirecurrence!(operator, V::OrthonormalBasis, h::AbstractVector,
                            orth::Orthogonalizer)
    w = operator(last(V))
    r, h = orthogonalize!(w, V, h, orth)
    return r, norm(r)
end
