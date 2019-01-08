# lanczos.jl

mutable struct LanczosFactorization{T, S<:Real} <: KrylovFactorization{T,S}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k
    αs::Vector{S}
    βs::Vector{S}
    r::T
end

Base.length(F::LanczosFactorization) = F.k
Base.sizehint!(F::LanczosFactorization, n) = begin
    sizehint!(F.V, n)
    sizehint!(F.αs, n)
    sizehint!(F.βs, n)
    return F
end
Base.eltype(F::LanczosFactorization) = eltype(typeof(F))
Base.eltype(::Type{<:LanczosFactorization{<:Any,S}}) where {S} = S

basis(F::LanczosFactorization) = length(F.V) == F.k ? F.V :
    error("Not keeping vectors during Lanczos factorization")
rayleighquotient(F::LanczosFactorization) = SymTridiagonal(F.αs, F.βs)
residual(F::LanczosFactorization) = F.r
@inbounds normres(F::LanczosFactorization) = F.βs[F.k]
rayleighextension(F::LanczosFactorization) = SimpleBasisVector(F.k, F.k)

# Lanczos iteration for constructing the orthonormal basis of a Krylov subspace.
struct LanczosIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    operator::F
    x₀::T
    orth::O
    keepvecs::Bool
    function LanczosIterator{F,T,O}(operator::F, x₀::T, orth::O, keepvecs::Bool) where
        {F,T,O<:Orthogonalizer}

        if !keepvecs && isa(orth, Reorthogonalizer)
            error("Cannot use reorthogonalization without keeping all Krylov vectors")
        end
        new{F,T,O}(operator, x₀, orth, keepvecs)
    end
end
LanczosIterator(operator::F, x₀::T, orth::O = KrylovDefaults.orth,
                keepvecs::Bool = true) where {F,T,O<:Orthogonalizer} =
    LanczosIterator{F,T,O}(operator, x₀, orth, keepvecs)
LanczosIterator(A::AbstractMatrix, x₀::AbstractVector, orth::O = KrylovDefaults.orth,
                keepvecs::Bool = true) where {O<:Orthogonalizer} =
    LanczosIterator(x->A*x, x₀, orth, keepvecs)

Base.IteratorSize(::Type{<:LanczosIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:LanczosIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::LanczosIterator)
    state = initialize(iter)
    return state, state
end
function Base.iterate(iter::LanczosIterator, state::LanczosFactorization)
    nr = normres(state)
    if nr < eps(typeof(nr))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end

function initialize(iter::LanczosIterator; verbosity::Int = 0)
    β₀ = norm(iter.x₀)
    β₀ == 0 && throw(ArgumentError("initial vector should not have norm zero"))
    invβ₀ = one(eltype(iter.x₀))/β₀
    T = typeof(invβ₀) # division might change eltype
    x₀ = mul!(similar(iter.x₀, T), iter.x₀, invβ₀)
    w = iter.operator(x₀) # applying the operator might change eltype
    v = eltype(x₀) == eltype(w) ? x₀ : copyto!(similar(w), x₀)
    r, α = orthogonalize!(w, v, iter.orth)
    β = norm(r)
    n = hypot(α,2*β)
    imag(α) <= sqrt(max(eps(n),eps(one(n)))) ||
        error("operator does not appear to be hermitian: $(imag(α)) vs $n")

    V = OrthonormalBasis([v])
    αs = [real(α)]
    βs = [β]
    if verbosity > 0
        @info "Lanczos iteration step 1: normres = $β"
    end

    return LanczosFactorization(1, V, αs, βs, r)
end
function initialize!(iter::LanczosIterator, state::LanczosFactorization; verbosity::Int = 0)
    x₀ = iter.x₀
    V = state.V
    while length(V) > 1
        pop!(V)
    end
    αs = empty!(state.αs)
    βs = empty!(state.βs)

    v = mul!(V[1], x₀, 1/norm(x₀))
    w = iter.operator(v)
    r, α = orthogonalize!(w, v, iter.orth)
    β = norm(r)
    n = hypot(α,β)
    imag(α) <= sqrt(max(eps(n),eps(one(n)))) ||
        error("operator does not appear to be hermitian: $(imag(α)) vs $n")

    state.k = 1
    push!(αs, real(α))
    push!(βs, β)
    state.r = r
    if verbosity > 0
        @info "Lanczos iteration step 1: normres = $β"
    end
    return state
end
function expand!(iter::LanczosIterator, state::LanczosFactorization; verbosity::Int = 0)
    βold = normres(state)
    V = state.V
    r = state.r
    V = push!(V, rmul!(r, 1/βold))
    r, α, β = lanczosrecurrence(iter.operator, V, βold, iter.orth)
    n = hypot(α, β, βold)
    imag(α) <= sqrt(max(eps(n),eps(one(n)))) ||
        error("operator does not appear to be hermitian: $(imag(α)) vs $n")

    αs = push!(state.αs, real(α))
    βs = push!(state.βs, β)

    !iter.keepvecs && popfirst!(state.V) # remove oldest V if not keepvecs

    state.k += 1
    state.r = r
    if verbosity > 0
        @info "Lanczos iteration step $(state.k): normres = $β"
    end
    return state
end
function shrink!(state::LanczosFactorization, k)
    length(state) == length(state.V) ||
        error("we cannot shrink LanczosFactorization without keeping Lanczos vectors")
    length(state) <= k && return state
    V = state.V
    while length(V) > k+1
        pop!(V)
    end
    r = pop!(V)
    resize!(state.αs, k)
    resize!(state.βs, k)
    state.k = k
    state.r = rmul!(r, normres(state))
    return state
end

# Exploit hermiticity to "simplify" orthonormalization process:
# Lanczos three-term recurrence relation
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt)
    v = V[end]
    w = apply(operator, v)
    α = dot(v, w)
    w = axpy!(-β, V[end-1], w)
    w = axpy!(-α, v, w)
    β = norm(w)
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt)
    v = V[end]
    w = apply(operator, v)
    w = axpy!( -β, V[end-1], w)
    α = dot(v, w)
    w = axpy!(-α, v, w)
    β = norm(w)
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt2)
    v = V[end]
    w = operator(v)
    α = dot(v, w)
    w = axpy!(-β, V[end-1], w)
    w = axpy!(-α, v, w)

    w, s = orthogonalize!(w, V, ClassicalGramSchmidt())
    α += s[end]
    β = norm(w)
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt2)
    v = V[end]
    w = operator(v)
    w = axpy!( -β, V[end-1], w)
    w, α = orthogonalize!(w, v, ModifiedGramSchmidt())

    s = α
    for q in V
        w, s = orthogonalize!(w, q, ModifiedGramSchmidt())
    end
    α += s
    β = norm(w)
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ClassicalGramSchmidtIR)
    v = V[end]
    w = apply(operator, v)
    α = dot(v, w)
    w = axpy!(-β, V[end-1], w)
    w = axpy!(-α, v, w)

    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β)+ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        w, s = orthogonalize!(w, V, ClassicalGramSchmidt())
        α += s[end]
        β = norm(w)
    end
    return w, α, β
end
function lanczosrecurrence(operator, V::OrthonormalBasis, β, orth::ModifiedGramSchmidtIR)
    v = V[end]
    w = apply(operator, v)
    w = axpy!( -β, V[end-1], w)

    w, α = orthogonalize!(w, v, ModifiedGramSchmidt())
    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β)+ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        s = zero(α)
        for q in V
            w, s = orthogonalize!(w, q, ModifiedGramSchmidt())
        end
        α += s
        β = norm(w)
    end
    return w, α, β
end
