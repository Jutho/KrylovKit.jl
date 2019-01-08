# gkl.jl

mutable struct GKLFactorization{T, S<:Real}
    k::Int # current Krylov dimension
    U::OrthonormalBasis{T} # basis of length k
    V::OrthonormalBasis{T} # basis of length k
    αs::Vector{S}
    βs::Vector{S}
    r::T
end

Base.length(F::GKLFactorization) = F.k
Base.sizehint!(F::GKLFactorization, n) = begin
    sizehint!(F.U, n)
    sizehint!(F.V, n)
    sizehint!(F.αs, n)
    sizehint!(F.βs, n)
    return F
end
Base.eltype(F::GKLFactorization) = eltype(typeof(F))
Base.eltype(::Type{<:GKLFactorization{<:Any,S}}) where {S} = S

function basis(F::GKLFactorization, which::Symbol)
    length(F.U) == F.k || error("Not keeping vectors during GKL bidiagonalization")
    which == :U || which == :V || error("invalid flag for specifying basis")
    return which == :U ? F.U : F.V
end
rayleighquotient(F::GKLFactorization) = Bidiagonal(view(F.αs,1:F.k),
                                        view(F.βs,1:(F.k-1)),:L)
residual(F::GKLFactorization) = F.r
@inbounds normres(F::GKLFactorization) = F.βs[F.k]
rayleighextension(F::GKLFactorization) = SimpleBasisVector(F.k, F.k)

# GKL iteration for constructing the orthonormal basis of a Krylov subspace.
struct GKLIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    operator::F
    u₀::T
    orth::O
    keepvecs::Bool
    function GKLIterator{F,T,O}(operator::F, u₀::T, orth::O, keepvecs::Bool) where
        {F,T,O<:Orthogonalizer}

        if !keepvecs && isa(orth, Reorthogonalizer)
            error("Cannot use reorthogonalization without keeping all Krylov vectors")
        end
        new{F,T,O}(operator, u₀, orth, keepvecs)
    end
end
GKLIterator(operator::F, u₀::T, orth::O = KrylovDefaults.orth, keepvecs::Bool = true) where
                {F,T,O<:Orthogonalizer} = GKLIterator{F,T,O}(operator, u₀, orth, keepvecs)
GKLIterator(A::AbstractMatrix, u₀::AbstractVector, orth::O = KrylovDefaults.orth,
                keepvecs::Bool = true) where {O<:Orthogonalizer} =
    GKLIterator((x,flag)->flag ? A'*x : A*x, u₀, orth, keepvecs)

Base.IteratorSize(::Type{<:GKLIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:GKLIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::GKLIterator)
    state = initialize(iter)
    return state, state
end
function Base.iterate(iter::GKLIterator, state::GKLFactorization)
    nr = normres(state)
    if nr < eps(typeof(nr))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end

function initialize(iter::GKLIterator; verbosity::Int = 0)
    β₀ = norm(iter.u₀)
    β₀ == 0 && throw(ArgumentError("initial vector should not have norm zero"))
    invβ₀ = one(eltype(iter.u₀))/β₀
    T = typeof(invβ₀) # division might change eltype
    u₀ = mul!(similar(iter.u₀, T), iter.u₀, invβ₀)
    v = iter.operator(u₀, true) # apply adjoint operator, might change eltype
    u = eltype(v) == eltype(u₀) ? u₀ : copyto!(similar(v), u₀)
    α = norm(v)
    rmul!(v, 1/α)
    r = iter.operator(v, false) # apply operator
    r = axpy!(-α, u, r)
    β = norm(r)

    U = OrthonormalBasis([u])
    V = OrthonormalBasis([v])
    S = eltype(α)
    αs = S[α]
    βs = S[β]
    if verbosity > 0
        @info "GKL iteration step 1: normres = $β"
    end

    return GKLFactorization(1, U, V, αs, βs, r)
end
function initialize!(iter::GKLIterator, state::GKLFactorization; verbosity::Int = 0)
    V = state.V
    while length(U) > 1
        pop!(U)
    end
    V = empty!(state.V)
    αs = empty!(state.αs)
    βs = empty!(state.βs)

    u = mul!(V[1], iter.u₀, 1/norm(iter.u₀))
    v = iter.operator(u, true)
    α = norm(v)
    rmul!(v, 1/α)
    r = iter.operator(v, false) # apply operator
    r = axpy!(-α, u, r)
    β = norm(r)

    state.k = 1
    push!(V, v)
    push!(αs, α)
    push!(βs, β)
    state.r = r
    if verbosity > 0
        @info "GKL iteration step 1: normres = $β"
    end

    return state
end
function expand!(iter::GKLIterator, state::GKLFactorization; verbosity::Int = 0)
    βold = normres(state)
    U = state.U
    V = state.V
    r = state.r
    U = push!(U, rmul!(r, 1/βold))
    v, r, α, β = gklrecurrence(iter.operator, U, V, βold, iter.orth)

    push!(V, v)
    push!(state.αs, α)
    push!(state.βs, β)

    #!iter.keepvecs && popfirst!(state.V) # remove oldest V if not keepvecs

    state.k += 1
    state.r = r
    if verbosity > 0
        @info "GKL iteration step $(state.k): normres = $β"
    end

    return state
end
function shrink!(state::GKLFactorization, k)
    length(state) == length(state.V) ||
        error("we cannot shrink GKLFactorization without keeping vectors")
    length(state) <= k && return state
    U = state.U
    V = state.V
    while length(V) > k+1
        pop!(U)
        pop!(V)
    end
    pop!(V)
    r = pop!(U)
    resize!(state.αs, k)
    resize!(state.βs, k)
    state.k = k
    state.r = rmul!(r, normres(state))
    return state
end

# Golub-Kahan-Lanczos recurrence relation
function gklrecurrence(operator, U::OrthonormalBasis, V::OrthonormalBasis, β,
                        orth::Union{ClassicalGramSchmidt,ModifiedGramSchmidt})
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    α = norm(v)
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    β = norm(r)
    return v, r, α, β
end
function gklrecurrence(operator, U::OrthonormalBasis, V::OrthonormalBasis, β,
                        orth::ClassicalGramSchmidt2)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v) # not necessary if we definitely reorthogonalize next step and previous step
    # v, = orthogonalize!(v, V, ClassicalGramSchmidt())
    α = norm(v)
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    r, = orthogonalize!(r, U, ClassicalGramSchmidt())
    β = norm(r)
    return v, r, α, β
end
function gklrecurrence(operator, U::OrthonormalBasis, V::OrthonormalBasis, β,
                        orth::ModifiedGramSchmidt2)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    # for q in V # not necessary if we definitely reorthogonalize next step and previous step
    #     v, = orthogonalize!(v, q, ModifiedGramSchmidt())
    # end
    α = norm(v)
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    for q in U
        r, = orthogonalize!(r, q, ModifiedGramSchmidt())
    end
    β = norm(r)
    return v, r, α, β
end
function gklrecurrence(operator, U::OrthonormalBasis, V::OrthonormalBasis, β,
                        orth::ClassicalGramSchmidtIR)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    α = norm(v)
    nold = sqrt(abs2(α) + abs2(β))
    while α < orth.η * nold
        nold = α
        v, = orthogonalize!(v, V, ClassicalGramSchmidt())
        α = norm(v)
    end
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    β = norm(r)
    nold = sqrt(abs2(α) + abs2(β))
    while eps(one(β)) < β < orth.η * nold
        nold = β
        r, = orthogonalize!(r, U, ClassicalGramSchmidt())
        β = norm(r)
    end

    return v, r, α, β
end
function gklrecurrence(operator, U::OrthonormalBasis, V::OrthonormalBasis, β,
                        orth::ModifiedGramSchmidtIR)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    α = norm(v)
    nold = sqrt(abs2(α) + abs2(β))
    while eps(one(α)) < α < orth.η * nold
        nold = α
        for q in V
            v, = orthogonalize!(v, q, ModifiedGramSchmidt())
        end
        α = norm(v)
    end
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    β = norm(r)
    nold = sqrt(abs2(α) + abs2(β))
    while eps(one(β)) < β < orth.η * nold
        nold = β
        for q in U
            r, = orthogonalize!(r, q, ModifiedGramSchmidt())
        end
        β = norm(r)
    end

    return v, r, α, β
end
