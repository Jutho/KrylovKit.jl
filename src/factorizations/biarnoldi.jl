# biarnoldi.jl
mutable struct BiArnoldiFactorization{T,S} <: KrylovFactorization{T,S}
    k::Int # current Krylov dimension
    V::OrthonormalBasis{T} # basis of length k
    W::OrthonormalBasis{T} # basis of length k
    H::Vector{S} # stores the Hessenberg matrix in packed form
    K::Vector{S} # stores the Hessenberg matrix in packed form
    rV::T # residual
    rW::T # residual
end

Base.length(F::BiArnoldiFactorization) = F.k
Base.sizehint!(F::BiArnoldiFactorization, n) = begin
    sizehint!(F.V, n)
    sizehint!(F.W, n)
    sizehint!(F.H, (n * n + 3 * n) >> 1)
    sizehint!(F.K, (n * n + 3 * n) >> 1)
    return F
end
Base.eltype(F::BiArnoldiFactorization) = eltype(typeof(F))
Base.eltype(::Type{<:BiArnoldiFactorization{<:Any,S}}) where {S} = S

basis(F::BiArnoldiFactorization) = (F.V, F.W)
function rayleighquotient(F::BiArnoldiFactorization)
    return (PackedHessenberg(F.H, F.k), PackedHessenberg(F.K, F.k))
end
residual(F::BiArnoldiFactorization) = (F.rV, F.rW)
@inbounds normres(F::BiArnoldiFactorization) = (abs(F.H[end]), abs(F.K[end]))
rayleighextension(F::BiArnoldiFactorization) = SimpleBasisVector(F.k, F.k)

struct BiArnoldiIterator{F,T,O<:Orthogonalizer} <: KrylovIterator{F,T}
    operator::F
    v₀::T
    w₀::T
    orth::O
end
BiArnoldiIterator(f, v₀, w₀) = BiArnoldiIterator(f, v₀, w₀, KrylovDefaults.orth)

Base.IteratorSize(::Type{<:BiArnoldiIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:BiArnoldiIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::BiArnoldiIterator)
    state = initialize(iter)
    return state, state
end
function Base.iterate(iter::BiArnoldiIterator, state)
    nr = normres(state)
    if nr < eps(typeof(nr))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end

function _initializebasis(operation, op, x₀, orth;
                          verbosity::Int=KrylovDefaults.verbosity[])
    # initialize without using eltype

    β₀ = norm(x₀)
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    Ax₀ = operation(op, x₀)
    α = inner(x₀, Ax₀) / (β₀ * β₀)
    T = typeof(α) # scalar type of the Rayleigh quotient
    # this line determines the vector type that we will henceforth use
    # vector scalar type can be different from `T`, e.g. for real inner products
    v = add!!(scale(Ax₀, zero(α)), x₀, 1 / β₀)
    if typeof(Ax₀) != typeof(v)
        r = add!!(zerovector(v), Ax₀, 1 / β₀)
    else
        r = scale!!(Ax₀, 1 / β₀)
    end
    βold = norm(r)
    r = add!!(r, v, -α)
    β = norm(r)
    # possibly reorthogonalize
    if orth isa Union{ClassicalGramSchmidt2,ModifiedGramSchmidt2}
        dα = inner(v, r)
        α += dα
        r = add!!(r, v, -dα)
        β = norm(r)
    elseif orth isa Union{ClassicalGramSchmidtIR,ModifiedGramSchmidtIR}
        while eps(one(β)) < β < orth.η * βold
            βold = β
            dα = inner(v, r)
            α += dα
            r = add!!(r, v, -dα)
            β = norm(r)
        end
    end
    V = OrthonormalBasis([v])
    H = T[α, β]
    if verbosity > EACHITERATION_LEVEL
        @info "Arnoldi initiation at dimension 1: subspace normres = $(normres2string(β))"
    end

    return V, H, r
end

function initialize(iter::BiArnoldiIterator; verbosity::Int=KrylovDefaults.verbosity[])
    V, H, rv = _initializebasis(apply_normal, iter.operator, iter.v₀, iter.orth;
                                verbosity)
    W, K, rw = _initializebasis(apply_adjoint, iter.operator, iter.w₀, iter.orth;
                                verbosity)

    return BiArnoldiFactorization(1, V, W, H, K, rv, rw)
end

function _initializebasis!(operation, op, x₀, V, H, orth; verbosity)
    while length(V) > 1
        pop!(V)
    end
    H = empty!(H)

    V[1] = scale!!(V[1], x₀, 1 / norm(x₀))
    w = operation(op, V[1])
    r, α = orthogonalize!!(w, V[1], orth)
    β = norm(r)
    # state.k = 1
    push!(H, α, β)
    # state.r = r
    if verbosity > EACHITERATION_LEVEL
        @info "Arnoldi initiation at dimension 1: subspace normres = $(normres2string(β))"
    end

    return r
end

function initialize!(iter::BiArnoldiIterator, state::BiArnoldiFactorization;
                     verbosity::Int=KrylovDefaults.verbosity[])
    state.rV = _initializebasis!(apply_normal, iter.operator, iter.v₀, iter.V, iter.H,
                                 iter.orth;
                                 verbosity)
    state.rW = _initializebasis!(apply_adjoint, iter.operator, iter.w₀, iter.W,
                                 iter.K, iter.orth;
                                 verbosity)

    return state
end

function _expand!(operation, op, k, V, H, r, β, orth; verbosity)
    push!(V, scale(r, 1 / β))
    m = length(H)
    resize!(H, m + k + 1)
    r, β = biarnoldirecurrence!!(operation, op, V, view(H, (m + 1):(m + k)), orth)
    H[m + k + 1] = β
    # state.r = r
    if verbosity > EACHITERATION_LEVEL
        @info "Arnoldi expansion to dimension $k: subspace normres = $(normres2string(β))"
    end

    return r
end

function biarnoldirecurrence!!(operation, op,
                               V::OrthonormalBasis,
                               h::AbstractVector,
                               orth::Orthogonalizer)
    w = operation(op, last(V))
    r, h = orthogonalize!!(w, V, h, orth)
    return r, norm(r)
end

function expand!(iter::BiArnoldiIterator, state::BiArnoldiFactorization;
                 verbosity::Int=KrylovDefaults.verbosity[])
    state.k += 1

    βv, βw = normres(state)
    state.rV = _expand!(apply_normal, iter.operator, state.k, state.V, state.H, state.rV,
                        βv, iter.orth;
                        verbosity)
    state.rW = _expand!(apply_adjoint, iter.operator, state.k, state.W, state.K,
                        state.rW, βw, iter.orth;
                        verbosity)

    return state
end

function _shrink!(V, H, k)
    while length(V) > k + 1
        pop!(V)
    end
    r = pop!(V)
    resize!(H, (k * k + 3 * k) >> 1)

    return r
end

function shrink!(state::BiArnoldiFactorization, k;
                 verbosity::Int=KrylovDefaults.verbosity[])
    length(state) <= k && return state

    state.k = k
    rV = _shrink!(state.V, state.H, k)
    rW = _shrink!(state.W, state.K, k)

    βv, βw = normres(state)
    if verbosity > EACHITERATION_LEVEL
        @info "Arnoldi reduction to dimension $k: subspace normres v = $(normres2string(βv)), w = $(normres2string(βw))"
    end
    state.rV = scale!!(rV, βv)
    state.rW = scale!!(rW, βw)

    return state
end
