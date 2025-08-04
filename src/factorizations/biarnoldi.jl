mutable struct BiArnoldiFactorization{T, S} <: KrylovFactorization{T, S}
    VH::ArnoldiFactorization{T, S}
    WK::ArnoldiFactorization{T, S}
end

Base.length(F::BiArnoldiFactorization) = length(F.VH)
Base.sizehint!(F::BiArnoldiFactorization, n) = begin
    sizehint!(F.VH, n)
    sizehint!(F.WK, n)
    return F
end
Base.eltype(F::BiArnoldiFactorization) = eltype(typeof(F))
Base.eltype(::Type{<:BiArnoldiFactorization{<:Any, S}}) where {S} = S

basis(F::BiArnoldiFactorization) = basis.((F.VH, F.WK))
rayleighquotient(F::BiArnoldiFactorization) = rayleighquotient.((F.VH, F.WK))
residual(F::BiArnoldiFactorization) = residual.((F.VH, F.WK))
normres(F::BiArnoldiFactorization) = normres.((F.VH, F.WK))
rayleighextension(F::BiArnoldiFactorization) = rayleighextension.((F.VH, F.WK))

struct BiArnoldiIterator{I1 <: ArnoldiIterator, I2 <: ArnoldiIterator}
    iterVH::I1
    iterWK::I2
    function BiArnoldiIterator(
            f::F, v₀, w₀, orth1::Orthogonalizer, orth2::Orthogonalizer
        ) where {F}
        iterVH = ArnoldiIterator(Base.Fix1(apply_normal, f), v₀, orth1)
        iterWK = ArnoldiIterator(Base.Fix1(apply_adjoint, f), w₀, orth2)
        I1 = typeof(iterVH)
        I2 = typeof(iterWK)
        return new{I1, I2}(iterVH, iterWK)
    end
end
function BiArnoldiIterator(f, v₀, w₀ = v₀, orth = KrylovDefaults.orth)
    return BiArnoldiIterator(f, v₀, w₀, orth, orth)
end

Base.IteratorSize(::Type{<:BiArnoldiIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:BiArnoldiIterator}) = Base.EltypeUnknown()

function Base.iterate(iter::BiArnoldiIterator)
    state = initialize(iter)
    return state, state
end
function Base.iterate(iter::BiArnoldiIterator, state)
    nr = normres(state)
    if minimum(nr) < eps(typeof(nr))
        return nothing
    else
        state = expand!(iter, deepcopy(state))
        return state, state
    end
end

function initialize(iter::BiArnoldiIterator; verbosity::Int = KrylovDefaults.verbosity[])
    VH = initialize(iter.iterVH; verbosity = verbosity)
    WK = initialize(iter.iterWK; verbosity = verbosity)
    return BiArnoldiFactorization(VH, WK)
end
function initialize!(
        iter::BiArnoldiIterator, state::BiArnoldiFactorization;
        verbosity::Int = KrylovDefaults.verbosity[]
    )
    state.VH = initialize!(iter.iterVH, state.VH; verbosity = verbosity)
    state.WK = initialize!(iter.iterWK, state.WK; verbosity = verbosity)
    return state
end
function expand!(
        iter::BiArnoldiIterator, state::BiArnoldiFactorization;
        verbosity::Int = KrylovDefaults.verbosity[]
    )
    state.VH = expand!(iter.iterVH, state.VH; verbosity = verbosity)
    state.WK = expand!(iter.iterWK, state.WK; verbosity = verbosity)
    return state
end
function shrink!(
        state::BiArnoldiFactorization, k;
        verbosity::Int = KrylovDefaults.verbosity[]
    )
    state.VH = shrink!(state.VH, k; verbosity = verbosity)
    state.WK = shrink!(state.WK, k; verbosity = verbosity)
    return state
end
