# Definition of a symplectic (Darboux) basis
"""
    SymplecticBasis{T} <: Basis{T}

A list of vector like objects of type `T` that form a symplectic (Darboux) basis with
respect to a skew-symmetric bilinear form `ω`. A symplectic basis satisfies the relations
`ω(u_{2m-1}, u_{2n}) = δ_{mn}`, `ω(u_{2m}, u_{2n}) = 0`, and `ω(u_{2m-1}, u_{2n-1}) = 0`.
See also [`Basis`](@ref).

Skew-orthonormality of the vectors contained in an instance `b` of `SymplecticBasis` is not
checked when elements are added; it is up to the algorithm that constructs `b` to guarantee
skew-orthonormality.

Vectors are added in pairs: odd-indexed vectors `u_{2m-1}` and their symplectic partners
`u_{2m}`. The function [`skeworthogonalize`](@ref) or [`skeworthonormalize`](@ref) can be
used to skew-orthogonalize a new vector with respect to the existing basis. These functions
require a skew-symmetric form `ω` to be defined for the vector type via
[`symplecticform`](@ref).

# Symplectic form requirements

To use a `SymplecticBasis` and the associated [`SkewOrthogonalizer`](@ref) algorithms with
your vector type, you must ensure that [`symplecticform(v, w)`](@ref symplecticform) is
defined. There are two approaches:

1. **Define `KrylovKit.symplecticform(v, w)`** for your vector type. A default
   implementation is provided for `AbstractVector` types, computing the canonical symplectic
   form `ω(v, w) = vᵀ J w = Σᵢ (v[2i-1] w[2i] - v[2i] w[2i-1])`.

2. **Wrap your vectors** in [`SymplecticFormVec(v, skewf)`](@ref SymplecticFormVec), which
   carries a custom symplectic form function `skewf`.

See also [`skeworthogonalize`](@ref), [`skeworthonormalize`](@ref),
[`SkewOrthogonalizer`](@ref), [`symplecticform`](@ref).
"""
struct SymplecticBasis{T} <: Basis{T}
    basis::Vector{T}
end
SymplecticBasis{T}() where {T} = SymplecticBasis{T}(Vector{T}(undef, 0))

# Iterator methods for SymplecticBasis
Base.IteratorSize(::Type{<:SymplecticBasis}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SymplecticBasis}) = Base.HasEltype()

Base.length(b::SymplecticBasis) = length(b.basis)
Base.eltype(b::SymplecticBasis{T}) where {T} = T

Base.iterate(b::SymplecticBasis) = Base.iterate(b.basis)
Base.iterate(b::SymplecticBasis, state) = Base.iterate(b.basis, state)

Base.getindex(b::SymplecticBasis, i) = getindex(b.basis, i)
Base.setindex!(b::SymplecticBasis, i, q) = setindex!(b.basis, i, q)
Base.firstindex(b::SymplecticBasis) = firstindex(b.basis)
Base.lastindex(b::SymplecticBasis) = lastindex(b.basis)

Base.first(b::SymplecticBasis) = first(b.basis)
Base.last(b::SymplecticBasis) = last(b.basis)

Base.popfirst!(b::SymplecticBasis) = popfirst!(b.basis)
Base.pop!(b::SymplecticBasis) = pop!(b.basis)
Base.push!(b::SymplecticBasis{T}, q::T) where {T} = (push!(b.basis, q); return b)
Base.empty!(b::SymplecticBasis) = (empty!(b.basis); return b)
Base.sizehint!(b::SymplecticBasis, k::Int) = (sizehint!(b.basis, k); return b)
Base.resize!(b::SymplecticBasis, k::Int) = (resize!(b.basis, k); return b)

numpairs(b::SymplecticBasis) = div(length(b), 2)

# Projection using the symplectic form for SymplecticBasis
"""
    skewproject!!(y::AbstractVector, b::SymplecticBasis, x,
        [α::Number = 1, β::Number = 0, r = Base.OneTo(2 * numpairs(b))])

Project the vector `x` onto the symplectic basis `b` using the symplectic form
[`symplecticform`](@ref). The index range `r` must consist of complete symplectic pairs,
i.e. `length(r)` must be even. The projection coefficients are computed using the
symplectic partner of each output index, with appropriate signs dictated by the Darboux
basis structure:

- for odd `j`: `y[j] = β*y[j] - α * symplecticform(b[r[j+1]], x)`
- for even `j`: `y[j] = β*y[j] + α * symplecticform(b[r[j-1]], x)`

That is, each coefficient is computed from the symplectic form with the *partner* basis
vector (even partner for odd indices, odd partner for even indices), with a sign flip for
odd indices. This cross-partner convention ensures correct skew-orthogonal projection in a
symplectic basis where `ω(u_{2m-1}, u_{2m}) = 1`.
"""
function skewproject!!(
        y::AbstractVector, b::SymplecticBasis, x,
        α::Number = true, β::Number = false, r = Base.OneTo(2 * numpairs(b))
    )
    length(y) == length(r) || throw(DimensionMismatch())
    iseven(length(r)) || throw(ArgumentError("index range `r` must cover complete symplectic pairs"))
    if get_num_threads() > 1
        @sync for J in splitrange(1:length(r), get_num_threads())
            Threads.@spawn for j in $J
                @inbounds begin
                    i = isodd(j) ? r[j + 1] : r[j - 1]
                    σ = isodd(j) ? -1 : 1
                    if β == 0
                        y[j] = α * σ * symplecticform(b[i], x)
                    else
                        y[j] = β * y[j] + α * σ * symplecticform(b[i], x)
                    end
                end
            end
        end
    else
        for j in 1:length(r)
            @inbounds begin
                i = isodd(j) ? r[j + 1] : r[j - 1]
                σ = isodd(j) ? -1 : 1
                if β == 0
                    y[j] = α * σ * symplecticform(b[i], x)
                else
                    y[j] = β * y[j] + α * σ * symplecticform(b[i], x)
                end
            end
        end
    end
    return y
end

# Skew-orthogonalization of a vector against a given SymplecticBasis
skeworthogonalize(v, args...) = skeworthogonalize!!(scale(v, true), args...)

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, alg::SkewOrthogonalizer
    ) where {T}
    S = promote_type(scalartype(v), scalartype(T))
    c = Vector{S}(undef, length(b))
    return skeworthogonalize!!(v, b, c, alg)
end

# See pages 3-4 of https://people.math.ethz.ch/%7Eacannas/Papers/lsg.pdf
function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ClassicalSymplecticGramSchmidt
    ) where {T}
    np = numpairs(b)
    idx_pairs = 1:2np
    skewproject!!(view(x, idx_pairs), b, v, 1, 0, idx_pairs)
    if isodd(length(b))
        if alg.esr == ESR2
            x[2np + 1] = inner(last(b), v)
        else
            x[2np + 1] = zero(eltype(x))
        end
    end
    v = unproject!!(v, b, x, -1, 1)
    return (v, x)
end

function reskeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ClassicalSymplecticGramSchmidt,
        s::AbstractVector = similar(x)
    ) where {T}
    (v, s) = skeworthogonalize!!(v, b, s, alg)
    x .+= s
    return (v, x)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ClassicalSymplecticGramSchmidt2
    ) where {T}
    csgs = ClassicalSymplecticGramSchmidt(alg.esr)
    (v, x) = skeworthogonalize!!(v, b, x, csgs)
    return reskeworthogonalize!!(v, b, x, csgs)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ClassicalSymplecticGramSchmidtIR
    ) where {T}
    csgs = ClassicalSymplecticGramSchmidt(alg.esr)
    nold = norm(v)
    (v, x) = skeworthogonalize!!(v, b, x, csgs)
    nnew = norm(v)
    if eps(one(nnew)) < nnew < alg.η * nold
        s = similar(x) # scratch storage shared by all reorthogonalization passes
        while eps(one(nnew)) < nnew < alg.η * nold
            nold = nnew
            (v, x) = reskeworthogonalize!!(v, b, x, csgs, s)
            nnew = norm(v)
        end
    end
    return (v, x)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ModifiedSymplecticGramSchmidt
    ) where {T}
    np = numpairs(b)
    for m in 1:np
        i_odd = 2m - 1
        i_even = 2m
        h_e = symplecticform(b[i_odd], v)
        h_f = symplecticform(b[i_even], v)
        x[i_odd] = -h_f
        x[i_even] = h_e
        v = add!!(v, b[i_odd], h_f)
        v = add!!(v, b[i_even], -h_e)
    end
    if isodd(length(b))
        if alg.esr == ESR2
            x[2np + 1] = inner(last(b), v)
            v = add!!(v, last(b), -x[2np + 1])
        else
            x[2np + 1] = zero(eltype(x))
        end
    end
    return (v, x)
end

function reskeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ModifiedSymplecticGramSchmidt
    ) where {T}
    np = numpairs(b)
    for m in 1:np
        i_odd = 2m - 1
        i_even = 2m
        h_e = symplecticform(b[i_odd], v)
        h_f = symplecticform(b[i_even], v)
        x[i_odd] -= h_f
        x[i_even] += h_e
        v = add!!(v, b[i_odd], h_f)
        v = add!!(v, b[i_even], -h_e)
    end
    if isodd(length(b)) && alg.esr == ESR2
        r11 = inner(last(b), v)
        x[2np + 1] += r11
        v = add!!(v, last(b), -r11)
    end
    return (v, x)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ModifiedSymplecticGramSchmidt2
    ) where {T}
    msgs = ModifiedSymplecticGramSchmidt(alg.esr)
    (v, x) = skeworthogonalize!!(v, b, x, msgs)
    return reskeworthogonalize!!(v, b, x, msgs)
end

function skeworthogonalize!!(
        v::T, b::SymplecticBasis{T}, x::AbstractVector, alg::ModifiedSymplecticGramSchmidtIR
    ) where {T}
    msgs = ModifiedSymplecticGramSchmidt(alg.esr)
    nold = norm(v)
    (v, x) = skeworthogonalize!!(v, b, x, msgs)
    nnew = norm(v)
    while eps(one(nnew)) < nnew < alg.η * nold
        nold = nnew
        (v, x) = reskeworthogonalize!!(v, b, x, msgs)
        nnew = norm(v)
    end
    return (v, x)
end

# Skew-orthonormalization: skew-orthogonalization and normalization
# For odd vectors: normalize with standard norm
# For even vectors: scale so that ω(partner, v) = 1
skeworthonormalize(v, args...) = skeworthonormalize!!(scale(v, VectorInterface.One()), args...)

function skeworthonormalize!!(v, b::SymplecticBasis, x::AbstractVector, alg::SkewOrthogonalizer)
    out = skeworthogonalize!!(v, b, x, alg)
    # rebind to the returned vector, as `!!` methods may return a new object
    v = first(out)
    if iseven(length(b))
        # Adding odd vector: normalize with standard norm / don't normalize if ESR3m
        β = alg.esr == ESR3m ? one(scalartype(v)) : norm(v)
    else
        # Adding even vector: scale so that ω(partner, v) = 1
        # The partner is the last vector in b (the odd vector of the current pair)
        β = symplecticform(last(b), v)
    end
    v = scale!!(v, inv(β))
    return (v, β, Base.tail(out)...)
end

"""
    skeworthogonalize(v, b::SymplecticBasis, [x::AbstractVector,] alg::SkewOrthogonalizer) -> w, x
    skeworthogonalize!!(v, b::SymplecticBasis, [x::AbstractVector,] alg::SkewOrthogonalizer) -> w, x

Skew-orthogonalize vector `v` against all the vectors in the symplectic basis `b` using the
skew-orthogonalization algorithm `alg` of type [`SkewOrthogonalizer`](@ref), and return the
resulting vector `w` and the overlap coefficients `x` of `v` with the basis vectors in `b`.

The skew-orthogonalization uses the symplectic form [`symplecticform`](@ref), which is
expected to satisfy `ω(u, v) = -ω(v, u)`. For vectors wrapped in
[`SymplecticFormVec`](@ref), a custom form function can be provided. For a symplectic basis
with pairs `(u_{2m-1}, u_{2m})` where `ω(u_{2m-1}, u_{2m}) = 1`, the
skew-orthogonalization ensures:
- `ω(u_{2m-1}, w) = 0` for all `m`
- `ω(u_{2m}, w) = 0` for all `m`

When the basis has odd length (i.e. after an odd vector has been added but before its even
partner), the behavior depends on the [`ESR`](@ref) variant of the algorithm:
- `ESR1`: no additional projection is performed against the unpaired odd vector.
- `ESR2`: additionally projects `v` against the unpaired odd vector using the standard
  inner product, i.e. `x[end] = inner(last(b), v)` and `w = v - x[end] * last(b)`. This
  makes `s₁` and `s₂` orthogonal and tends to improve numerical stability.
- `ESR3m`: same as `ESR1` (no additional projection).

See [Salam (2014)](https://journal.austms.org.au/ojs/index.php/ANZIAMJ/article/view/9380/1920)
for details on the elementary symplectic reflections and their effect on stability.

In case of `skeworthogonalize!!`, the vector `v` is mutated in place. In both functions,
storage for the overlap coefficients `x` can be provided as optional argument
`x::AbstractVector` with `length(x) >= length(b)`.

Note that `w` is not normalized, see also [`skeworthonormalize`](@ref).

For more information on possible skew-orthogonalization algorithms, see
[`SkewOrthogonalizer`](@ref) and its concrete subtypes
[`ClassicalSymplecticGramSchmidt`](@ref), [`ModifiedSymplecticGramSchmidt`](@ref),
[`ClassicalSymplecticGramSchmidt2`](@ref), [`ModifiedSymplecticGramSchmidt2`](@ref),
[`ClassicalSymplecticGramSchmidtIR`](@ref) and [`ModifiedSymplecticGramSchmidtIR`](@ref).
"""
skeworthogonalize, skeworthogonalize!!

"""
    skeworthonormalize(v, b::SymplecticBasis, [x::AbstractVector,] alg::SkewOrthogonalizer) -> w, β, x
    skeworthonormalize!!(v, b::SymplecticBasis, [x::AbstractVector,] alg::SkewOrthogonalizer) -> w, β, x

Skew-orthonormalize vector `v` against all the vectors in the symplectic basis `b` using
the skew-orthogonalization algorithm `alg` of type [`SkewOrthogonalizer`](@ref).

The normalization depends on the current length of the basis and the [`ESR`](@ref) variant
of the algorithm. Vectors are added in pairs `(s₁, s₂)` via an elementary symplectic
reflection (ESR) factorization of `(x₁, x₂)` (the input vectors before and after
skew-orthogonalization). The scaling factors `r₁₁` and `r₂₂` in the ESR depend on the
variant:

- **`ESR1`**: `r₁₁ = ‖x₁‖`, `r₁₂ = 0`, `r₂₂ = ω(s₁, x₂)`. The odd vector `s₁` is
  normalized to unit norm.
- **`ESR2`**: `r₁₁ = ‖x₁‖`, `r₁₂ = s₁ᵀ x₂`, `r₂₂ = ω(s₁, y)` where
  `y = x₂ - r₁₂ s₁`. The odd vector `s₁` is normalized to unit norm, and the even
  vector is additionally projected against `s₁` using the standard inner product before
  symplectic scaling, improving stability.
- **`ESR3m`**: `r₁₁ = 1` (no norm scaling), `r₁₂ = 0`, `r₂₂ = ω(s₁, x₂)`. The odd
  vector `s₁ = x₁` is used without normalization; `r₁₁` is recorded as
  `one(scalartype(v))` rather than `‖x₁‖`.

See [Salam (2014)](https://journal.austms.org.au/ojs/index.php/ANZIAMJ/article/view/9380/1920)
for details on the elementary symplectic reflections.

**When `length(b)` is even** (adding an odd vector `s₁`): returns the resulting vector `w`
normalized to unit norm (`‖w‖ = 1`) for `ESR1`/`ESR2`, or unnormalized for `ESR3m`. The
scaling factor `β = r₁₁` is returned.

**When `length(b)` is odd** (adding an even vector `s₂`): returns the resulting vector `w`
scaled such that `ω(last(b), w) = 1`, where `last(b)` is the odd partner of the pair.
The scaling factor `β = r₂₂ = ω(last(b), v)` after skew-orthogonalizing is returned.

In case of `skeworthonormalize!!`, the vector `v` is mutated in place. In both functions,
storage for the overlap coefficients `x` can be provided as optional argument
`x::AbstractVector` with `length(x) >= length(b)`.

See [`skeworthogonalize`](@ref) if `w` does not need to be normalized.

For more information on possible skew-orthogonalization algorithms, see
[`SkewOrthogonalizer`](@ref) and its concrete subtypes
[`ClassicalSymplecticGramSchmidt`](@ref), [`ModifiedSymplecticGramSchmidt`](@ref),
[`ClassicalSymplecticGramSchmidt2`](@ref), [`ModifiedSymplecticGramSchmidt2`](@ref),
[`ClassicalSymplecticGramSchmidtIR`](@ref) and [`ModifiedSymplecticGramSchmidtIR`](@ref).
"""
skeworthonormalize, skeworthonormalize!!
