struct InnerProductVec{F,T}
    vec::T
    vecdotf::F
end
Base.eltype(v::InnerProductVec) = eltype(typeof(v))
Base.eltype(::Type{InnerProductVec{F,T}}) where {F,T} = eltype(T)

Base.:-(v::InnerProductVec) = InnerProductVec(-v.vec, v.vecdotf)
Base.:+(v::InnerProductVec{F}, w::InnerProductVec{F}) where F = InnerProductVec(v.vec+w.vec, v.vecdotf)
Base.:-(v::InnerProductVec{F}, w::InnerProductVec{F}) where F = InnerProductVec(v.vec-w.vec, v.vecdotf)
Base.:*(v::InnerProductVec, a) = InnerProductVec(v.vec*a, v.vecdotf)
Base.:*(a, v::InnerProductVec) = InnerProductVec(a*v.vec, v.vecdotf)
Base.:/(v::InnerProductVec, a) = InnerProductVec(v.vec/a, v.vecdotf)
Base.:\(a, v::InnerProductVec) = InnerProductVec(a\v.vec, v.vecdotf)

Base.similar(v::InnerProductVec, ::Type{T} = eltype(v)) where {T} = InnerProductVec(similar(v.vec), v.vecdotf)

function Base.fill!(v::InnerProductVec, a)
    fill!(v.vec, a)
    return v
end

function Base.copy!(w::InnerProductVec{F}, v::InnerProductVec{F}) where F
    copy!(w.vec, v.vec)
    return w
end

function Base.scale!(w::InnerProductVec{F}, a, v::InnerProductVec{F}) where F
    scale!(w.vec, a, v.vec)
    return w
end

function Base.scale!(w::InnerProductVec{F}, v::InnerProductVec{F}, a) where F
    scale!(w.vec, v.vec, a)
    return w
end

function Base.scale!(v::InnerProductVec, a)
    scale!(v.vec, a)
    return v
end

function LinAlg.axpy!(a, v::InnerProductVec{F}, w::InnerProductVec{F}) where F
    LinAlg.axpy!(a, v.vec, w.vec)
    return w
end

LinAlg.vecdot(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F} = v.vecdotf(v.vec, w.vec)
LinAlg.vecnorm(v::InnerProductVec) = sqrt(real(LinAlg.vecdot(v,v)))

function LinAlg.vecnorm(v::InnerProductVec, p::Real)
    p == 2 || error("can only compute 2-norm for vectors with custom inner product")
    return LinAlg.vecnorm(v)
end
