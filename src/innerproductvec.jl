struct InnerProductVec{F,T}
    vec::T
    dotf::F
end

Base.:-(v::InnerProductVec) = InnerProductVec(-v.vec, v.dotf)
Base.:+(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F} =
    InnerProductVec(v.vec + w.vec, v.dotf)
Base.:-(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F} =
    InnerProductVec(v.vec - w.vec, v.dotf)
Base.:*(v::InnerProductVec, a::Number) = InnerProductVec(v.vec * a, v.dotf)
Base.:*(a::Number, v::InnerProductVec) = InnerProductVec(a * v.vec, v.dotf)
Base.:/(v::InnerProductVec, a::Number) = InnerProductVec(v.vec / a, v.dotf)
Base.:\(a::Number, v::InnerProductVec) = InnerProductVec(a \ v.vec, v.dotf)

Base.similar(v::InnerProductVec, ::Type{T} = eltype(v)) where {T} =
    InnerProductVec(similar(v.vec), v.dotf)

Base.getindex(v::InnerProductVec) = v.vec

function Base.copy!(w::InnerProductVec{F}, v::InnerProductVec{F}) where {F}
    copy!(w.vec, v.vec)
    return w
end

function LinearAlgebra.mul!(
    w::InnerProductVec{F},
    a::Number,
    v::InnerProductVec{F}
) where {F}
    mul!(w.vec, a, v.vec)
    return w
end

function LinearAlgebra.mul!(
    w::InnerProductVec{F},
    v::InnerProductVec{F},
    a::Number
) where {F}
    mul!(w.vec, v.vec, a)
    return w
end

function LinearAlgebra.rmul!(v::InnerProductVec, a::Number)
    rmul!(v.vec, a)
    return v
end

function LinearAlgebra.axpy!(
    a::Number,
    v::InnerProductVec{F},
    w::InnerProductVec{F}
) where {F}
    axpy!(a, v.vec, w.vec)
    return w
end
function LinearAlgebra.axpby!(
    a::Number,
    v::InnerProductVec{F},
    b,
    w::InnerProductVec{F}
) where {F}
    axpby!(a, v.vec, b, w.vec)
    return w
end

LinearAlgebra.dot(v::InnerProductVec{F}, w::InnerProductVec{F}) where {F} =
    v.dotf(v.vec, w.vec)
LinearAlgebra.norm(v::InnerProductVec) = sqrt(real(dot(v, v)))
