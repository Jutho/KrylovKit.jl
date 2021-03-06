struct MinimalVec{V<:AbstractVector}
    vec::V
end

Base.getindex(v::MinimalVec) = v.vec # for convience, should not interfere

# minimal interface according to docs
Base.:*(a::Number, v::MinimalVec) = MinimalVec(a*v[])

Base.similar(v::MinimalVec) = MinimalVec(similar(v[]))

LinearAlgebra.axpy!(α, v::MinimalVec, w::MinimalVec) = (axpy!(α, v[], w[]); return w)
LinearAlgebra.axpby!(α, v::MinimalVec, β, w::MinimalVec) =
    (axpby!(α, v[], β, w[]); return w)
LinearAlgebra.rmul!(v::MinimalVec, α) = (rmul!(v[], α); return v)

LinearAlgebra.mul!(w::MinimalVec, v::MinimalVec, α) = (mul!(w[], v[], α); return w)
LinearAlgebra.dot(v::MinimalVec, w::MinimalVec) = dot(v[], w[])
LinearAlgebra.norm(v::MinimalVec) = norm(v[])
