apply(A::AbstractMatrix, x::AbstractVecOrMat) = A * x
apply(f, x) = f(x)

function apply(operator, x, α₀, α₁)
    y = apply(operator, x)
    if α₀ != zero(α₀) || α₁ != one(α₁)
        y = add!!(y, x, α₀, α₁)
    end

    return y
end

# GKL, SVD, LSMR
apply_normal(A::AbstractMatrix, x::AbstractVector) = A * x
apply_adjoint(A::AbstractMatrix, x::AbstractVector) = A' * x
apply_normal((f, fadjoint)::Tuple{Any,Any}, x) = f(x)
apply_adjoint((f, fadjoint)::Tuple{Any,Any}, x) = fadjoint(x)
apply_normal(f, x) = f(x, Val(false))
apply_adjoint(f, x) = f(x, Val(true))

# generalized eigenvalue problem
genapply((A, B)::Tuple{Any,Any}, x) = (apply(A, x), apply(B, x))
genapply(f, x) = f(x)
