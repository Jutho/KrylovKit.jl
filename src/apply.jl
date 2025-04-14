@doc """
    apply(operator, x)

Apply the operator `operator` to the vector `x`, returning the result.
By default this falls back to `operator * x` for `AbstractMatrix`, and `f(x)` in other cases.
""" apply

apply(A::AbstractMatrix, x::AbstractVector) = A * x
apply(f, x) = f(x)

function apply(operator, x, α₀, α₁)
    y = apply(operator, x)
    if α₀ != zero(α₀) || α₁ != one(α₁)
        y = add!!(y, x, α₀, α₁)
    end

    return y
end

# GKL, SVD, LSMR
@doc """
    apply_normal(operator, x)

Apply the operator to the vector `x`, returning the result.
By default this falls back to `operator * x` for `AbstractMatrix`, `operator[1](x)` when
the input is a tuple of functions, and `operator(x, Val(false))` for other cases.
""" apply_normal

@doc """
    apply_adjoint(operator, x)

Apply the adjoint of the operator to the vector `x`, returning the result.
By default this falls back to `operator' * x` for `AbstractMatrix`, `operator[2](x)` when
the input is a tuple of functions, and `operator(x, Val(true))` for other cases.
""" apply_adjoint

apply_normal(A::AbstractMatrix, x::AbstractVector) = A * x
apply_adjoint(A::AbstractMatrix, x::AbstractVector) = A' * x
apply_normal((f, fadjoint)::Tuple{Any,Any}, x) = f(x)
apply_adjoint((f, fadjoint)::Tuple{Any,Any}, x) = fadjoint(x)
apply_normal(f, x) = f(x, Val(false))
apply_adjoint(f, x) = f(x, Val(true))

# generalized eigenvalue problem
genapply((A, B)::Tuple{Any,Any}, x) = (apply(A, x), apply(B, x))
genapply(f, x) = f(x)
