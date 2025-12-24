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
apply_normal(A::AbstractMatrix, x::AbstractVector) = A * x
apply_adjoint(A::AbstractMatrix, x::AbstractVector) = A' * x
apply_normal((f, fadjoint)::Tuple{Any, Any}, x) = f(x)
apply_adjoint((f, fadjoint)::Tuple{Any, Any}, x) = fadjoint(x)
apply_normal(f, x) = f(x, Val(false))
apply_adjoint(f, x) = f(x, Val(true))

# generalized eigenvalue problem
genapply((A, B)::Tuple{Any, Any}, x) = (apply(A, x), apply(B, x))
genapply(f, x) = f(x)

# attempt type inference first but fall back to actual values if failed
function apply_scalartype(f, x, as::Number...)
    0 <= length(as) <= 2 || throw(ArgumentError("unknown type of function application"))
    Tfx = Base.promote_op(apply, typeof(f), typeof(x), typeof.(as)...)
    if Tfx !== Union{} && Tfx !== Any
        T = Base.promote_op(inner, typeof(x), Tfx)
        T <: Number && return T
    end
    # if this is reached, type inference failed due to instability or error
    # retry in value domain - will also give better stacktrace
    return typeof(inner(x, apply(f, x, as...)))
end

function genapply_scalartype(f, x)
    Tfx = Base.promote_op(genapply, typeof(f), typeof(x))
    if Tfx <: Tuple
        @assert length(fieldtypes(Tfx)) == 2
        Tfx1, Tfx2 = fieldtypes(Tfx)
        T1 = Base.promote_op(inner, typeof(x), Tfx1)
        T2 = Base.promote_op(inner, typeof(x), Tfx2)
        T = promote_type(T1, T2)
        T <: Number && return T
    end
    # if this is reached, type inference failed due to instability or error
    # retry in value domain - will also give better stacktrace
    fx1, fx2 = genapply(f, x)
    α1 = inner(x, fx1)
    α2 = inner(x, fx2)
    return promote_type(typeof(α1), typeof(α2))
end
