module TestSetup

export tolerance, ≊, MinimalVec, isinplace, stack
export wrapop, wrapvec, unwrapvec, buildrealmap
export relax_tol, mat_with_eigrepition

import VectorInterface as VI
using VectorInterface
using LinearAlgebra: LinearAlgebra

# Utility functions
# -----------------
"function for determining the precision of a type"
tolerance(T::Type{<:Number}) = eps(real(T))^(2 // 3)
relax_tol(T::Type{<:Number}) = eps(real(T))^(1 // 2)

"function for comparing sets of eigenvalues"
function ≊(list1::AbstractVector, list2::AbstractVector)
    length(list1) == length(list2) || return false
    n = length(list1)
    ind2 = collect(1:n)
    p = sizehint!(Int[], n)
    for i in 1:n
        j = argmin(abs.(view(list2, ind2) .- list1[i]))
        p = push!(p, ind2[j])
        ind2 = deleteat!(ind2, j)
    end
    return list1 ≈ view(list2, p)
end

function buildrealmap(A, B)
    function f(x)
        return A * x + B * conj(x)
    end
    function f(x, ::Val{C}) where {C}
        if C == false
            return A * x + B * conj(x)
        else
            return adjoint(A) * x + transpose(B) * conj(x)
        end
    end
    return f
end

"function for generating a matrix with repeated eigenvalues"
function mat_with_eigrepition(T, N, multiplicity)
    U = LinearAlgebra.qr(randn(T, (N, N))).Q # Haar random matrix
    D = sort(randn(real(T), N))
    i = 0
    while multiplicity >= 2 && (i + multiplicity) <= N ÷ 2
        D[i .+ (1:multiplicity)] .= D[i + 1]
        D[N + 1 - i .- (1:multiplicity)] .= D[N - i]
        i += multiplicity
        multiplicity -= 1
    end
    A = U * LinearAlgebra.Diagonal(D) * U'
    return (A + A') / 2
end

# Wrappers
# --------
using VectorInterface: MinimalSVec, MinimalMVec, MinimalVec
# dispatch on val is necessary for type stability

function wrapvec(v, ::Val{mode}) where {mode}
    return mode === :vector ? v :
        mode === :inplace ? MinimalMVec(v) :
        mode === :outplace ? MinimalSVec(v) :
        mode === :mixed ? MinimalSVec(v) :
        throw(ArgumentError("invalid mode ($mode)"))
end
function wrapvec2(v, ::Val{mode}) where {mode}
    return mode === :mixed ? MinimalMVec(v) : wrapvec(v, mode)
end

unwrapvec(v::MinimalVec) = v.vec
unwrapvec(v) = v

function wrapop(A, ::Val{mode}) where {mode}
    if mode === :vector
        return A
    elseif mode === :inplace || mode === :outplace
        return function (v, flag = Val(false))
            if flag === Val(true)
                return wrapvec(A' * unwrapvec(v), Val(mode))
            else
                return wrapvec(A * unwrapvec(v), Val(mode))
            end
        end
    elseif mode === :mixed
        return (
            x -> wrapvec(A * unwrapvec(x), Val(mode)),
            y -> wrapvec2(A' * unwrapvec(y), Val(mode)),
        )
    else
        throw(ArgumentError("invalid mode ($mode)"))
    end
end

if VERSION < v"1.9"
    stack(f, itr) = mapreduce(f, hcat, itr)
    stack(itr) = reduce(hcat, itr)
end

end
