eigsolve(A::AbstractMatrix, k::Int = 1, which::Symbol = :LM, T::Type = eltype(A); issymmetric = issymmetric(A), ishermitian = ishermitian(A), method = none) =
    eigsolve(x->(A*x), size(A,1), k, which, T; issymmetric = issymmetric, ishermitian = ishermitian, method)

eigsolve(A::AbstractMatrix, x0::VecOrMat, k::Int = 1, which::Symbol = :LM, T::Type = promote_type(eltype(A),eltype(x0)); issymmetric = issymmetric(A), ishermitian = ishermitian(A), method = none) =
    eigsolve(x->(A*x), x0, k, which, T; issymmetric = issymmetric, ishermitian = ishermitian, method)

eigsolve(f, n::Int, k::Int = 1, which::Symbol = :LM, T::Type = Float64; kwargs...) =
    eigsolve(f, rand(T, n), k, which; kwargs...)

function eigsolve(f, x0::VecOrMat, k::Int = 1, which::Symbol = :LM, T::Type = eltype(x0); issymmetric = false, ishermitian = issymmetric, method = none)
    if method != none
        return eigsolve(f, x0, k, which, method)
    end
    if T<:Real
        (which == :LI || which == :SI) && throw(ArgumentError("work in complex domain to find eigenvalues with largest or smallest imaginary part"))
    end
    if k == 1
        if (T<:Real && issymmetric) || ishermitian
            return eigsolve(f, x0, k, which, Lanczos(ExplicitRestart(100)))
        else
            return eigsolve(f, x0, k, which, Arnoldi(ExplicitRestart(100)))
        end
    else
        if (T<:Real && issymmetric) || ishermitian
            return eigsolve(f, x0, k, which, Lanczos(ImplicitRestart(100)))
        else
            return eigsolve(f, x0, k, which, Arnoldi(ImplicitRestart(100)))
        end
    end
end

function eigsort(which::Symbol)
    if which == :LM
        by = abs
        rev = true
    elseif which == :LR
        by = real
        rev = true
    elseif which == :SR
        by = real
        rev = false
    elseif which == :LI
        by = imag
        rev = true
    elseif which == :SI
        by = imag
        rev = false
    else
        error("incorrect value of which: $which")
    end
    return by, rev
end
