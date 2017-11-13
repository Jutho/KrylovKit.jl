eigsolve(A::AbstractMatrix, howmany::Int = 1, which::Symbol = :LM, T::Type = eltype(A); issymmetric = issymmetric(A), ishermitian = ishermitian(A), method = nothing) =
    eigsolve(x->(A*x), size(A,1), howmany, which, T; issymmetric = issymmetric, ishermitian = ishermitian, method)

eigsolve(A::AbstractMatrix, x₀::VecOrMat, howmany::Int = 1, which::Symbol = :LM, T::Type = promote_type(eltype(A), eltype(x₀)); issymmetric = issymmetric(A), ishermitian = ishermitian(A), method = nothing) =
    eigsolve(x->(A*x), x₀, howmany, which, T; issymmetric = issymmetric, ishermitian = ishermitian, method)

eigsolve(f, n::Int, howmany::Int = 1, which::Symbol = :LM, T::Type = Float64; kwargs...) =
    eigsolve(f, rand(T, n), howmany, which; kwargs...)

function eigsolve(f, x₀, howmany::Int = 1, which::Symbol = :LM, T::Type = eltype(x₀); issymmetric = false, ishermitian = T<:Real && issymmetric, method = nothing)
    x = eltype(x₀) == T ? x₀ : copy!(similar(x₀, T), x₀)
    if method != nothing
        return eigsolve(f, x, howmany, which, method)
    end
    if T<:Real
        (which == :LI || which == :SI) && throw(ArgumentError("work in complex domain to find eigenvalues with largest or smallest imaginary part"))
    end
    if (T<:Real && issymmetric) || ishermitian
        return eigsolve(f, x, howmany, which, Lanczos(ImplicitRestart(100)))
    else
        return eigsolve(f, x, howmany, which, Arnoldi(ImplicitRestart(100)))
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
