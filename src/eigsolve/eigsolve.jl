struct ClosestTo{T}
    λ::T
end

const Selector = Union{ClosestTo, Symbol}

function eigsolve(A::AbstractMatrix, howmany::Int = 1, which::Selector = :LM, T::Type = eltype(A);
        issymmetric = issymmetric(A), ishermitian = ishermitian(A),
        krylovdim::Int = Defaults.krylovdim, maxiter::Int = Defaults.maxiter, tol::Real = Defaults.tol)
    eigsolve(x->(A*x), size(A,1), howmany, which, T; issymmetric = issymmetric, ishermitian = ishermitian, krylovdim = krylovdim, maxiter = maxiter, tol = tol)
end

function eigsolve(A::AbstractMatrix, x₀::VecOrMat, howmany::Int = 1, which::Selector = :LM, T::Type = promote_type(eltype(A), eltype(x₀));
        issymmetric = issymmetric(A), ishermitian = ishermitian(A),
        krylovdim::Int = Defaults.krylovdim, maxiter::Int = Defaults.maxiter, tol::Real = Defaults.tol)
    eigsolve(x->(A*x), x₀, howmany, which, T; issymmetric = issymmetric, ishermitian = ishermitian, krylovdim = krylovdim, maxiter = maxiter, tol = tol)
end

eigsolve(f, n::Int, howmany::Int = 1, which::Selector = :LM, T::Type = Float64; kwargs...) =
    eigsolve(f, rand(T, n), howmany, which; kwargs...)

function eigsolve(f, x₀, howmany::Int = 1, which::Selector = :LM, T::Type = eltype(x₀);
        issymmetric = false, ishermitian = T<:Real && issymmetric,
        krylovdim::Int = Defaults.krylovdim, maxiter::Int = Defaults.maxiter, tol::Real = Defaults.tol)
    x = eltype(x₀) == T ? x₀ : copyto!(similar(x₀, T), x₀)
    if T<:Real
        (which == :LI || which == :SI) && throw(ArgumentError("work in complex domain to find eigenvalues with largest or smallest imaginary part"))
    end
    if (T<:Real && issymmetric) || ishermitian
        return eigsolve(f, x, howmany, which, Lanczos(krylovdim = krylovdim, maxiter = maxiter, tol=tol))
    else
        return eigsolve(f, x, howmany, which, Arnoldi(krylovdim = krylovdim, maxiter = maxiter, tol=tol))
    end
end


function eigsort(which::ClosestTo)
    by = x->abs(x-which.λ)
    rev = false
    return by, rev
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
