abstract AbstractKrylovFactorization{T}

type KrylovFactorization{T, M<:AbstractMatrix} <: AbstractKrylovFactorization{T}
    k::Int # current Krylov dimension
    V::Orthonormal{T} # orthonormal basis of length k+1
    H::M # matrix of size (m+1, m) with m the maximal Krylov dimension
end
