Base.@deprecate(RecursiveVec(args...), tuple(args...))

Base.@deprecate(basis(F::GKLFactorization, which::Symbol), basis(F, Val(which)))

import LinearAlgebra: mul!
Base.@deprecate(mul!(y, b::OrthonormalBasis, x::AbstractVector), unproject!!(y, b, x))


Base.@deprecate(eigsolve(A::AbstractMatrix, howmany::Int, which::Selector, T::Type;
                         kwargs...),
                eigsolve(A, Random.rand!(similar(A, T, size(A, 1))), howmany, which;
                         kwargs...),
                false)

Base.@deprecate(eigsolve(f, n::Int, howmany::Int, which::Selector, T::Type; kwargs...),
                eigsolve(f, Random.rand(T, n), howmany, which; kwargs...),
                false)

