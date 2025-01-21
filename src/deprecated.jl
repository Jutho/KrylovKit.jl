Base.@deprecate(RecursiveVec(args...), tuple(args...))

Base.@deprecate(basis(F::GKLFactorization, which::Symbol), basis(F, Val(which)))

import LinearAlgebra: mul!
Base.@deprecate(mul!(y, b::OrthonormalBasis, x::AbstractVector), unproject!!(y, b, x))
