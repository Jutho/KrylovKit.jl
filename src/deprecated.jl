Base.@deprecate(RecursiveVec(args...), tuple(args...))

Base.@deprecate(basis(F::GKLFactorization, which::Symbol), basis(F, Val(which)))
