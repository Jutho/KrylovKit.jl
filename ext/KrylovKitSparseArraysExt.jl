module KrylovKitSparseArraysExt

using SparseArrays
using KrylovKit
using VectorInterface: scalartype

KrylovKit.initialize_vector(A::SparseMatrixCSC) = rand(scalartype(A), size(A, 1))

end
