module KrylovKitSparseArraysExt

using SparseArrays
using KrylovKit

function KrylovKit.eigsolve_init(A::SparseMatrixCSC, ::Type{T}=eltype(A)) where {T<:Number}
    return rand(T, size(A, 1))
end

end
