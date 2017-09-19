# Left divide by an upper triangular matrix in-place.
# The computation is restricted to the contiguous block parameterized by the
# unitrange r
# Based on Julia base naivesub!

function utldiv!(R::AbstractMatrix, y::AbstractVector, r::UnitRange{Int} = 1:length(y))
    @inbounds for j in reverse(r)
        R[j,j] == zero(R[j,j]) && throw(SingularException(j))
        yj = (y[j] = R[j,j] \ y[j])
        @simd for i in first(r):j-1
            y[i] -= R[i,j] * yj
        end
    end
    return y
end
