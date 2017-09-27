import Base.LinAlg: BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare
import Base.BLAS: @blasfunc, libblas, BlasReal, BlasComplex
import Base.LAPACK: liblapack, chklapackerror


# Some modified wrappers for Lapack
function qr!(A::StridedMatrix{<:BlasFloat})
    m, n = size(A)
    A, T = LAPACK.geqrt!(A, min(minimum(size(A)), 36))
    Q = LAPACK.gemqrt!('L', 'N', A, T, eye(eltype(A), m, min(m,n)))
    R = triu!(A[1:min(m,n), :])
    return Q, R
end

# for some reason this is faster than LAPACKs triangular division
function ldiv!(A::UpperTriangular, y::AbstractVector, r::UnitRange{Int} = 1:length(y))
    R = A.data
    @inbounds for j in reverse(r)
        R[j,j] == zero(R[j,j]) && throw(SingularException(j))
        yj = (y[j] = R[j,j] \ y[j])
        @simd for i in first(r):j-1
            y[i] -= R[i,j] * yj
        end
    end
    return y
end
