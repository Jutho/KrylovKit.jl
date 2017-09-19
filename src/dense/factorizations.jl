function qr!(A::AbstractMatrix, U = novecs)
    m,n = size(A)
    for k in 1:m
        h, ν = householder(A, k:m, k)
        lmul!(A, h, k+1:n)
        rmulc!(U, h)
        A[k,k] = ν
        @inbounds for j=k+1:m
            A[j,k] = 0
        end
    end
    return A, U
end

function hessenberg!(A::AbstractMatrix, U = novecs)
    n = checksquare(A)
    for k = 1:n-1
        h, ν = householder(A,k+1:n,k)
        A[k+1,k] = ν
        @inbounds for j=k+2:n
            A[j,k] = 0
        end
        lmul!(A, h, k+1:n)
        rmulc!(A, h)
        rmulc!(U, h)
    end
    return A, U
end

schur!(A::AbstractMatrix, U = novecs) = hschur!(hessenberg!(A, U)...)
