# Some modified wrappers for Lapack
import LinearAlgebra: BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare
import LinearAlgebra.BLAS: @blasfunc, libblas, BlasReal, BlasComplex
import LinearAlgebra.LAPACK: liblapack, chklapackerror

struct RowIterator{A<:AbstractMatrix,R<:IndexRange}
    a::A
    r::R
end
rows(a::AbstractMatrix, r::IndexRange = axes(a,1)) = RowIterator(a, r)

function Base.iterate(iter::RowIterator)
    next = iterate(iter.r)
    if next === nothing
        return nothing
    else
        i, s = next
        return view(iter.a, i, :), s
    end
end
function Base.iterate(iter::RowIterator, s)
    next = iterate(iter.r, s)
    if next === nothing
        return nothing
    else
        i, s = next
        return view(iter.a, i, :), s
    end
end

Base.IteratorSize(::Type{<:RowIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:RowIterator}) = Base.HasEltype()
Base.length(iter::RowIterator) = length(iter.r)
Base.eltype(iter::RowIterator{A}) where {T,A<:DenseArray{T}} = SubArray{T,1,A,Tuple{Int64,Base.Slice{Base.OneTo{Int64}}},true}

struct ColumnIterator{A<:AbstractMatrix,R<:IndexRange}
    a::A
    r::R
end
cols(a::AbstractMatrix, r::IndexRange = axes(a,2)) = ColumnIterator(a, r)

function Base.iterate(iter::ColumnIterator)
    next = iterate(iter.r)
    if next === nothing
        return nothing
    else
        i, s = next
        return view(iter.a, :, i), s
    end
end
function Base.iterate(iter::ColumnIterator, s)
    next = iterate(iter.r, s)
    if next === nothing
        return nothing
    else
        i, s = next
        return view(iter.a, :, i), s
    end
end

Base.IteratorSize(::Type{<:ColumnIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ColumnIterator}) = Base.HasEltype()
Base.length(iter::ColumnIterator) = length(iter.r)
Base.eltype(iter::ColumnIterator{A}) where {T,A<:DenseArray{T}} = SubArray{T,1,A,Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true}

# # QR decomposition
# function qr!(A::StridedMatrix{<:BlasFloat})
#     m, n = size(A)
#     A, T = LAPACK.geqrt!(A, min(minimum(size(A)), 36))
#     Q = LAPACK.gemqrt!('L', 'N', A, T, eye(eltype(A), m, min(m,n)))
#     R = triu!(A[1:min(m,n), :])
#     return Q, R
# end

# Triangular division: for some reason this is faster than LAPACKs trsv
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


# Eigenvalue decomposition of SymTridiagonal matrix
eig!(A::SymTridiagonal{T}, Z::StridedMatrix{T} = one(A)) where {T<:BlasFloat} = steqr!(A.dv, A.ev, Z)

# Schur factorization of a Hessenberg matrix
hschur!(H::StridedMatrix{T}, Z::StridedMatrix{T} = one(H)) where {T<:BlasFloat} = hseqr!(H, Z)

schur2eigvals(T::StridedMatrix{<:BlasFloat}) = schur2eigvals(T, 1:size(T,1))

function schur2eigvals(T::StridedMatrix{<:BlasComplex}, which::AbstractVector{Int})
    n = checksquare(T)
    which2 = unique(which)
    length(which2) == length(which) || throw(ArgumentError("which should contain unique values"))
    return [T[i,i] for i in which2]
end

function schur2eigvals(T::StridedMatrix{<:BlasReal}, which::AbstractVector{Int})
    n = checksquare(T)
    which2 = unique(which)
    length(which2) == length(which) || throw(ArgumentError("which should contain unique values"))
    D = zeros(Complex{eltype(T)}, length(which2))
    for k = 1:length(which)
        i = which[k]
        if i < n && !iszero(T[i+1,i])
            halftr = (T[i,i]+T[i+1,i+1])/2
            diff = (T[i,i]-T[i+1,i+1])/2
            d = diff*diff + T[i,i+1]*T[i+1,i]  # = hafltr*halftr - det
            D[i] = halftr + im*sqrt(-d)
        elseif i > 1 && !iszero(T[i,i-1])
            halftr = (T[i,i]+T[i-1,i-1])/2
            diff = -(T[i,i]-T[i-1,i-1])/2
            d = diff*diff + T[i,i-1]*T[i-1,i]  # = hafltr*halftr - det
            D[i] = halftr - im*sqrt(-d)
        else
            D[i] = T[i,i]
        end
    end
    return D
end

function schur2eigvecs(T::StridedMatrix{<:BlasComplex})
    n = checksquare(T)
    VR = similar(T, n, n)
    VL = similar(T, n, 0)
    select = Vector{BlasInt}(undef, 0)
    trevc!('R','A', select, T, VL, VR)
    return VR
end
function schur2eigvecs(T::StridedMatrix{<:BlasComplex}, which::AbstractVector{Int})
    n = checksquare(T)
    which2 = unique(which)
    length(which2) == length(which) || throw(ArgumentError("which should contain unique values"))
    m = BlasInt(length(which2))
    VR = similar(T, n, m)
    VL = similar(T, n, 0)

    select = zeros(BlasInt, n)
    for k = 1:length(which2)
        i = which2[k]
        select[i] = one(BlasInt)
        trevc!('R','S', select, T, VL, view(VR,:,k:k))
        select[i] = zero(BlasInt)
    end
    return VR
end
function schur2eigvecs(T::StridedMatrix{<:BlasReal})
    n = checksquare(T)
    VR = similar(T, Complex{eltype(T)}, n, n)
    VR′ = similar(T, n, n)
    VL′ = similar(T, n, 0)
    select = Vector{BlasInt}(undef, 0)
    trevc!('R','A', select, T, VL′, VR′)
    i = 1
    while i <= n
        if i == n || iszero(T[i+1,i])
            @inbounds @simd for k = 1:n
                VR[k,i]= VR′[k,i]
            end
            i += 1
        else
            @inbounds @simd for k = 1:n
                VR[k,i]= VR′[k,i] + im*VR′[k,i+1]
                VR[k,i+1]= VR′[k,i] - im*VR′[k,i+1]
            end
            i += 2
        end
    end
    return VR
end
function schur2eigvecs(T::StridedMatrix{<:BlasReal}, which::AbstractVector{Int})
    n = checksquare(T)
    which2 = unique(which)
    length(which2) == length(which) || throw(ArgumentError("which should contain unique values"))
    m = length(which2)
    VR = similar(T, Complex{eltype(T)}, n, m)
    VR′ = similar(T, n, 2)
    VL′ = similar(T, n, 0)

    select = zeros(BlasInt, n)
    i = 1
    while i <= n
        if i == n || iszero(T[i+1,i])
            j = findfirst(isequal(i), which2)
            if j !== nothing
                select[i] = one(BlasInt)
                trevc!('R','S', select, T, VL′, VR′)
                @inbounds @simd for k = 1:n
                    VR[k,j]= VR′[k,1]
                end
                select[i] = zero(BlasInt)
            end
            i += 1
        else
            j1 = findfirst(isequal(i), which2)
            j2 = findfirst(isequal(i+1), which2)
            if j1 !== nothing || j2 !== nothing
                select[i] = one(BlasInt)
                select[i+1] = one(BlasInt)
                trevc!('R','S', select, T, VL′, VR′)
                @inbounds @simd for k = 1:n
                    if j1 !== nothing
                        VR[k,j1]= VR′[k,1] + im*VR′[k,2]
                    end
                    if j2 !== nothing
                        VR[k,j2]= VR′[k,1] - im*VR′[k,2]
                    end
                end
                select[i] = zero(BlasInt)
                select[i+1] = zero(BlasInt)
            end
            i += 2
        end
    end
    return VR
end

function permuteeig!(D::StridedVector{S}, V::StridedMatrix{S}, perm::AbstractVector{Int}) where {S}
    n = checksquare(V)
    p = collect(perm) # makes copy cause will be overwritten
    isperm(p) && length(p) == n || throw(ArgumentError("not a valid permutation of length $n"))
    i = 1
    @inbounds while true
        if p[i] == i
            i = 1
            while i <= n && p[i] == i
                i += 1
            end
            i > n && break
        else
            iprev = findfirst(isequal(i), p)
            inext = p[i]
            p[iprev] = inext
            p[i] = i

            D[i], D[inext] = D[inext], D[i]
            for j = 1:n
                V[j,i], V[j,inext] = V[j,inext], V[j,i]
            end
            i = inext
        end
    end
    return D, V
end

permuteschur!(T::StridedMatrix{<:BlasFloat}, p::AbstractVector{Int}) = permuteschur!(T, one(T), p)
function permuteschur!(T::StridedMatrix{S}, Q::StridedMatrix{S}, perm::AbstractVector{Int}) where {S<:BlasComplex}
    n = checksquare(T)
    p = collect(perm) # makes copy cause will be overwritten
    isperm(p) && length(p) == n || throw(ArgumentError("not a valid permutation of length $n"))
    @inbounds for i = 1:n
        ifirst::BlasInt = p[i]
        ilast::BlasInt = i
        T, Q = trexc!(ifirst, ilast, T, Q)
        for k = (i+1):n
            if p[k] < p[i]
                p[k] += 1
            end
        end
    end
    return T, Q
end

function permuteschur!(T::StridedMatrix{S}, Q::StridedMatrix{S}, perm::AbstractVector{Int}) where {S<:BlasReal}
    n = checksquare(T)
    p = collect(perm) # makes copy cause will be overwritten
    isperm(p) && length(p) == n || throw(ArgumentError("not a valid permutation of length $n"))
    i = 1
    @inbounds while i <= n
        ifirst::BlasInt = p[i]
        ilast::BlasInt = i
        if ifirst == n || iszero(T[ifirst+1,ifirst])
            T, Q = trexc!(ifirst, ilast, T, Q)
            @inbounds for k = (i+1):n
                if p[k] < p[i]
                    p[k] += 1
                end
            end
            i += 1
        else
            p[i+1] == ifirst+1 || error("cannot split 2x2 blocks when permuting schur decomposition")
            T, Q = trexc!(ifirst, ilast, T, Q)
            @inbounds for k = (i+2):n
                if p[k] < p[i]
                    p[k] += 2
                end
            end
            i += 2
        end
    end
    return T,Q
end

# redefine LAPACK interface to tridiagonal eigenvalue problem
for (steqr, elty) in ((:dsteqr_, :Float64), (:ssteqr_, :Float32))
    @eval begin
        function steqr!(D::StridedVector{$elty}, E::StridedVector{$elty}, Z::StridedMatrix{$elty})
            n = length(D)
            chkstride1(Z)
            checksquare(Z) == n || throw(DimensionMismatch())
            length(E) >= n-1 || throw(DimensionMismatch())
            compz = 'V'
            ldz = stride(Z, 2)
            work = Vector{$elty}(undef, max(1,2*n-2))
            info = Ref{BlasInt}()
            ccall((@blasfunc($steqr), liblapack), Nothing,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}),
                    compz, n, D, E, Z, ldz, work, info)
            chklapackerror(info[])
            return D, Z
        end
    end
end

# redefine LAPACK interface to schur
trexc!(ifst::BlasInt, ilst::BlasInt, T::StridedMatrix{S}, Q::StridedMatrix{S}) where {S<:BlasFloat}=
    trexc!('V', ifst, ilst, T, Q)

for (hseqr, trevc, trexc, trsen, elty) in
    ((:dhseqr_, :dtrevc_, :dtrexc_, :dtrsen_, :Float64),
     (:shseqr_, :strevc_, :strexc_, :stgsen_, :Float32))
    @eval begin
        function hseqr!(H::StridedMatrix{$elty}, Z::StridedMatrix{$elty} = one(H))
            chkstride1(H, Z)
            n = checksquare(H)
            checksquare(Z) == n || throw(DimensionMismatch())
            job = 'S'
            compz = 'V'
            ilo = 1
            ihi = n
            ldh = stride(H, 2)
            ldz = stride(Z, 2)
            wr = similar(H, $elty, n)
            wi = similar(H, $elty, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($hseqr), liblapack), Nothing,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                        job, compz, n, ilo, ihi,
                        H, ldh, wr, wi,
                        Z, ldz, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            H, Z, complex.(wr, wi)
        end
        function trevc!(side::Char, howmny::Char, select::StridedVector{BlasInt}, T::StridedMatrix{$elty},
                        VL::StridedMatrix{$elty},
                        VR::StridedMatrix{$elty})
            # Extract
            if side ∉ ['L','R','B']
                throw(ArgumentError("side argument must be 'L' (left eigenvectors), 'R' (right eigenvectors), or 'B' (both), got $side"))
            end
            n = checksquare(T)
            mm = side == 'L' ? size(VL,2) : (side == 'R' ? size(VR,2) : min(size(VL,2), size(VR,2)))
            ldt, ldvl, ldvr = stride(T, 2), stride(VL, 2), stride(VR, 2)

            # Check
            chkstride1(T, select, VL, VR)

            # Allocate
            m = Ref{BlasInt}()
            work = Vector{$elty}(undef, 3n)
            info = Ref{BlasInt}()

            ccall((@blasfunc($trevc), liblapack), Nothing,
                (Ref{UInt8}, Ref{UInt8}, Ptr{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt},Ref{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{BlasInt}),
                side, howmny, select, n,
                T, ldt, VL, ldvl,
                VR, ldvr, mm, m,
                work, info)
            chklapackerror(info[])

            return VL, VR, m
        end
        function trexc!(compq::Char, ifst::BlasInt, ilst::BlasInt, T::StridedMatrix{$elty}, Q::StridedMatrix{$elty})
            chkstride1(T, Q)
            n = checksquare(T)
            ldt = stride(T, 2)
            ldq = stride(Q, 2)
            work = Vector{$elty}(undef, n)
            info = Ref{BlasInt}()
            ccall((@blasfunc($trexc), liblapack), Nothing,
                  (Ref{UInt8},  Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ref{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}),
                  compq, n,
                  T, ldt, Q, ldq,
                  ifst, ilst,
                  work, info)
            chklapackerror(info[])
            T, Q
        end
        function trsen!(job::Char, compq::Char, select::StridedVector{BlasInt},
                        T::StridedMatrix{$elty}, Q::StridedMatrix{$elty})
            chkstride1(T, Q, select)
            n = checksquare(T)
            ldt = max(1, stride(T, 2))
            ldq = max(1, stride(Q, 2))
            wr = similar(T, $elty, n)
            wi = similar(T, $elty, n)
            m = sum(select)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info = Ref{BlasInt}()
            select = convert(Array{BlasInt}, select)
            s = Ref{$elty}(zero($elty))
            sep = Ref{$elty}(zero($elty))
            for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($trsen), liblapack), Nothing,
                    (Ref{UInt8}, Ref{UInt8}, Ptr{BlasInt}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                    Ptr{BlasInt}),
                    job, compq, select, n,
                    T, ldt, Q, ldq,
                    wr, wi, m, s, sep,
                    work, lwork, iwork, liwork,
                    info)
                chklapackerror(info[])
                if i == 1 # only estimated optimal lwork, liwork
                    lwork  = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = BlasInt(real(iwork[1]))
                    resize!(iwork, liwork)
                end
            end
            T, Q, complex.(wr, wi), s[], sep[]
        end
    end
end

 for (hseqr, trevc, trexc, trsen, elty, relty) in
    ((:zhseqr_, :ztrevc_, :ztrexc_, :ztrsen_, :ComplexF64, :Float64),
     (:chseqr_, :ctrevc_, :ctrexc_, :ctrsen_, :ComplexF32, :Float32))
    @eval begin
        function hseqr!(H::StridedMatrix{$elty}, Z::StridedMatrix{$elty} = one(H))
            chkstride1(H)
            chkstride1(Z)
            n     = checksquare(H)
            checksquare(Z) == n || throw(DimensionMismatch())
            job = 'S'
            compz = 'V'
            ilo = 1
            ihi = n
            ldh = stride(H, 2)
            ldz = stride(Z, 2)
            w    = similar(H, $elty, n)
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info  = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($hseqr), liblapack), Nothing,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                        job, compz, n, ilo, ihi,
                        H, ldh, w,
                        Z, ldz, work, lwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            H, Z, w
        end
        function trevc!(side::Char, howmny::Char, select::StridedVector{BlasInt}, T::StridedMatrix{$elty},
                        VL::StridedMatrix{$elty} = similar(T),
                        VR::StridedMatrix{$elty} = similar(T))

            # Extract
            if side ∉ ['L','R','B']
                throw(ArgumentError("side argument must be 'L' (left eigenvectors), 'R' (right eigenvectors), or 'B' (both), got $side"))
            end
            n = checksquare(T)
            mm = side == 'L' ? size(VL,2) : (side == 'R' ? size(VR,2) : min(size(VL,2), size(VR,2)))
            ldt, ldvl, ldvr = stride(T, 2), stride(VL, 2), stride(VR, 2)

            # Check
            chkstride1(T, select, VL, VR)

            # Allocate
            m = Ref{BlasInt}()
            work = Vector{$elty}(undef, 2n)
            rwork = Vector{$relty}(undef, n)
            info = Ref{BlasInt}()
            ccall((@blasfunc($trevc), liblapack), Nothing,
                (Ref{UInt8}, Ref{UInt8}, Ptr{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ptr{$relty}, Ptr{BlasInt}),
                side, howmny, select, n,
                T, ldt, VL, ldvl,
                VR, ldvr, mm, m,
                work, rwork, info)
            chklapackerror(info[])

            return VL, VR, m
        end
        function trexc!(compq::Char, ifst::BlasInt, ilst::BlasInt, T::StridedMatrix{$elty}, Q::StridedMatrix{$elty})
            chkstride1(T, Q)
            n = checksquare(T)
            ldt = max(1, stride(T, 2))
            ldq = max(1, stride(Q, 2))
            info = Ref{BlasInt}()
            ccall((@blasfunc($trexc), liblapack), Nothing,
                  (Ref{UInt8},  Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ref{BlasInt}, Ref{BlasInt},
                   Ptr{BlasInt}),
                  compq, n,
                  T, ldt, Q, ldq,
                  ifst, ilst,
                  info)
            chklapackerror(info[])
            T, Q
        end
        function trsen!(job::Char, compq::Char, select::StridedVector{BlasInt},
                        T::StridedMatrix{$elty}, Q::StridedMatrix{$elty})
            chkstride1(select, T, Q)
            n = checksquare(T)
            ldt = max(1, stride(T, 2))
            ldq = max(1, stride(Q, 2))
            w = similar(T, $elty, n)
            m = sum(select)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            select = convert(Array{BlasInt}, select)
            s = Ref{$relty}(zero($relty))
            sep = Ref{$relty}(zero($relty))
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($trsen), liblapack), Nothing,
                    (Ref{UInt8}, Ref{UInt8}, Ptr{BlasInt}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ref{$relty}, Ref{$relty},
                    Ptr{$elty}, Ref{BlasInt},
                    Ptr{BlasInt}),
                    job, compq, select, n,
                    T, ldt, Q, ldq,
                    w, m, s, sep,
                    work, lwork,
                    info)
                chklapackerror(info[])
                if i == 1 # only estimated optimal lwork, liwork
                    lwork  = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            T, Q, w, s[], sep[]
        end
    end
end
