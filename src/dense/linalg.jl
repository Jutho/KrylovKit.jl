# Some modified wrappers for Lapack
import LinearAlgebra: BlasFloat, BlasInt,
                      LAPACKException, DimensionMismatch, SingularException,
                      PosDefException,
                      chkstride1, checksquare
import LinearAlgebra.BLAS: @blasfunc, BlasReal, BlasComplex
import LinearAlgebra.LAPACK: chklapackerror
@static if VERSION >= v"1.7"
    const liblapack = LinearAlgebra.BLAS.libblastrampoline
else
    const liblapack = LinearAlgebra.LAPACK.liblapack
end

@static if isdefined(Base, :require_one_based_indexing)
    import Base: require_one_based_indexing
else
    function require_one_based_indexing(A...)
        return !Base.has_offset_axes(A...) ||
               throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
    end
end

struct RowIterator{A<:AbstractMatrix,R<:IndexRange}
    a::A
    r::R
end
rows(a::AbstractMatrix, r::IndexRange=axes(a, 1)) = RowIterator(a, r)

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
function Base.eltype(iter::RowIterator{A}) where {T,A<:DenseArray{T}}
    return SubArray{T,1,A,Tuple{Int,Base.Slice{Base.OneTo{Int}}},true}
end

struct ColumnIterator{A<:AbstractMatrix,R<:IndexRange}
    a::A
    r::R
end
cols(a::AbstractMatrix, r::IndexRange=axes(a, 2)) = ColumnIterator(a, r)

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
function Base.eltype(iter::ColumnIterator{A}) where {T,A<:DenseArray{T}}
    return SubArray{T,1,A,Tuple{Base.Slice{Base.OneTo{Int}},Int},true}
end

# # QR decomposition
# function qr!(A::StridedMatrix{<:BlasFloat})
#     m, n = size(A)
#     A, T = LAPACK.geqrt!(A, min(minimum(size(A)), 36))
#     Q = LAPACK.gemqrt!('L', 'N', A, T, eye(eltype(A), m, min(m,n)))
#     R = triu!(A[1:min(m,n), :])
#     return Q, R
# end

# Triangular division: for some reason this is faster than LAPACK's trsv
function ldiv!(A::UpperTriangular, y::AbstractVector, r::UnitRange{Int}=1:length(y))
    R = A.data
    @inbounds for j in reverse(r)
        R[j, j] == zero(R[j, j]) && throw(SingularException(j))
        yj = (y[j] = R[j, j] \ y[j])
        @simd for i in first(r):(j - 1)
            y[i] -= R[i, j] * yj
        end
    end
    return y
end

# Eigenvalue decomposition of SymTridiagonal matrix
function tridiageigh!(A::SymTridiagonal{T}) where {T<:BlasFloat}
    Z = copyto!(similar(A.ev, size(A)), LinearAlgebra.I)
    return tridiageigh!(A, Z)
end
function tridiageigh!(A::SymTridiagonal{T}, Z::StridedMatrix{T}) where {T<:BlasFloat}
    return stegr!(A.dv, A.ev, Z)
end # redefined

# Generalized eigenvalue decomposition of symmetric / Hermitian problem
function geneigh!(A::StridedMatrix{T}, B::StridedMatrix{T}) where {T<:BlasFloat}
    return LAPACK.sygvd!(1, 'V', 'U', A, B)
end

# Singular value decomposition of a Bidiagonal matrix
function bidiagsvd!(B::Bidiagonal{T},
                    U::AbstractMatrix{T}=one(B),
                    VT::AbstractMatrix{T}=one(B)) where {T<:BlasReal}
    s, Vt, U = LAPACK.bdsqr!(B.uplo, B.dv, B.ev, VT, U, similar(U, (size(B, 1), 0)))
    return U, s, Vt
end

function reversecols!(U::AbstractMatrix)
    n = size(U, 2)
    @inbounds for j in 1:div(n, 2)
        @simd for i in 1:size(U, 1)
            U[i, j], U[i, n + 1 - j] = U[i, n + 1 - j], U[i, j]
        end
    end
    return U
end
function reverserows!(V::AbstractVecOrMat)
    m = size(V, 1)
    @inbounds for j in 1:size(V, 2)
        @simd for i in 1:div(m, 2)
            V[i, j], V[m + 1 - i, j] = V[m + 1 - i, j], V[i, j]
        end
    end
    return V
end

# Schur factorization of a Hessenberg matrix
function hschur!(H::AbstractMatrix{T}, Z::AbstractMatrix{T}=one(H)) where {T<:BlasFloat}
    return hseqr!(H, Z)
end

schur2eigvals(T::AbstractMatrix{<:BlasFloat}) = schur2eigvals(T, 1:size(T, 1))

function schur2eigvals(T::AbstractMatrix{<:BlasComplex}, which::AbstractVector{Int})
    n = checksquare(T)
    which2 = unique(which)
    length(which2) == length(which) ||
        throw(ArgumentError("which should contain unique values"))
    return [T[i, i] for i in which2]
end

function schur2eigvals(T::AbstractMatrix{<:BlasReal}, which::AbstractVector{Int})
    n = checksquare(T)
    which2 = unique(which)
    length(which2) == length(which) ||
        throw(ArgumentError("which should contain unique values"))
    D = zeros(Complex{eltype(T)}, length(which2))
    for k in 1:length(which)
        i = which[k]
        if i < n && !iszero(T[i + 1, i])
            halftr = (T[i, i] + T[i + 1, i + 1]) / 2
            diff = (T[i, i] - T[i + 1, i + 1]) / 2
            d = diff * diff + T[i, i + 1] * T[i + 1, i]  # = hafltr*halftr - det
            D[i] = halftr + im * sqrt(-d)
        elseif i > 1 && !iszero(T[i, i - 1])
            halftr = (T[i, i] + T[i - 1, i - 1]) / 2
            diff = -(T[i, i] - T[i - 1, i - 1]) / 2
            d = diff * diff + T[i, i - 1] * T[i - 1, i]  # = hafltr*halftr - det
            D[i] = halftr - im * sqrt(-d)
        else
            D[i] = T[i, i]
        end
    end
    return D
end

function _normalizevecs!(V)
    @inbounds for k in 1:size(V, 2)
        normalize!(view(V, :, k))
    end
    return V
end
function schur2eigvecs(T::AbstractMatrix{<:BlasComplex})
    n = checksquare(T)
    VR = similar(T, n, n)
    VL = similar(T, n, 0)
    select = Vector{BlasInt}(undef, 0)
    trevc!('R', 'A', select, T, VL, VR)
    return _normalizevecs!(VR)
end
function schur2eigvecs(T::AbstractMatrix{<:BlasComplex}, which::AbstractVector{Int})
    n = checksquare(T)
    which2 = unique(which)
    length(which2) == length(which) ||
        throw(ArgumentError("which should contain unique values"))
    m = BlasInt(length(which2))
    VR = similar(T, n, m)
    VL = similar(T, n, 0)

    select = zeros(BlasInt, n)
    for k in 1:length(which2)
        i = which2[k]
        select[i] = one(BlasInt)
        trevc!('R', 'S', select, T, VL, view(VR, :, k:k))
        select[i] = zero(BlasInt)
    end
    return _normalizevecs!(VR)
end
function schur2eigvecs(T::StridedMatrix{<:BlasReal})
    n = checksquare(T)
    VR = similar(T, Complex{eltype(T)}, n, n)
    VR′ = similar(T, n, n)
    VL′ = similar(T, n, 0)
    select = Vector{BlasInt}(undef, 0)
    trevc!('R', 'A', select, T, VL′, VR′)
    i = 1
    while i <= n
        if i == n || iszero(T[i + 1, i])
            @inbounds @simd for k in 1:n
                VR[k, i] = VR′[k, i]
            end
            i += 1
        else
            @inbounds @simd for k in 1:n
                VR[k, i] = VR′[k, i] + im * VR′[k, i + 1]
                VR[k, i + 1] = VR′[k, i] - im * VR′[k, i + 1]
            end
            i += 2
        end
    end
    return _normalizevecs!(VR)
end
function schur2realeigvecs(T::StridedMatrix{<:BlasReal})
    n = checksquare(T)
    for i in 1:(n - 1)
        iszero(T[i + 1, i]) || throw(ArgumentError("T must be upper triangular"))
    end
    VR = similar(T, n, n)
    VL = similar(T, n, 0)
    select = Vector{BlasInt}(undef, 0)
    trevc!('R', 'A', select, T, VL, VR)
    return _normalizevecs!(VR)
end
function schur2eigvecs(T::AbstractMatrix{<:BlasReal}, which::AbstractVector{Int})
    n = checksquare(T)
    which2 = unique(which)
    length(which2) == length(which) ||
        throw(ArgumentError("which should contain unique values"))
    m = length(which2)
    VR = similar(T, Complex{eltype(T)}, n, m)
    VR′ = similar(T, n, 2)
    VL′ = similar(T, n, 0)

    select = zeros(BlasInt, n)
    i = 1
    while i <= n
        if i == n || iszero(T[i + 1, i])
            j = findfirst(isequal(i), which2)
            if j !== nothing
                select[i] = one(BlasInt)
                trevc!('R', 'S', select, T, VL′, VR′)
                @inbounds @simd for k in 1:n
                    VR[k, j] = VR′[k, 1]
                end
                select[i] = zero(BlasInt)
            end
            i += 1
        else
            j1 = findfirst(isequal(i), which2)
            j2 = findfirst(isequal(i + 1), which2)
            if j1 !== nothing || j2 !== nothing
                select[i] = one(BlasInt)
                select[i + 1] = one(BlasInt)
                trevc!('R', 'S', select, T, VL′, VR′)
                @inbounds @simd for k in 1:n
                    if j1 !== nothing
                        VR[k, j1] = VR′[k, 1] + im * VR′[k, 2]
                    end
                    if j2 !== nothing
                        VR[k, j2] = VR′[k, 1] - im * VR′[k, 2]
                    end
                end
                select[i] = zero(BlasInt)
                select[i + 1] = zero(BlasInt)
            end
            i += 2
        end
    end
    return _normalizevecs!(VR)
end

function permuteeig!(D::AbstractVector, V::AbstractMatrix, perm::AbstractVector{Int})
    n = checksquare(V)
    p = collect(perm) # makes copy cause will be overwritten
    isperm(p) && length(p) == n ||
        throw(ArgumentError("not a valid permutation of length $n"))
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
            @simd for j in 1:n
                V[j, i], V[j, inext] = V[j, inext], V[j, i]
            end
            i = inext
        end
    end
    return D, V
end

function permuteschur!(T::AbstractMatrix{<:BlasFloat}, p::AbstractVector{Int})
    return permuteschur!(T, one(T), p)
end
function permuteschur!(T::AbstractMatrix{S}, Q::AbstractMatrix{S},
                       order::AbstractVector{Int}) where {S<:BlasComplex}
    n = checksquare(T)
    p = collect(order) # makes copy cause will be overwritten
    @inbounds for i in 1:length(p)
        ifirst::BlasInt = p[i]
        ilast::BlasInt = i
        T, Q = LAPACK.trexc!(ifirst, ilast, T, Q)
        for k in (i + 1):length(p)
            if p[k] < p[i]
                p[k] += 1
            end
        end
    end
    return T, Q, schur2eigvals(T)
end

function permuteschur!(T::AbstractMatrix{S}, Q::AbstractMatrix{S},
                       order::AbstractVector{Int}) where {S<:BlasReal}
    n = checksquare(T)
    p = collect(order) # makes copy cause will be overwritten
    i = 1
    @inbounds while i <= length(p)
        ifirst::BlasInt = p[i]
        ilast::BlasInt = i
        if ifirst == n || iszero(T[ifirst + 1, ifirst])
            T, Q = LAPACK.trexc!(ifirst, ilast, T, Q)
            @inbounds for k in (i + 1):length(p)
                if p[k] < p[i]
                    p[k] += 1
                end
            end
            i += 1
        else
            p[i + 1] == ifirst + 1 ||
                error("cannot split 2x2 blocks when permuting schur decomposition")
            T, Q = LAPACK.trexc!(ifirst, ilast, T, Q)
            @inbounds for k in (i + 2):length(p)
                if p[k] < p[i]
                    p[k] += 2
                end
            end
            i += 2
        end
    end
    return T, Q, schur2eigvals(T)
end

function partitionschur!(T::AbstractMatrix{S}, Q::AbstractMatrix{S},
                         select::AbstractVector{Bool}) where {S<:BlasFloat}
    T, Q, vals = trsen!('N', 'V', convert(Vector{BlasInt}, select), T, Q)
    return T, Q, vals
end

# redefine LAPACK interface to tridiagonal eigenvalue problem
for (stegr, elty) in ((:dstegr_, :Float64), (:sstegr_, :Float32))
    @eval begin
        function stegr!(dv::AbstractVector{$elty}, ev::AbstractVector{$elty},
                        Z::AbstractMatrix{$elty})
            require_one_based_indexing(dv, ev, Z)
            chkstride1(dv, ev, Z)
            n = length(dv)
            if length(ev) == n - 1
                eev = [ev; zero($elty)]
            elseif length(ev) == n
                eev = ev
            else
                throw(DimensionMismatch("ev has length $(length(ev)) but needs one less than dv's length, $n)"))
            end
            checksquare(Z) == n || throw(DimensionMismatch())
            ldz = max(1, stride(Z, 2))
            jobz = 'V'
            range = 'A'
            abstol = Vector{$elty}(undef, 1)
            il = 1
            iu = n
            vl = zero($elty)
            vu = zero($elty)
            m = Ref{BlasInt}()
            w = similar(dv, $elty, n)
            isuppz = similar(dv, BlasInt, 2 * size(Z, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($stegr), liblapack),
                      Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{$elty}, Ref{$elty},
                       Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                       Ptr{BlasInt}, Clong, Clong),
                      jobz, range, n,
                      dv, eev, vl, vu,
                      il, iu, abstol, m,
                      w, Z, ldz,
                      isuppz, work, lwork, iwork, liwork,
                      info, 1, 1)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(work[1])
                    resize!(work, lwork)
                    liwork = iwork[1]
                    resize!(iwork, liwork)
                end
            end
            return w, Z
        end
    end
end

# redefine LAPACK interface to schur
for (hseqr, trevc, trsen, elty) in
    ((:dhseqr_, :dtrevc_, :dtrsen_, :Float64), (:shseqr_, :strevc_, :strsen_, :Float32))
    @eval begin
        function hseqr!(H::StridedMatrix{$elty}, Z::StridedMatrix{$elty}=one(H))
            require_one_based_indexing(H, Z)
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
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($hseqr), liblapack),
                      Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                       Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                      job, compz, n, ilo, ihi,
                      H, ldh, wr, wi, Z, ldz,
                      work, lwork, info, 1, 1)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return H, Z, complex.(wr, wi)
        end
        function trevc!(side::Char, howmny::Char, select::StridedVector{BlasInt},
                        T::AbstractMatrix{$elty}, VL::AbstractMatrix{$elty},
                        VR::AbstractMatrix{$elty})
            # Extract
            if side ∉ ['L', 'R', 'B']
                throw(ArgumentError("side argument must be 'L' (left eigenvectors), 'R' (right eigenvectors), or 'B' (both), got $side"))
            end
            n = checksquare(T)
            mm = side == 'L' ? size(VL, 2) :
                 (side == 'R' ? size(VR, 2) : min(size(VL, 2), size(VR, 2)))
            ldt, ldvl, ldvr = stride(T, 2), stride(VL, 2), stride(VR, 2)

            # Check
            chkstride1(T, select, VL, VR)

            # Allocate
            m = Ref{BlasInt}()
            work = Vector{$elty}(undef, 3n)
            info = Ref{BlasInt}()

            ccall((@blasfunc($trevc), liblapack),
                  Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ptr{BlasInt}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt},
                   Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Clong, Clong),
                  side, howmny, select, n,
                  T, ldt, VL, ldvl, VR, ldvr,
                  mm, m, work, info, 1, 1)
            chklapackerror(info[])

            return VL, VR, m
        end
        function trsen!(job::AbstractChar, compq::AbstractChar,
                        select::AbstractVector{BlasInt},
                        T::AbstractMatrix{$elty}, Q::AbstractMatrix{$elty})
            chkstride1(T, Q, select)
            n = checksquare(T)
            checksquare(Q) == n || throw(DimensionMismatch())
            length(select) == n || throw(DimensionMismatch())
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
            for i in 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($trsen), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ptr{BlasInt}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                       Ptr{BlasInt}, Clong, Clong),
                      job, compq, select, n,
                      T, ldt, Q, ldq,
                      wr, wi, m, s, sep,
                      work, lwork, iwork, liwork,
                      info, 1, 1)
                chklapackerror(info[])
                if i == 1 # only estimated optimal lwork, liwork
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = BlasInt(real(iwork[1]))
                    resize!(iwork, liwork)
                end
            end
            return T, Q, complex.(wr, wi), s[], sep[]
        end
    end
end

for (hseqr, trevc, trsen, elty, relty) in
    ((:zhseqr_, :ztrevc_, :ztrsen_, :ComplexF64, :Float64),
     (:chseqr_, :ctrevc_, :ctrsen_, :ComplexF32, :Float32))
    @eval begin
        function hseqr!(H::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty}=one(H))
            require_one_based_indexing(H, Z)
            chkstride1(H, Z)
            n = checksquare(H)
            checksquare(Z) == n || throw(DimensionMismatch())
            job = 'S'
            compz = 'V'
            ilo = 1
            ihi = n
            ldh = stride(H, 2)
            ldz = stride(Z, 2)
            w = similar(H, $elty, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($hseqr), liblapack),
                      Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                      job, compz, n, ilo, ihi,
                      H, ldh, w, Z, ldz,
                      work, lwork, info, 1, 1)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return H, Z, w
        end
        function trevc!(side::Char, howmny::Char, select::AbstractVector{BlasInt},
                        T::AbstractMatrix{$elty}, VL::AbstractMatrix{$elty}=similar(T),
                        VR::AbstractMatrix{$elty}=similar(T))

            # Check
            require_one_based_indexing(select, T, VL, VR)
            chkstride1(T, select, VL, VR)

            # Extract
            if side ∉ ['L', 'R', 'B']
                throw(ArgumentError("side argument must be 'L' (left eigenvectors), 'R' (right eigenvectors), or 'B' (both), got $side"))
            end
            n = checksquare(T)
            mm = side == 'L' ? size(VL, 2) :
                 (side == 'R' ? size(VR, 2) : min(size(VL, 2), size(VR, 2)))
            ldt, ldvl, ldvr = stride(T, 2), stride(VL, 2), stride(VR, 2)

            # Allocate
            m = Ref{BlasInt}()
            work = Vector{$elty}(undef, 2n)
            rwork = Vector{$relty}(undef, n)
            info = Ref{BlasInt}()
            ccall((@blasfunc($trevc), liblapack),
                  Cvoid,
                  (Ref{UInt8},
                   Ref{UInt8},
                   Ptr{BlasInt},
                   Ref{BlasInt},
                   Ptr{$elty},
                   Ref{BlasInt},
                   Ptr{$elty},
                   Ref{BlasInt},
                   Ptr{$elty},
                   Ref{BlasInt},
                   Ref{BlasInt},
                   Ptr{BlasInt},
                   Ptr{$elty},
                   Ptr{$relty},
                   Ptr{BlasInt},
                   Clong,
                   Clong),
                  side,
                  howmny,
                  select,
                  n,
                  T,
                  ldt,
                  VL,
                  ldvl,
                  VR,
                  ldvr,
                  mm,
                  m,
                  work,
                  rwork,
                  info,
                  1,
                  1)
            chklapackerror(info[])

            return VL, VR, m
        end
        function trsen!(job::Char,
                        compq::Char,
                        select::AbstractVector{BlasInt},
                        T::AbstractMatrix{$elty},
                        Q::AbstractMatrix{$elty})
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
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($trsen), liblapack),
                      Nothing,
                      (Ref{UInt8},
                       Ref{UInt8},
                       Ptr{BlasInt},
                       Ref{BlasInt},
                       Ptr{$elty},
                       Ref{BlasInt},
                       Ptr{$elty},
                       Ref{BlasInt},
                       Ptr{$elty},
                       Ref{BlasInt},
                       Ref{$relty},
                       Ref{$relty},
                       Ptr{$elty},
                       Ref{BlasInt},
                       Ptr{BlasInt}),
                      job,
                      compq,
                      select,
                      n,
                      T,
                      ldt,
                      Q,
                      ldq,
                      w,
                      m,
                      s,
                      sep,
                      work,
                      lwork,
                      info)
                chklapackerror(info[])
                if i == 1 # only estimated optimal lwork, liwork
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return T, Q, w, s[], sep[]
        end
    end
end
