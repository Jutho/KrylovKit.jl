import Base.LinAlg: SingularException, checksquare

# QR ALGORITHM:
# Compute Schur form of a Hessenberg matrix

# Givens transform the upper right corner of a matrix.
function transform!(H::AbstractMatrix, U, G::Givens, imax::Int = size(H,1), jmin::Int = 1)
    lmul!(H, G, jmin:size(H,2))
    rmulc!(H, G, 1:imax)
    rmulc!(U, G)
end

# Single QR step
function qrstep!(H::AbstractMatrix{T}, U, σ::T = zero(T), start::Int = 1, stop::Int = size(H,2)) where {T}
    # H[start:stop,start:stop] is assumed to be of Hessenberg format
    @inbounds begin
        i = start
        # Initial Givens rotations determined from shift
        G, x = givens(H[i,i] - σ, H[i+1,i], i, i+1)
        transform!(H, U, G, min(i+2,stop), i)
        for i = start:stop-2 # bulge chase
            G, x = givens(H[i+1,i], H[i+2,i], i+1, i+2) # bulge = H[i+2,i]
            H[i+1,i] = x
            H[i+2,i] = zero(T)
            transform!(H, U, G, min(i+3,stop), i+1)
        end
    end
end

# Double QR step
function qrdoublestep!(H::AbstractMatrix{T}, U, s::T, t::T, start::Int = 1, stop::Int = size(H,2)) where {T}
    # H[start:stop,start:stop] is assumed to be of Hessenberg format
    @inbounds begin
        i = start
        # Initial Givens rotation determined from shifts
        x = H[i,i]*H[i,i] + H[i,i+1]*H[i+1,i] - s*H[i,i] + t
        y = H[i+1,i]*(H[i,i] + H[i+1,i+1] - s)
        z = H[i+2,i+1]*H[i+1,i]
        G1, x = givens(x,z,i,i+2)
        G2, x = givens(x,y,i,i+1)
        transform!(H, U, G1, min(i+3,stop), i)
        transform!(H, U, G2, min(i+3,stop), i)

        for i = start:stop-3 # double bulge chase
            x = H[i+1,i]
            y = H[i+2,i] # bulge 1
            z = H[i+3,i] # bulge 2
            G1, x = givens(x,z,i+1,i+3)
            G2, x = givens(x,y,i+1,i+2)
            H[i+1, i] = x
            H[i+2, i] = zero(T)
            H[i+3, i] = zero(T)
            transform!(H, U, G1, min(i+4,stop), i+1)
            transform!(H, U, G2, min(i+4,stop), i+1)
        end
        i = stop-2 # final single bulge
        G, x = givens(H[i+1,i], H[i+2,i], i+1, i+2) # bulge = H[i+2,i]
        H[i+1,i] = x
        H[i+2,i] = zero(T)
        transform!(H, U, G, stop, i+1)
    end
end

"""
    hschur!(H::AbstractMatrix, U = novecs)

Transforms the Hessenberg matrix `H` into Schur form (upper triangular in the
complex case and quasi upper triangular in the real case) using a series of
Givens rotations which are multiplied onto `U`. Initialize `U` as the identity
matrix in order to obtain the Schur vectors upon return. If `U` is initialized as
`novecs` (the default), the corresponding basis change is discarded.
"""
function hschur!(H::AbstractMatrix, U = novecs)
    n = checksquare(H)
    T = eltype(H)
    ϵ = eps(real(T))
    TC = complex(T)
    values = Vector{TC}(n)

    p = n
    @inbounds while p > 0
        i = p
        while i > 1 && abs(H[i,i-1]) > ϵ*(abs(H[i-1,i-1])+abs(H[i,i]))
            i -= 1
        end
        i > 1 && (H[i,i-1] = zero(T))
        if i == p
            values[p] = H[p,p]
            p = p-1
        elseif i == p-1
            values[i], values[i+1] = qrnormalform!(H, i, U)
            p = p-2
        else
            q = p-1
            # Wilkinson shifts
            s = H[q,q]+H[p,p]
            t = H[q,q]*H[p,p] - H[q,p]*H[p,q]
            IterativeToolbox.qrdoublestep!(H, U, s, t, i, p)
        end
    end
    return H, U, values
end

function schur2rightvecs(H::AbstractMatrix, select::AbstractVector{Bool})
    checksquare(H) == length(select) || throw(ArgumentError("select has wrong length"))
    schur2eigvecs(H, find(select))
end
schur2rightvecs(H::AbstractMatrix, select::Int) = vec(schur2eigvecs(H, [select]))

function schur2rightvecs(H::AbstractMatrix{<:Real}, select::AbstractVector{Int} = 1:size(H,1))
    # real schur, H is quasi upper triangular
    # based on naivesub! from Julia (base/linalg/triangular.jl)
    T = eltype(H)
    n = checksquare(H)
    selected = sort!(collect(select))
    (minimum(selected) < 1 || maximum(selected) > n) && throw(ArgumentError("invalid input select"))

    V = zeros(T,n,length(selected))

    ind = 1
    @inbounds while ind <= length(selected)
        k = selected[ind]

        if k==n || H[k+1,k] == zero(T)
            λ = subeig11!(H, V, k, ind)
            j = k-1
            while j > 0
                if j==1 || H[j,j-1] == zero(T)
                    subinv11!(H, V, λ, j, k, ind)
                    j-=1
                else
                    subinv21!(H, V, λ, j, k, ind)
                    j-=2
                end
            end
            normalize!(view(V,1:k,ind))
            k += 1
            ind += 1
        else
            if ind == length(selected) || k+1 != selected[ind+1]
                throw(ArgumentError("you need to select both vectors of a complex pair"))
            end

            λR, λI = subeig22!(H, V, k, ind)
            j = k-1
            while j > 0
                if j==1 || H[j,j-1] == zero(T)
                    subinv12!(H, V, λR, λI, j, k, ind)
                    j-=1
                else
                    subinv22!(H, V, λR, λI, j, k, ind)
                    j-=2
                end
            end
            v = view(V,1:k+1,ind:ind+1)
            scale!(v,1/vecnorm(v))
            k += 2
            ind += 2
        end
    end
    return V
end

function schur2rightvecs(H::AbstractMatrix{<:Complex}, select::AbstractVector{Int}=1:size(H,1))
    # complex schur, H is upper triangular
    T = eltype(H)
    n = checksquare(H)
    V = zeros(T,n,length(select))
    @inbounds for (ind, k) in enumerate(select)
        λ = subeig11!(H, V, k, ind)
        for j = k-1:-1:1
            subinv11!(H, V, λ, j, k, ind)
        end
        normalize!(view(V,1:k,ind))
    end
    return V
end

function reorderschur!(H::AbstractMatrix, select::AbstractVector{Bool}, U = novecs)
    n = size(H,2)
    n == length(select) || throw(ArgumentError("select has wrong length"))
    p = Vector{Int}(n)
    i = 1
    j = sum(select)+1
    @inbounds for k = 1:n
        if select[k]
            p[i] = k
            i += 1
        else
            p[j] = k
            j += 1
        end
    end
    reorderschur!(H, p, U)
end

function reorderschur!(H::AbstractMatrix, p::AbstractVector{Int}, U = novecs)
    T = eltype(H)
    n = checksquare(H)
    (n == length(p) && isperm(p)) || throw(ArgumentError("invalid reordering"))
    order = collect(p)
    H2 = copy(H)
    U2 = one(H)

    last = n
    local i::Int, j::Int
    while !isempty(order)
        i = pop!(order)
        if T<:Real && i < n && H[i+1,i] != zero(T)
            throw(ArgumentError("rows of a 2x2 block should be kept together: $i"))
        elseif T<:Real && i > 1 && H[i,i-1] != zero(T)
            j = i
            i = pop!(order)
            i == j-1 || throw(ArgumentError("rows of a 2x2 block should be kept together: $i & $j"))
            @inbounds for k = 1:(last-2)
                order[k] > i && (order[k] -= 2)
            end
            while i != last-1
                j = i+2
                if T<:Complex || j == n || H[j+1,j] == zero(T)
                    swapschur21!(H, i, U)
                    i += 1
                else
                    swapschur22!(H, i, U)
                    i += 2
                end
            end
            last -= 2
        else
            @inbounds for k = 1:(last-1)
                order[k] > i && (order[k] -= 1)
            end
            while i != last
                j = i+1
                if T<:Complex || j == n || H[j+1,j] == zero(T)
                    swapschur11!(H, i, U)
                    i += 1
                else
                    swapschur12!(H, i, U)
                    i += 2
                end
            end
            last -= 1
        end
    end
    return H, U
end

# AUXILIARY ROUTINES: Unsafe for direct use, assumed to be inbounds

# Bring 2x2 block on rows i and i+1 into normal form
function qrnormalform!(H::AbstractMatrix{<:Real}, i, U)
    T = eltype(H)
    j = i+1
    @inbounds begin
        H[j,i] == zero(H[j,i]) && return complex(H[i,i]), complex(H[j,j])

        halftr = (H[i,i]+H[j,j])/2
        # det = H[i,i]*H[j,j] - H[i,j]*H[j,i]
        diff = (H[i,i]-H[j,j])/2
        d = diff*diff + H[i,j]*H[j,i]  # = hafltr*halftr - det

        if d >= zero(T) # diagonalize in real arithmetic
            sqrtd = sqrt(d)
            if H[j,j] < H[i,i]
                a = halftr+sqrt(d)-H[j,j]
                b = H[j,i]
                G, = givens(a, b, i, j)
            else
                a = H[i,j]
                b = halftr+sqrt(d)-H[i,i]
                G, = givens(a, b, i, j)
            end
            transform!(H, U, G, j, i)
            H[j,i] = zero(T)
            return complex(H[i,i]), complex(H[j,j])
        else # bring into standard form
            Θ = atan2(H[j,j]-H[i,i], H[i,j]+H[j,i])/2
            c = cos(Θ)
            s = sin(Θ)
            G = Givens(i, j, c, s)

            transform!(H, U, G, j, i)
            sqrtmd = sqrt(-d)
            return complex(halftr,+sqrtmd), complex(halftr,-sqrtmd)
        end
    end
end

function qrnormalform!(H::AbstractMatrix{<:Complex}, i, U)
    T = eltype(H)
    j = i+1
    @inbounds begin
        H[j,i] == zero(H[j,i]) && return H[i,i], H[j,j]

        halftr = (H[i,i]+H[j,j])/2
        det = H[i,i]*H[j,j] - H[i,j]*H[j,i]
        d = halftr*halftr-det
        sqrtd = sqrt(d)
        a1 = halftr+sqrt(d)-H[j,j]
        b1 = H[j,i]
        a2 = H[i,j]
        b2 = halftr+sqrt(d)-H[i,i]
        if abs2(a1)+abs2(b1) > abs2(a2)+abs2(b2)
            G, = givens(a1, b1, i, j)
        else
            G, = givens(a2, b2, i, j)
        end
        transform!(H, U, G, j, i)
        H[j,i] = zero(T)

        return H[i,i], H[j,j]
    end
end

# swap block on line i with block below
function swapschur11!(H, i, U)
    @inbounds begin
        j = i+1
        H[i,i] == H[j,j] && return
        c = -H[i,j]/(H[i,i]-H[j,j])
        G, = givens(c, one(c), i, j)
        transform!(H, U, G, j, i)
        H[j,i] = zero(H[j,i])
    end
end
function swapschur21!(H, i, U)
    # swap block on line i with block below
    @inbounds begin
        i1, i2, j = i, i+1, i+2

        A = (H[i1,i1]-H[j,j], H[i2,i1], H[i1,i2], H[i2,i2]-H[j,j])
        c1, c2 = linear22(A, (-H[i1,j], -H[i2,j]))

        G1, c1 = givens(c1, c2, i1, i2)
        G2, = givens(c1, one(c1), i1, j)

        transform!(H, U, G1, j, i1)
        transform!(H, U, G2, j, i1)
        H[i+1,i] = H[i+2,i] = zero(H[i,i])

        qrnormalform!(H, i+1, U)
    end
end
function swapschur12!(H, i, U)
    # swap block on line i with block below
    @inbounds begin
        j1, j2 = i+1, i+2

        A = (H[j1,j1]-H[i,i], H[j1,j2], H[j2,j1], H[j2,j2]-H[i,i])
        c1, c2 = linear22(A, (H[i,j1], H[i,j2]))

        G1, = givens(c1, one(c1), i, j1)
        c2, c3 = lmul2((c2, zero(c2)), G1)
        G2, = givens(c3, one(c1), j1, j2)

        transform!(H, U, G1, j2, i)
        transform!(H, U, G2, j2, i)
        H[i+2,i] = H[i+2,i+1] = zero(H[i,i])

        qrnormalform!(H, i, U)
    end
end
function swapschur22!(H, i, U)
    @inbounds begin
        i1, i2, j1, j2 = i, i+1, i+2, i+3

        A = (H[i1,i1], H[i2,i1], H[i1, i2], H[i2, i2])
        B = (-H[j1,j1], -H[j2,j1], -H[j1, j2], -H[j2, j2])
        C = (H[i1,j1], H[i2,j1], H[i1,j2], H[i2,j2])

        x11, x21, x12, x22 = sylvester22(A, B, C)
        G1, x11 = givens(x11, x21, i1, i2)
        G2, = givens(x11, one(x11), i1, j1)

        x12, x22 = lmul2((x12, x22), G1)
        x12, x32 = lmul2((x12, zero(x12)), G2)

        G3, x32 = givens(x32, one(x32), j1, j2)
        G4, = givens(x22, x32, i2, j1)

        transform!(H, U, G1, j2, i1)
        transform!(H, U, G2, j2, i1)
        transform!(H, U, G3, j2, i1)
        transform!(H, U, G4, j2, i1)
        H[j1,i1] = H[j2,i1] = H[j1,i2] = H[j2,i2] = zero(H[i,i])

        qrnormalform!(H, i, U)
        qrnormalform!(H, i+2, U)
    end
end

# Solve local problems for finding eigenvectors
function subeig11!(H, V, k, ind)
    @inbounds begin
        V[k,ind] = one(V[k,ind])
        for i = 1:k-1
            V[i,ind] = H[i,k]
        end
        λ = H[k,k]
    end
    return λ
end
function subeig22!(H, V, k, ind)
    @inbounds begin
        k1, k2 = k, k+1
        ind1, ind2 = ind, ind+1
        a = H[k1,k2]
        b = H[k2,k1]
        den = abs(a)+abs(b)
        c1 = V[k1,ind1] = sqrt(abs(a)/den)
        V[k2,ind1] = zero(a)
        V[k1,ind2] = zero(a)
        c2 = V[k2,ind2] = copysign(sqrt(abs(b)/den), a)
        for i in 1:k-1
            V[i,ind1] = H[i,k1]*c1
            V[i,ind2] = H[i,k2]*c2
        end
        λR = H[k1,k1]
        λI = sqrt(abs(a*b))
    end
    return λR, λI
end

function subinv11!(H, V, λ, j, k, ind)
    @inbounds begin
        d = H[j,j] - λ
        d == zero(d) && throw(LinAlg.SingularException(j))
        c = V[j,ind] = d \ (-V[j,ind])
        for i in 1:j-1
            V[i,ind] += H[i,j]*c
        end
    end
end
function subinv21!(H, V, λ, j, k, ind)
    @inbounds begin
        j1, j2 = j-1, j

        A = (H[j1,j1]-λ, H[j2,j1], H[j1,j2], H[j2,j2]-λ)
        c1, c2 = V[j1,ind], V[j2,ind] = linear22(A, (-V[j1,ind], -V[j2,ind]))

        for i in 1:j-2
            V[i,ind] += H[i,j1]*c1 + H[i,j2]*c2
        end
    end
end
function subinv12!(H, V, λR, λI, j, k, ind)
    @inbounds begin
        ind1, ind2 = ind, ind+1

        A = (H[j,j]-λR, -λI, λI, H[j,j]-λR)
        b = (-V[j,ind1], -V[j,ind2])
        c1, c2 = V[j,ind1], V[j,ind2] = linear22(A, (-V[j,ind1], -V[j,ind2]))

        for i in 1:j-1
            V[i,ind1] += H[i,j]*c1
            V[i,ind2] += H[i,j]*c2
        end
    end
end
function subinv22!(H, V, λR, λI, j, k, ind)
    @inbounds begin
        j1, j2 = j-1, j
        ind1, ind2 = ind, ind+1

        A = (H[j1,j1], H[j2,j1], H[j1,j2], H[j2,j2])
        B = (-λR, λI, -λI, -λR)
        C = (V[j1,ind1], V[j2,ind1], V[j1,ind2], V[j2,ind2])

        V[j1,ind1], V[j2,ind1], V[j1,ind2], V[j2,ind2] = sylvester22(A, B, C)

        for i in 1:j-2
            V[i,ind1] += H[i,j1]*V[j1,ind1] + H[i,j2]*V[j2,ind1]
            V[i,ind2] += H[i,j1]*V[j1,ind2] + H[i,j2]*V[j2,ind2]
        end
    end
end

# 2x2 matrix and vector algebra
lmul2(x::NTuple{2}, G::Givens) = (G.c*x[1] + G.s*x[2], -conj(G.s)*x[1] + G.c*x[2])

function gemv22(A::NTuple{4}, b::NTuple{2}, α, c::NTuple{2})
    a11, a21, a12, a22 = A
    b1, b2 = b
    c1, c2 = c
    c1 = a11 * b1 + a12 * b2 + α * c1
    c2 = a21 * b1 + a22 * b2 + α * c2
    return (c1, c2)
end
function matmat22(A::NTuple{4}, B::NTuple{4})
    a11, a21, a12, a22 = A
    b11, b21, b12, b22 = B
    c11 = a11 * b11 + a12 * b21
    c21 = a21 * b11 + a22 * b21
    c12 = a11 * b12 + a12 * b22
    c22 = a21 * b12 + a22 * b22
    return (c11, c21, c12, c22)
end
function linear22(A::NTuple{4}, b::NTuple{2})
    a11, a21, a12, a22 = A
    b1, b2 = b
    det = a11*a22-a12*a21
    det == zero(det) && throw(LinAlg.SingularException(0))
    c1 = (+a22*b1-a12*b2)/det
    c2 = (-a21*b1+a11*b2)/det
    return (c1, c2)
end

matplus22(A::NTuple{4}, α) = (A[1] + α, A[2], A[3], A[4] + α)
function sylvester22(A::NTuple{4}, B::NTuple{4}, C::NTuple{4})
    b11, b21, b12, b22 = B

    c1 = (-C[1], -C[2])
    c2 = (-C[3], -C[4])

    A1 = matplus22(A, b11)
    A2 = matplus22(A, b22)

    y1 = gemv22(A2, c1, -b21, c2)
    x1 = linear22( matplus22( matmat22(A2,A1), -b12*b21), y1)

    y2 = gemv22(A1, c2, -b12, c1)
    x2 = linear22( matplus22( matmat22(A1,A2), -b12*b21), y2)

    return (x1..., x2...)
end
