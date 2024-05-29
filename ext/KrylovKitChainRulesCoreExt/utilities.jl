safe_inv(a, tol) = abs(a) < tol ? zero(a) : inv(a)

# vecs are assumed orthonormal
function orthogonalprojector(vecs, n)
    function projector(w)
        w′ = zerovector(w)
        @inbounds for i in 1:n
            w′ = VectorInterface.add!!(w′, vecs[i], inner(vecs[i], w))
        end
        return w′
    end
    return projector
end
function orthogonalcomplementprojector(vecs, n)
    function projector(w)
        w′ = scale(w, 1)
        @inbounds for i in 1:n
            w′ = VectorInterface.add!!(w′, vecs[i], -inner(vecs[i], w))
        end
        return w′
    end
    return projector
end
# vecs are not assumed orthonormal, G is the Cholesky factorisation of the overlap matrix
function orthogonalprojector(vecs, n, G::Cholesky)
    overlaps = zeros(eltype(G), n)
    function projector(w)
        @inbounds for i in 1:n
            overlaps[i] = inner(vecs[i], w)
        end
        overlaps = ldiv!(G, overlaps)
        w′ = zerovector(w)
        @inbounds for i in 1:n
            w′ = VectorInterface.add!!(w′, vecs[i], +overlaps[i])
        end
        return w′
    end
    return projector
end
function orthogonalcomplementprojector(vecs, n, G::Cholesky)
    overlaps = zeros(eltype(G), n)
    function projector(w)
        @inbounds for i in 1:n
            overlaps[i] = inner(vecs[i], w)
        end
        overlaps = ldiv!(G, overlaps)
        w′ = scale(w, 1)
        @inbounds for i in 1:n
            w′ = VectorInterface.add!!(w′, vecs[i], -overlaps[i])
        end
        return w′
    end
    return projector
end

function _realview(v::AbstractVector{Complex{T}}) where {T}
    v_real = reinterpret(T, v)
    return view(v_real, axes(v_real, 1)[begin:2:end])
end

function _imagview(v::AbstractVector{Complex{T}}) where {T}
    v_real = reinterpret(T, v)
    return view(v_real, axes(v_real, 1)[(begin + 1):2:end])
end
