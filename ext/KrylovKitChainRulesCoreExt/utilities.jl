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

function reverse_which(which)
    by, rev = KrylovKit.eigsort(which)
    return EigSorter(by ∘ conj, rev)
end

function _prepare_inputs(config, f, vecs, alg_primal)
    T = scalartype(vecs[1])
    config isa RuleConfig{>:HasReverseMode} ||
        throw(ArgumentError("`eigsolve` reverse-mode AD requires AD engine that supports calling back into AD"))
    f_pullbacks = map(x -> rrule_via_ad(config, f, x)[2], vecs)
    if alg_primal isa Lanczos
        fᴴ = v -> f(v)
    else
        fᴴ = v -> f_pullbacks[1](v)[2]
    end
    construct∂f = let f_pullbacks = f_pullbacks
        function (ws)
            ∂f = f_pullbacks[1](ws[1])[1]
            for i in 2:length(ws)
                ∂f = ChainRulesCore.add!!(∂f, f_pullbacks[i](ws[i])[1])
            end
            return ∂f
        end
    end
    return T, fᴴ, construct∂f
end

function _prepare_inputs(config, A::AbstractMatrix, vecs, alg_primal)
    T = scalartype(vecs) # will be real for real symmetric problems and complex otherwise
    fᴴ = v -> A' * v
    if A isa StridedMatrix
        construct∂A = ws -> InplaceableThunk(Ā -> _buildĀ!(Ā, ws, vecs),
                                             @thunk(_buildĀ!(zero(A), ws, vecs)))
    else
        construct∂A = let project_A = ProjectTo(A)
            ws -> @thunk(project_A(_buildĀ!(zero(A), ws, vecs)))
        end
    end
    return T, fᴴ, construct∂A
end

function _buildĀ!(Ā, ws, vs)
    for i in 1:length(ws)
        w = ws[i]
        v = vs[i]
        if !(w isa AbstractZero)
            if eltype(Ā) <: Real && eltype(w) <: Complex
                mul!(Ā, _realview(w), _realview(v)', +1, +1)
                mul!(Ā, _imagview(w), _imagview(v)', +1, +1)
            else
                mul!(Ā, w, v', +1, 1)
            end
        end
    end
    return Ā
end

function _realview(v::AbstractVector{Complex{T}}) where {T}
    v_real = reinterpret(T, v)
    return view(v_real, axes(v_real, 1)[begin:2:end])
end

function _imagview(v::AbstractVector{Complex{T}}) where {T}
    v_real = reinterpret(T, v)
    return view(v_real, axes(v_real, 1)[(begin+1):2:end])
end
