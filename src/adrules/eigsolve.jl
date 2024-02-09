function ChainRulesCore.rrule(::typeof(eigsolve),
                              A::AbstractMatrix,
                              x₀,
                              howmany,
                              which,
                              alg)
    (vals, vecs, info) = eigsolve(A, x₀, howmany, which, alg)
    project_A = ProjectTo(A)
    T = scalartype(vecs[1]) # will be real for real symmetric problems and complex otherwise

    function eigsolve_pullback(ΔX)
        _Δvals = unthunk(ΔX[1])
        _Δvecs = unthunk(ΔX[2])

        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂howmany = NoTangent()
        ∂which = NoTangent()
        ∂alg = NoTangent()
        if _Δvals isa AbstractZero && _Δvecs isa AbstractZero
            ∂A = ZeroTangent()
            return ∂self, ∂A, ∂x₀, ∂howmany, ∂which, ∂alg
        end

        if _Δvals isa AbstractZero
            Δvals = fill(NoTangent(), length(Δvecs))
        else
            Δvals = _Δvals
        end
        if _Δvecs isa AbstractZero
            Δvecs = fill(NoTangent(), length(Δvals))
        else
            Δvecs = _Δvecs
        end

        @assert length(Δvals) == length(Δvecs)
        @assert length(Δvals) <= length(vals)

        # Determine algorithm to solve linear problem
        # TODO: Is there a better choice? Should we make this user configurable?
        linalg = GMRES(;
                       tol=alg.tol,
                       krylovdim=alg.krylovdim,
                       maxiter=alg.maxiter,
                       orth=alg.orth)

        ws = similar(vecs, length(Δvecs))
        for i in 1:length(Δvecs)
            Δλ = Δvals[i]
            Δv = Δvecs[i]
            λ = vals[i]
            v = vecs[i]

            # First threat special cases
            if isa(Δv, AbstractZero) && isa(Δλ, AbstractZero) # no contribution
                ws[i] = Δv # some kind of zero
                continue
            end
            if isa(Δv, AbstractZero) && isa(alg, Lanczos) # simple contribution
                ws[i] = Δλ * v
                continue
            end

            # General case :
            if isa(Δv, AbstractZero)
                b = RecursiveVec(zero(T) * v, T[Δλ])
            else
                @assert isa(Δv, typeof(v))
                b = RecursiveVec(Δv, T[Δλ])
            end

            if i > 1 && eltype(A) <: Real &&
               vals[i] == conj(vals[i - 1]) && Δvals[i] == conj(Δvals[i - 1]) &&
               vecs[i] == conj(vecs[i - 1]) && Δvecs[i] == conj(Δvecs[i - 1])
                ws[i] = conj(ws[i - 1])
                continue
            end

            w, reverse_info = let λ = λ, v = v, Aᴴ = A'
                linsolve(b, zero(T) * b, linalg) do x
                    x1, x2 = x
                    γ = 1
                    # γ can be chosen freely and does not affect the solution theoretically
                    # The current choice guarantees that the extended matrix is Hermitian if A is
                    # TODO: is this the best choice in all cases?
                    y1 = axpy!(-γ * x2[], v, axpy!(-conj(λ), x1, A' * x1))
                    y2 = T[-dot(v, x1)]
                    return RecursiveVec(y1, y2)
                end
            end
            if info.converged >= i && reverse_info.converged == 0
                @warn "The cotangent linear problem did not converge, whereas the primal eigenvalue problem did."
            end
            ws[i] = w[1]
        end

        if A isa StridedMatrix
            ∂A = InplaceableThunk(Ā -> _buildĀ!(Ā, ws, vecs),
                                  @thunk(_buildĀ!(zero(A), ws, vecs)))
        else
            ∂A = @thunk(project_A(_buildĀ!(zero(A), ws, vecs)))
        end
        return ∂self, ∂A, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return (vals, vecs, info), eigsolve_pullback
end

function _buildĀ!(Ā, ws, vs)
    for i in 1:length(ws)
        w = ws[i]
        v = vs[i]
        if !(w isa AbstractZero)
            if eltype(Ā) <: Real && eltype(w) <: Complex
                mul!(Ā, _realview(w), _realview(v)', -1, 1)
                mul!(Ā, _imagview(w), _imagview(v)', -1, 1)
            else
                mul!(Ā, w, v', -1, 1)
            end
        end
    end
    return Ā
end
function _realview(v::AbstractVector{Complex{T}}) where {T}
    return view(reinterpret(T, v), 2 * (1:length(v)) .- 1)
end
function _imagview(v::AbstractVector{Complex{T}}) where {T}
    return view(reinterpret(T, v), 2 * (1:length(v)))
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode},
                              ::typeof(eigsolve),
                              A::AbstractMatrix,
                              x₀,
                              howmany,
                              which,
                              alg)
    return ChainRulesCore.rrule(eigsolve, A, x₀, howmany, which, alg)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode},
                              ::typeof(eigsolve),
                              f,
                              x₀,
                              howmany,
                              which,
                              alg)
    (vals, vecs, info) = eigsolve(f, x₀, howmany, which, alg)
    T = scalartype(vecs)
    f_pullbacks = map(x -> rrule_via_ad(config, f, x)[2], vecs)

    function eigsolve_pullback(ΔX)
        _Δvals = unthunk(ΔX[1])
        _Δvecs = unthunk(ΔX[2])

        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂howmany = NoTangent()
        ∂which = NoTangent()
        ∂alg = NoTangent()
        if _Δvals isa AbstractZero && _Δvecs isa AbstractZero
            ∂A = ZeroTangent()
            return (∂self, ∂A, ∂x₀, ∂howmany, ∂which, ∂alg)
        end

        if _Δvals isa AbstractZero
            Δvals = fill(NoTangent(), length(_Δvecs))
        else
            Δvals = _Δvals
        end
        if _Δvecs isa AbstractZero
            Δvecs = fill(NoTangent(), length(_Δvals))
        else
            Δvecs = _Δvecs
        end
        @assert length(Δvals) == length(Δvecs)

        # Determine algorithm to solve linear problem
        # TODO: Is there a better choice? Should we make this user configurable?
        linalg = GMRES(;
                       tol=alg.tol,
                       krylovdim=alg.krylovdim,
                       maxiter=alg.maxiter,
                       orth=alg.orth)
        # linalg = BiCGStab(;
        #     tol = alg.tol,
        #     maxiter = alg.maxiter*alg.krylovdim,
        # )

        ws = similar(Δvecs)
        for i in 1:length(Δvecs)
            Δλ = Δvals[i]
            Δv = Δvecs[i]
            λ = vals[i]
            v = vecs[i]

            # First threat special cases
            if isa(Δv, AbstractZero) && isa(Δλ, AbstractZero) # no contribution
                ws[i] = Δv # some kind of zero
                continue
            end
            if isa(Δv, AbstractZero) && isa(alg, Lanczos) # simple contribution
                ws[i] = scale!!(ws[i], v, Δλ)
                continue
            end

            # General case :
            b2 = Δλ isa AbstractZero ? zero(T) : T(-Δλ)
            if isa(Δv, AbstractZero)
                b = (zerovector(v), b2)
            else
                @assert isa(Δv, typeof(v))
                b = (scale(Δv, -one(T)), b2)
            end

            # TODO: is there any analogy to this for general vector-like user types
            # if i > 1 && eltype(A) <: Real &&
            #     vals[i] == conj(vals[i-1]) && Δvals[i] == conj(Δvals[i-1]) &&
            #     vecs[i] == conj(vecs[i-1]) && Δvecs[i] == conj(Δvecs[i-1])
            #
            #     ws[i] = conj(ws[i-1])
            #     continue
            # end

            w, reverse_info = let λ = λ, v = v, fᴴ = x -> f_pullbacks[i](x)[2]
                linsolve(b, zerovector(b), linalg) do x
                    x1, x2 = x
                    γ = 1
                    # γ can be chosen freely and does not affect the solution theoretically
                    # The current choice guarantees that the extended matrix is Hermitian if A is
                    # TODO: is this the best choice in all cases?
                    y1 = add!!(add!!(fᴴ(x1), x1, -conj(λ)), v, -γ * x2)
                    # y1 = axpy!(-γ * x2[], v, axpy!(-conj(λ), x1, fᴴ(x1)))
                    y2 = -inner(v, x1)
                    return (y1, y2)
                end
            end
            if info.converged >= i && reverse_info.converged == 0
                @warn "The cotangent linear problem ($i) did not converge, whereas the primal eigenvalue problem did."
            end
            ws[i] = w[1]
        end

        ∂f = f_pullbacks[1](ws[1])[1]
        for i in 2:length(ws)
            ∂f = add!!(∂f, f_pullbacks[i](ws[i])[1])
        end
        return ∂self, ∂f, ∂x₀, ∂howmany, ∂which, ∂alg
    end
    return (vals, vecs, info), eigsolve_pullback
end
