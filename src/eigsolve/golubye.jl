function geneigsolve(f, x₀, howmany::Int, which::Selector, alg::GolubYe)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim &&
        error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    ax₀, bx₀ = genapply(f, x₀)
    numops = 1
    β₀ = norm(x₀)
    iszero(β₀) && throw(ArgumentError("initial vector should not have norm zero"))
    xax = inner(x₀, ax₀) / β₀^2
    xbx = inner(x₀, bx₀) / β₀^2
    T = promote_type(typeof(xax), typeof(xbx))
    invβ₀ = one(T) / β₀
    v = scale(x₀, invβ₀)
    av = scale!!(zerovector(v), ax₀, invβ₀)
    bv = scale!!(zerovector(v), bx₀, invβ₀)
    ρ = checkhermitian(xax) / checkposdef(xbx)
    r = add!!(av, bv, -ρ)
    tol::typeof(ρ) = alg.tol

    # allocate storage
    HHA = fill(zero(T), krylovdim + 1, krylovdim + 1)
    HHB = fill(zero(T), krylovdim + 1, krylovdim + 1)

    # Start Lanczos iteration with A - ρ ⋅ B
    numiter = 1
    vold = v
    V = OrthonormalBasis([v])
    BV = [bv]
    sizehint!(V, krylovdim + 1)
    sizehint!(BV, krylovdim + 1)

    r, α = orthogonalize!!(r, v, alg.orth) # α should be zero, otherwise ρ was miscalculated
    β = norm(r)
    converged = 0

    values = resize!(Vector{typeof(ρ)}(undef, howmany), 0)
    vectors = resize!(Vector{typeof(v)}(undef, howmany), 0)
    residuals = resize!(Vector{typeof(r)}(undef, howmany), 0)
    normresiduals = resize!(Vector{typeof(β)}(undef, howmany), 0)

    K = 1
    HHA[K, K] = real(α)
    if alg.verbosity >= EACHITERATION_LEVEL + 1
        @info "Golub-Ye iteration $numiter, step $K: normres = $(normres2string(β))"
    end
    while true
        β = norm(r)
        if β <= tol && K < howmany
            if alg.verbosity >= WARN_LEVEL
                msg = "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), "
                msg *= "which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`);"
                msg *= "setting `howmany = $K`."
                @warn msg
            end
            howmany = K
        end
        if K == krylovdim - converged || β <= tol # process
            if numiter > 1
                # add vold - v, or thus just vold as v is first vector in subspace
                v, = orthonormalize!!(vold, V, alg.orth)
                av, bv = genapply(f, v)
                numops += 1
                av = add!!(av, bv, -ρ)
                for i in 1:K
                    HHA[i, K + 1] = inner(V[i], av)
                    HHA[K + 1, i] = conj(HHA[i, K + 1])
                end
                K += 1
                HHA[K, K] = checkhermitian(inner(v, av))
                push!(V, v)
                push!(BV, bv)
            end
            for i in 1:converged
                # add converged vectors
                v, = orthonormalize(vectors[i], V, alg.orth)
                av, bv = genapply(f, v)
                numops += 1
                av = add!!(av, bv, -ρ)
                for j in 1:K
                    HHA[j, K + 1] = inner(V[j], av)
                    HHA[K + 1, j] = conj(HHA[j, K + 1])
                end
                K += 1
                HHA[K, K] = checkhermitian(inner(v, av))
                push!(V, v)
                push!(BV, bv)
            end

            # Process
            HA = view(HHA, 1:K, 1:K)
            HB = view(HHB, 1:K, 1:K)
            buildHB!(HB, V, BV)
            HA .+= ρ .* HB

            D, Z = geneigh!(HA, HB)
            by, rev = eigsort(which)
            p = sortperm(D; by, rev)
            xold = V[1]

            converged = 0
            resize!(values, 0)
            resize!(vectors, 0)
            resize!(residuals, 0)
            resize!(normresiduals, 0)
            for k in 1:K
                z = view(Z, :, p[k])
                v = unproject!!(zerovector(vold), V, z)
                av, bv = genapply(f, v)
                numops += 1
                ρ = checkhermitian(inner(v, av)) / checkposdef(inner(v, bv))
                r = add!!(av, bv, -ρ)
                β = norm(r)

                if β < tol * norm(z)
                    converged += 1
                elseif numiter < maxiter
                    break # in last iteration, keep adding nonconverged vectors up to howmany
                end
                push!(values, ρ)
                push!(vectors, v)
                push!(residuals, r)
                push!(normresiduals, β)
                if (k == howmany && numiter == maxiter)
                    break
                end
            end
            if converged >= howmany
                howmany = converged
                break
            end
            if alg.verbosity >= EACHITERATION_LEVEL
                @info "Golub-Ye geneigsolve in iter $numiter: $converged values converged, normres = $(normres2string(normresiduals))"
            end
        end

        if K < krylovdim - converged
            # expand
            v = scale!!(r, 1 / β)
            push!(V, v)
            HHA[K + 1, K] = β
            HHA[K, K + 1] = β
            βold = β
            r, α, β, bv = golubyerecurrence(f, ρ, V, βold, alg.orth)
            numops += 1
            K += 1
            n = hypot(α, β, βold)
            HHA[K, K] = checkhermitian(α, n)
            push!(BV, bv)

            if alg.verbosity >= EACHITERATION_LEVEL + 1
                @info "Golub-Ye iteration $numiter, step $K: normres = $(normres2string(β))"
            end
        else # restart
            numiter == maxiter && break
            resize!(V, 0)
            resize!(BV, 0)
            fill!(HHA, zero(T))
            fill!(HHB, zero(T))
            K = 1

            invβ = 1 / norm(v)
            v = scale!!(v, invβ)
            bv = scale!!(bv, invβ)
            r = scale!!(r, invβ)
            r, α = orthogonalize!!(r, v, alg.orth) # α should be zero, otherwise ρ was miscalculated
            β = norm(r)
            push!(V, v)
            HHA[K, K] = real(α)
            push!(BV, bv)
            numiter += 1
            if alg.verbosity >= EACHITERATION_LEVEL + 1
                @info "Golub-Ye iteration $numiter, step $K: normres = $(normres2string(β))"
            end
        end
    end
    if (converged < howmany) && alg.verbosity >= WARN_LEVEL
        @warn """Golub-Ye geneigsolve stopped without convergence after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info """Golub-Ye geneigsolve finished after $numiter iterations:
        * $converged eigenvalues converged
        * norm of residuals = $(normres2string(normresiduals))
        * number of operations = $numops"""
    end

    return values, vectors,
           ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!!(av, bv, -ρ)
    α = inner(v, w)

    w = add!!(w, V[end - 1], -β)
    w = add!!(w, v, -α)
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!!(av, bv, -ρ)
    w = add!!(w, V[end - 1], -β)

    w, α = orthogonalize!!(w, v, orth)
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt2)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!!(av, bv, -ρ)
    α = inner(v, w)
    w = add!!(w, V[end - 1], -β)
    w = add!!(w, v, -α)

    w, s = orthogonalize!!(w, V, ClassicalGramSchmidt())
    α += s[end]
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt2)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!!(av, bv, -ρ)
    w = add!!(w, V[end - 1], -β)
    w, α = orthogonalize!!(w, v, ModifiedGramSchmidt())

    s = α
    for q in V
        w, s = orthogonalize!!(w, q, ModifiedGramSchmidt())
    end
    α += s
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ClassicalGramSchmidtIR)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!!(av, bv, -ρ)
    α = inner(v, w)
    w = add!!(w, V[end - 1], -β)
    w = add!!(w, v, -α)

    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β) + ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        w, s = orthogonalize!!(w, V, ClassicalGramSchmidt())
        α += s[end]
        β = norm(w)
    end
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ModifiedGramSchmidtIR)
    v = V[end]
    av, bv = genapply(f, v)
    w = add!!(av, bv, -ρ)
    w = add!!(w, V[end - 1], -β)

    w, α = orthogonalize!!(w, v, ModifiedGramSchmidt())
    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β) + ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        s = zero(α)
        for q in V
            w, s = orthogonalize!!(w, q, ModifiedGramSchmidt())
        end
        α += s
        β = norm(w)
    end
    return w, α, β, bv
end

function buildHB!(HB, V, BV)
    m = length(V)
    return @inbounds for j in 1:m
        HB[j, j] = checkposdef(inner(V[j], BV[j]))
        for i in (j + 1):m
            HB[i, j] = inner(V[i], BV[j])
            HB[j, i] = conj(HB[i, j])
        end
    end
end
