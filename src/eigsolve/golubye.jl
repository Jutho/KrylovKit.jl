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
    xax = dot(x₀, ax₀) / β₀^2
    xbx = dot(x₀, bx₀) / β₀^2
    T = promote_type(typeof(xax), typeof(xbx))
    invβ₀ = one(T) / β₀
    v = invβ₀ * x₀ # v = mul!(similar(x₀, T), x₀, invβ₀)
    av = mul!(similar(v), ax₀, invβ₀)
    bv = mul!(similar(v), bx₀, invβ₀)
    ρ = checkhermitian(xax) / checkposdef(xbx)
    r = axpy!(-ρ, bv, av)
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
    r, α = orthogonalize!(r, v, alg.orth) # α should be zero, otherwise ρ was miscalculated
    β = norm(r)
    converged = 0

    values = resize!(Vector{typeof(ρ)}(undef, howmany), 0)
    vectors = resize!(Vector{typeof(v)}(undef, howmany), 0)
    residuals = resize!(Vector{typeof(r)}(undef, howmany), 0)
    normresiduals = resize!(Vector{typeof(β)}(undef, howmany), 0)

    K = 1
    HHA[K, K] = real(α)
    while true
        β = norm(r)
        if β <= tol && K < howmany
            @warn "Invariant subspace of dimension $K (up to requested tolerance `tol = $tol`), which is smaller than the number of requested eigenvalues (i.e. `howmany == $howmany`); setting `howmany = $K`."
            howmany = K
        end
        if K == krylovdim - converged || β <= tol # process
            if numiter > 1
                # add vold - v, or thus just vold as v is first vector in subspace
                v, = orthonormalize!(vold, V, alg.orth)
                av, bv = genapply(f, v)
                numops += 1
                av = axpy!(-ρ, bv, av)
                for i in 1:K
                    HHA[i, K+1] = dot(V[i], av)
                    HHA[K+1, i] = conj(HHA[i, K+1])
                end
                K += 1
                HHA[K, K] = checkhermitian(dot(v, av))
                push!(V, v)
                push!(BV, bv)
            end
            for i in 1:converged
                # add converged vectors
                v, = orthonormalize(vectors[i], V, alg.orth)
                av, bv = genapply(f, v)
                numops += 1
                av = axpy!(-ρ, bv, av)
                for j in 1:K
                    HHA[j, K+1] = dot(V[j], av)
                    HHA[K+1, j] = conj(HHA[j, K+1])
                end
                K += 1
                HHA[K, K] = checkhermitian(dot(v, av))
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
            p = sortperm(D, by = by, rev = rev)

            # replace vold
            vold = V[1]

            converged = 0
            resize!(values, 0)
            resize!(vectors, 0)
            resize!(residuals, 0)
            resize!(normresiduals, 0)
            while converged < K
                z = view(Z, :, p[converged+1])
                v = mul!(similar(vold), V, z)
                av, bv = genapply(f, v)
                numops += 1
                ρ = checkhermitian(dot(v, av)) / checkposdef(dot(v, bv))
                r = axpy!(-ρ, bv, av)
                β = norm(r)

                if β > tol * norm(z)
                    break
                end

                push!(values, ρ)
                push!(vectors, v)
                push!(residuals, r)
                push!(normresiduals, β)
                converged += 1
            end

            if converged >= howmany
                howmany = converged
                break
            elseif numiter == maxiter
                for k in converged+1:howmany
                    z = view(Z, :, p[k])
                    v = mul!(similar(vold), V, z)
                    av, bv = genapply(f, v)
                    numops += 1
                    ρ = checkhermitian(dot(v, av)) / checkposdef(dot(v, bv))
                    r = axpy!(-ρ, bv, av)
                    β = norm(r)
            
                    push!(values, ρ)
                    push!(vectors, v)
                    push!(residuals, r)
                    push!(normresiduals, β)
                end
            elseif alg.verbosity > 1
                msg = "Golub-Ye geneigsolve in iter $numiter: "
                msg *= "$converged values converged, normres = ("
                for i in 1:converged
                    msg *= @sprintf("%.2e", normresiduals[i])
                    msg *= ", "
                end
                msg *= @sprintf("%.2e", β) * ")"
                @info msg
            end
        end

        if K < krylovdim - converged
            # expand
            v = rmul!(r, 1 / β)
            push!(V, v)
            HHA[K+1, K] = β
            HHA[K, K+1] = β
            βold = β
            r, α, β, bv = golubyerecurrence(f, ρ, V, βold, alg.orth)
            numops += 1
            K += 1
            n = hypot(α, β, βold)
            HHA[K, K] = checkhermitian(α, n)
            push!(BV, bv)

            if alg.verbosity > 2
                @info "Golub-Ye iteration $numiter, step $K: normres = $β"
            end
        else # restart
            numiter == maxiter && break
            resize!(V, 0)
            resize!(BV, 0)    
            fill!(HHA, zero(T))
            fill!(HHB, zero(T))
            K = 1

            invβ = 1 / norm(v)
            v = rmul!(v, invβ)
            bv = rmul!(bv, invβ)
            r = rmul!(r, invβ)
            r, α = orthogonalize!(r, v, alg.orth) # α should be zero, otherwise ρ was miscalculated
            β = norm(r)
            push!(V, v)
            HHA[K, K] = real(α)
            push!(BV, bv)
            numiter += 1
        end
    end
    if alg.verbosity > 0
        if converged < howmany
            @warn """Golub-Ye geneigsolve finished without convergence after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        else
            @info """Golub-Ye geneigsolve finished after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        end
    end

    return values,
    vectors,
    ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt)
    v = V[end]
    av, bv = genapply(f, v)
    w = axpy!(-ρ, bv, av)
    α = dot(v, w)

    w = axpy!(-β, V[end-1], w)
    w = axpy!(-α, v, w)
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt)
    v = V[end]
    av, bv = genapply(f, v)
    w = axpy!(-ρ, bv, av)
    w = axpy!(-β, V[end-1], w)

    w, α = orthogonalize!(w, v, orth)
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ClassicalGramSchmidt2)
    v = V[end]
    av, bv = genapply(f, v)
    w = axpy!(-ρ, bv, av)
    α = dot(v, w)
    w = axpy!(-β, V[end-1], w)
    w = axpy!(-α, v, w)

    w, s = orthogonalize!(w, V, ClassicalGramSchmidt())
    α += s[end]
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ModifiedGramSchmidt2)
    v = V[end]
    av, bv = genapply(f, v)
    w = axpy!(-ρ, bv, av)
    w = axpy!(-β, V[end-1], w)
    w, α = orthogonalize!(w, v, ModifiedGramSchmidt())

    s = α
    for q in V
        w, s = orthogonalize!(w, q, ModifiedGramSchmidt())
    end
    α += s
    β = norm(w)
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ClassicalGramSchmidtIR)
    v = V[end]
    av, bv = genapply(f, v)
    w = axpy!(-ρ, bv, av)
    α = dot(v, w)
    w = axpy!(-β, V[end-1], w)
    w = axpy!(-α, v, w)

    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β) + ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        w, s = orthogonalize!(w, V, ClassicalGramSchmidt())
        α += s[end]
        β = norm(w)
    end
    return w, α, β, bv
end
function golubyerecurrence(f, ρ, V::OrthonormalBasis, β, orth::ModifiedGramSchmidtIR)
    v = V[end]
    av, bv = genapply(f, v)
    w = axpy!(-ρ, bv, av)
    w = axpy!(-β, V[end-1], w)

    w, α = orthogonalize!(w, v, ModifiedGramSchmidt())
    ab2 = abs2(α) + abs2(β)
    β = norm(w)
    nold = sqrt(abs2(β) + ab2)
    while eps(one(β)) < β < orth.η * nold
        nold = β
        s = zero(α)
        for q in V
            w, s = orthogonalize!(w, q, ModifiedGramSchmidt())
        end
        α += s
        β = norm(w)
    end
    return w, α, β, bv
end

function buildHB!(HB, V, BV)
    m = length(V)
    @inbounds for j in 1:m
        HB[j, j] = checkposdef(dot(V[j], BV[j]))
        for i in j+1:m
            HB[i, j] = dot(V[i], BV[j])
            HB[j, i] = conj(HB[i, j])
        end
    end
end
