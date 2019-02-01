"""
    schursolve(A, x₀, howmany, which, algorithm)

Compute a partial Schur decomposition containing `howmany` eigenvalues from the linear map
encoded in the matrix or function `A`. Return the reduced Schur matrix, the basis of Schur
vectors, the extracted eigenvalues and a `ConvergenceInfo` structure.

See also [`eigsolve`](@eigsolve) to obtain the eigenvectors instead. For real symmetric or
complex hermitian problems, the (partial) Schur decomposition is identical to the (partial)
eigenvalue decomposition, and `eigsolve` should always be used.

### Arguments:
The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or callable
object, that acts on vector like objects similar to `x₀`, which is the starting guess from
which a Krylov subspace will be built. `howmany` specifies how many Schur vectors should be
converged before the algorithm terminates; `which` specifies which eigenvalues should be targetted.
Valid specifications of `which` are
*   `LM`: eigenvalues of largest magnitude
*   `LR`: eigenvalues with largest (most positive) real part
*   `SR`: eigenvalues with smallest (most negative) real part
*   `LI`: eigenvalues with largest (most positive) imaginary part, only if `T <: Complex`
*   `SI`: eigenvalues with smallest (most negative) imaginary part, only if `T <: Complex`
*   [`ClosestTo(λ)`](@ref): eigenvalues closest to some number `λ`
!!! note "Note about selecting `which` eigenvalues"
    Krylov methods work well for extremal eigenvalues, i.e. eigenvalues on the periphery of
    the spectrum of the linear map. Even with `ClosestTo`, no shift and invert is performed.
    This is useful if, e.g., you know the spectrum to be within the unit circle in the complex
    plane, and want to target the eigenvalues closest to the value `λ = 1`.

The final argument `algorithm` can currently only be an instance of [`Arnoldi`](@ref), but
should nevertheless be specified. Since `schursolve` is less commonly used as `eigsolve`, no
convenient keyword syntax is currently available.

### Return values:
The return value is always of the form `T, vecs, vals, info = schursolve(...)` with
*   `T`: a `Matrix` containing the partial Schur decomposition of the linear map, i.e. it's
    elements are given by `T[i,j] = dot(vecs[i], f(vecs[j]))`. It is of Schur form, i.e.
    upper triangular in case of complex arithmetic, and block upper triangular (with at most
    2x2 blocks) in case of real arithmetic.
*   `vecs`: a `Vector` of corresponding Schur vectors, of the same length as `vals`. Note
    that Schur vectors are not returned as a matrix, as the linear map could act on any
    custom  Julia type with vector like behavior, i.e. the elements of the list `vecs` are
    objects that are typically similar to the starting guess `x₀`, up to a possibly
    different `eltype`. When the linear map is a simple `AbstractMatrix`, `vecs` will be
    `Vector{Vector{<:Number}}`. Schur vectors are by definition orthogonal, i.e.
    `dot(vecs[i],vecs[j]) = I[i,j]`. Note that Schur vectors are real if the problem (i.e.
    the linear map and the initial guess) are real.
*   `vals`: a `Vector` of eigenvalues, i.e. the diagonal elements of `T` in case of complex
    arithmetic, or extracted from the diagonal blocks in case of real arithmetic. Note that
    `vals` will always be complex, independent of the underlying arithmetic.
*   `info`: an object of type [`ConvergenceInfo`], which has the following fields
    -   `info.converged::Int`: indicates how many eigenvalues and Schur vectors were
        actually converged to the specified tolerance (see below under keyword arguments)
    -   `info.residuals::Vector`: a list of the same length as `vals` containing the actual
        residuals
        ```julia
          info.residuals[i] = f(vecs[i]) - sum(vecs[j]*T[j,i] for j = 1:i+1)
        ```
        where `T[i+1,i]` is definitely zero in case of complex arithmetic and possibly zero
        in case of real arithmetic
    -   `info.normres::Vector{<:Real}`: list of the same length as `vals` containing the
        norm of the residual for every Schur vector, i.e.
        `info.normes[i] = norm(info.residual[i])`
    -   `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
    -   `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)
!!! warning "Check for convergence"
    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged >= howmany`.

### Algorithm
The actual algorithm is an implementation of the Krylov-Schur algorithm, where the
[`Arnoldi`](@ref) algorithm is used to generate the Krylov subspace. During the algorith,
the Krylov subspace is dynamically grown and shrunk, i.e. the restarts are so-called thick
restarts where a part of the current Krylov subspace is kept.
"""
function schursolve(A, x₀, howmany::Int, which::Selector, alg::Arnoldi)
    T, U, fact, converged, numiter, numops = _schursolve(A, x₀, howmany, which, alg)
    if eltype(T) <: Real && howmany < length(fact) && T[howmany+1,howmany] != 0
        howmany += 1
    end
    if converged > howmany
        howmany = converged
    end
    TT = view(T,1:howmany,1:howmany)
    values = schur2eigvals(TT)
    vectors = let B = basis(fact)
        [B*u for u in cols(U, 1:howmany)]
    end
    residuals = let r = residual(fact)
        [mul!(similar(r), r, last(u)) for u in cols(U, 1:howmany)]
    end
    normresiduals = [normres(fact)*abs(last(u)) for u in cols(U, 1:howmany)]

    if alg.verbosity > 0
        if converged < howmany
            @warn """Arnoldi schursolve finished without convergence after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))"""
        else
            @info """Arnoldi schursolve finished after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))"""
        end
    end
    return TT, vectors, values, ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

function eigsolve(A, x₀, howmany::Int, which::Selector, alg::Arnoldi)
    T, U, fact, converged, numiter, numops = _schursolve(A, x₀, howmany, which, alg)
    if eltype(T) <: Real && howmany < length(fact) && T[howmany+1,howmany] != 0
        howmany += 1
    end
    if converged > howmany
        howmany = converged
    end
    TT = view(T,1:howmany,1:howmany)
    values = schur2eigvals(TT)

    # Compute eigenvectors
    V = view(U, :, 1:howmany)*schur2eigvecs(TT)
    vectors = let B = basis(fact)
        [B*v for v in cols(V)]
    end
    residuals = let r = residual(fact)
        [mul!(similar(r, Base.promote_type(eltype(V), eltype(r))), r, last(v)) for v in cols(V)]
    end
    normresiduals = [normres(fact)*abs(last(v)) for v in cols(V)]

    if alg.verbosity > 0
        if converged < howmany
            @warn """Arnoldi eigsolve finished without convergence after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        else
            @info """Arnoldi eigsolve finished after $numiter iterations:
             *  $converged eigenvalues converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        end
    end
    return values, vectors, ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

function _schursolve(A, x₀, howmany::Int, which::Selector, alg::Arnoldi)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim && error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")

    ## FIRST ITERATION: setting up
    numiter = 1
    # Compute arnoldi factorization
    iter = ArnoldiIterator(A, x₀, alg.orth)
    fact = initialize(iter; verbosity = alg.verbosity - 2)
    numops = 1
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::eltype(β) = alg.tol
    if normres(fact) > tol || howmany > 1
        while length(fact) < krylovdim
            fact = expand!(iter, fact; verbosity = alg.verbosity-2)
            numops += 1
            normres(fact) < tol && length(fact) >= howmany && break
        end
    end

    # Process
    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim+1, krylovdim)
    UU = fill(zero(eltype(fact)), krylovdim, krylovdim)

    # initialize
    β = normres(fact)
    m = length(fact)
    H = view(HH, 1:m, 1:m)
    U = view(UU, 1:m, 1:m)
    f = view(HH, m+1, 1:m)
    copyto!(U, I)
    copyto!(H, rayleighquotient(fact))

    # compute dense schur factorization
    T, U, values = hschur!(H, U)
    by, rev = eigsort(which)
    p = sortperm(values, by = by, rev = rev)
    T, U = permuteschur!(T, U, p)
    f = mul!(f, view(U, m, :), β)
    converged = 0
    while converged < length(fact) && abs(f[converged+1]) < tol
        converged += 1
    end
    if eltype(T) <: Real && 0< converged < length(fact) && T[converged+1,converged] != 0
        converged -= 1
    end

    if alg.verbosity > 1
        msg = "Arnoldi schursolve in iter $numiter: "
        msg *= "$converged values converged, normres = ("
        msg *= @sprintf("%.2e", abs(f[1]))
        for i = 2:howmany
            msg *= ", "
            msg *= @sprintf("%.2e", abs(f[i]))
        end
        msg *= ")"
        @info msg
    end
    ## OTHER ITERATIONS: recycle
    while numiter < maxiter && converged < howmany
        numiter += 1

        # Determine how many to keep
        keep = div(3*m + 2*converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged
        if eltype(H) <: Real && H[keep+1,keep] != 0 # we are in the middle of a 2x2 block
            keep += 1 # conservative choice
            keep >= krylovdim && error("krylov dimension $(krylovdim) too small to compute $howmany eigenvalues")
        end

        # Restore Arnoldi form in the first keep columns
        @inbounds for j = 1:keep
            H[keep+1,j] = f[j]
        end
        @inbounds for j = keep:-1:1
            h, ν = householder(H, j+1, 1:j, j)
            H[j+1,j] = ν
            H[j+1,1:j-1] .= 0
            lmul!(h, H)
            rmul!(view(H, 1:j,:), h')
            rmul!(U, h')
        end
        copyto!(rayleighquotient(fact), H) # copy back into fact

        # Update B by applying U
        B = basis(fact)
        basistransform!(B, view(U, :, 1:keep))
        # for j = 1:m
        #     h, ν = householder(U, j:m, j)
        #     lmul!(h, view(U, :, j+1:krylovdim))
        #     rmul!(B, h')
        # end
        r = residual(fact)
        B[keep+1] = rmul!(r, 1/normres(fact))

        # Shrink Arnoldi factorization
        fact = shrink!(fact, keep)

        # Arnoldi factorization: recylce fact
        while length(fact) < krylovdim
            fact = expand!(iter, fact; verbosity = alg.verbosity-2)
            numops += 1
            normres(fact) < tol && length(fact) >= howmany && break
        end

        # post process
        β = normres(fact)
        m = length(fact)
        H = view(HH, 1:m, 1:m)
        U = view(UU, 1:m, 1:m)
        f = view(HH, m+1, 1:m)
        copyto!(U, I)
        copyto!(H, rayleighquotient(fact))

        # compute dense schur factorization
        T, U, values = hschur!(H, U)
        by, rev = eigsort(which)
        p = sortperm(values, by = by, rev = rev)
        T, U = permuteschur!(T, U, p)
        mul!(f, view(U,m,:), β)
        converged = 0
        while converged < length(fact) && abs(f[converged+1]) < tol
            converged += 1
        end
        if eltype(T) <: Real && 0 < converged < length(fact) && T[converged+1,converged] != 0
            converged -= 1
        end
        if alg.verbosity > 1
            msg = "Arnoldi schursolve in iter $numiter: "
            msg *= "$converged values converged, normres = ("
            msg *= @sprintf("%.2e", abs(f[1]))
            for i = 2:howmany
                msg *= ", "
                msg *= @sprintf("%.2e", abs(f[i]))
            end
            msg *= ")"
            @info msg
        end
    end
    return T, U, fact, converged, numiter, numops
end
