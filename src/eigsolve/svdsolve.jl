"""
    svdsolve(A::AbstractMatrix, [howmany = 1, which = :LR, T = eltype(A)]; kwargs...)
    svdsolve(f, m::Int, n::Int, [howmany = 1, which = :LR, T = Float64]; kwargs...)
    svdsolve(f, x₀, y₀, [howmany = 1, which = :LM]; kwargs...)
    svdsolve(f, x₀, y₀, howmany, which, algorithm)

Compute `howmany` singular values from the linear map encoded in the matrix `A` or by the
function `f`. Return singular values, left and right singular vectors and a
`ConvergenceInfo` structure.

### Arguments:
The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or
callable object. Since both the action of the linear map and its adjoint are required in
order to compute singular values, `f` can either be a tuple of two callable objects (each
accepting a single argument), representing the linear map and its adjoint respectively, or,
`f` can be a single callable object that accepts two input arguments, where the second
argument is a flag that indicates whether the adjoint or the normal action of the linear
map needs to be computed. The latter form still combines well with the `do` block syntax of
Julia, as in
```julia
vals, lvecs, rvecs, info = svdsolve(x₀, y₀, howmany, which; kwargs...) do (x, flag)
    if flag
        # y = compute action of adjoint map on x
    else
        # y = compute action of linear map on x
    end
    return y
end
```

For a general linear map encoded using either the tuple or the two-argument form, the best
approach is to provide a start vector `x₀` (in the domain of the linear map).
Alternatively, one can specify the number `n` of columns of the linear map, in which case
`x₀ = rand(T, n)` is used, where the default value of `T` is `Float64`, unless specified
differently. If an `AbstractMatrix` is used, a starting vector `x₀` does not need to be
provided; it is chosen as `rand(T, size(A,1))`.

The next arguments are optional, but should typically be specified. `howmany` specifies how
many singular values and vectors should be computed; `which` specifies which singular
values should be targetted. Valid specifications of `which` are
*   `LR`: largest singular values
*   `SR`: smallest singular values
However, the largest singular values tend to converge more rapidly.

### Return values:
The return value is always of the form `vals, lvecs, rvecs, info = svdsolve(...)` with
*   `vals`: a `Vector{<:Real}` containing the singular values, of length at least `howmany`,
    but could be longer if more singular values were converged at the same cost.
*   `lvecs`: a `Vector` of corresponding left singular vectors, of the same length as
    `vals`.
*   `rvecs`: a `Vector` of corresponding right singular vectors, of the same length as
    `vals`. Note that singular vectors are not returned as a matrix, as the linear map
    could act on any custom Julia type with vector like behavior, i.e. the elements of the
    lists `lvecs`(`rvecs`) are objects that are typically similar to the starting guess `y₀`
    (`x₀`), up to a possibly different `eltype`. When the linear map is a simple
    `AbstractMatrix`, `lvecs` and `rvecs` will be `Vector{Vector{<:Number}}`.
*   `info`: an object of type [`ConvergenceInfo`], which has the following fields
    -   `info.converged::Int`: indicates how many singular values and vectors were actually
        converged to the specified tolerance `tol` (see below under keyword arguments)
    -   `info.residual::Vector`: a list of the same length as `vals` containing the
        residuals
        `info.residual[i] = A * rvecs[i] - vals[i] * lvecs[i]`.
    -   `info.normres::Vector{<:Real}`: list of the same length as `vals` containing the
        norm of the residual `info.normres[i] = norm(info.residual[i])`
    -   `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A` or `A'`.
    -   `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)
!!! warning "Check for convergence"
    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged >= howmany`.

### Keyword arguments:
Keyword arguments and their default values are given by:
*   `verbosity::Int = 0`: verbosity level, i.e. 0 (no messages), 1 (single message
    at the end), 2 (information after every iteration), 3 (information per Krylov step)
*   `krylovdim`: the maximum dimension of the Krylov subspace that will be constructed.
    Note that the dimension of the vector space is not known or checked, e.g. `x₀` should not
    necessarily support the `Base.length` function. If you know the actual problem dimension
    is smaller than the default value, it is useful to reduce the value of `krylovdim`, though
    in principle this should be detected.
*   `tol`: the requested accuracy according to `normres` as defined above. If you work in
    e.g. single precision (`Float32`), you should definitely change the default value.
*   `maxiter`: the number of times the Krylov subspace can be rebuilt; see below for further
    details on the algorithms.
*   `orth`: the orthogonalization method to be used, see [`Orthogonalizer`](@ref)

### Algorithm
The last method, without default values and keyword arguments, is the one that is finally
called, and can also be used directly. Here the algorithm is specified, though currently
only [`GKL`](@ref) is available. `GKL` refers to the the partial Golub-Kahan-Lanczos
bidiagonalization which forms the basis for computing the approximation to the singular
values. This factorization is dynamically shrunk and expanded (i.e. thick restart) similar
to the Krylov-Schur factorization for eigenvalues.
"""
function svdsolve end

function svdsolve(A::AbstractMatrix, howmany::Int = 1, which::Selector = :LR, T::Type = eltype(A); kwargs...)
    svdsolve(A, rand(T, size(A,1)), howmany, which; kwargs...)
end
function svdsolve(f, n::Int, howmany::Int = 1, which::Selector = :LR, T::Type = Float64; kwargs...)
    svdsolve(f, rand(T, n), howmany, which; kwargs...)
end
function svdsolve(f, x₀, howmany::Int = 1, which::Symbol = :LR; kwargs...)
    which == :LR || which == :SR || error("invalid specification of which singular values to target: which = $which")
    alg = GKL(; kwargs...)
    svdsolve(f, x₀, howmany, which, alg)
end

function svdsolve(A, x₀, howmany::Int, which::Symbol, alg::GKL)
    krylovdim = alg.krylovdim
    maxiter = alg.maxiter
    howmany > krylovdim && error("krylov dimension $(krylovdim) too small to compute $howmany singular values")

    ## FIRST ITERATION: setting up
    numiter = 1
    # Compute Lanczos factorization
    iter = GKLIterator(svdfun(A), x₀, alg.orth)
    fact = initialize(iter; verbosity = alg.verbosity-2)
    numops = 2
    sizehint!(fact, krylovdim)
    β = normres(fact)
    tol::eltype(β) = alg.tol
    while length(fact) < krylovdim
        fact = expand!(iter, fact; verbosity = alg.verbosity-2)
        numops += 2
        normres(fact) < tol && length(fact) >= howmany && break
    end

    # Process
    # allocate storage
    HH = fill(zero(eltype(fact)), krylovdim+1, krylovdim)
    PP = fill(zero(eltype(fact)), krylovdim, krylovdim)
    QQ = fill(zero(eltype(fact)), krylovdim, krylovdim)

    # initialize
    β = normres(fact)
    m = length(fact)
    P = copyto!(view(PP, 1:m, 1:m), I)
    Q = copyto!(view(QQ, 1:m, 1:m), I)
    f = view(HH, m+1, 1:m)
    B = rayleighquotient(fact) # Bidiagional (lower)

    # compute singular value decomposition
    P, S, Q = bidiagsvd!(B, P, Q)
    if which == :SR
        reversecols!(P)
        reverserows!(S)
        reverserows!(Q)
    elseif which != :LR
        error("invalid specification of which singular values to target: which = $which")
    end
    mul!(f, view(Q', m, :), β)

    converged = 0
    while converged < length(fact) && abs(f[converged+1]) < tol
        converged += 1
    end

    if alg.verbosity > 1
        msg = "GKL svdsolve in iter $numiter: "
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
        keep = div(3*krylovdim + 2*converged, 5) # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

        # Update basis by applying P and Q using Householder reflections
        U = basis(fact, :U)
        basistransform!(U, view(P, :, 1:keep))
        # for j = 1:m
        #     h, ν = householder(P, j:m, j)
        #     lmul!(h, view(P, :, j+1:krylovdim))
        #     rmul!(U, h')
        # end
        V = basis(fact, :V)
        basistransform!(V, view(Q', :, 1:keep))
        # for j = 1:m
        #     h, ν = householder(Q, j, j:m)
        #     rmul!(view(Q, j+1:krylovdim, :), h)
        #     rmul!(V, h)
        # end

        # Shrink GKL factorization (no longer strictly GKL)
        r = residual(fact)
        U[keep+1] = rmul!(r, 1/normres(fact))
        H = fill!(view(HH, 1:keep+1, 1:keep), 0)
        @inbounds for j = 1:keep
            H[j,j] = S[j]
            H[keep+1,j] = f[j]
        end

        # Restore bidiagonal form in the first keep columns
        @inbounds for j = keep:-1:1
            h, ν = householder(H, j+1, 1:j, j)
            H[j+1,j] = ν
            H[j+1,1:j-1] .= 0
            rmul!(view(H, 1:j, :), h')
            rmul!(V, h')
            h, ν = householder(H, 1:j, j, j)
            H[j,j] = ν
            @inbounds H[1:j-1,j] .= 0
            lmul!(h, view(H, :, 1:j-1))
            rmul!(U, h')
        end
        @inbounds for j = 1:keep
            fact.αs[j] = H[j,j]
            fact.βs[j] = H[j+1,j]
        end
        fact = shrink!(fact, keep)

        # GKL factorization: recylce fact
        while length(fact) < krylovdim
            fact = expand!(iter, fact; verbosity = alg.verbosity-2)
            numops += 2
            normres(fact) < tol && length(fact) >= howmany && break
        end

        # post process
        β = normres(fact)
        m = length(fact)
        P = copyto!(view(PP, 1:m, 1:m), I)
        Q = copyto!(view(QQ, 1:m, 1:m), I)
        f = view(HH, m+1, 1:m)
        B = rayleighquotient(fact) # Bidiagional (lower)

        # compute singular value decomposition
        P, S, Q = bidiagsvd!(B, P, Q)
        if which == :SR
            reversecols!(P)
            reverserows!(S)
            reverserows!(Q)
        elseif which != :LR
            error("invalid specification of which singular values to target: which = $which")
        end
        mul!(f, view(Q', m, :), β)
        converged = 0
        while converged < length(fact) && abs(f[converged+1]) < tol
            converged += 1
        end

        if alg.verbosity > 1
            msg = "GKL svdsolve in iter $numiter: "
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

    if converged > howmany
        howmany = converged
    end
    values = S[1:howmany]

    # Compute schur vectors
    Pv = view(P,:, 1:howmany)
    Qv = view(Q, 1:howmany, :)

    # Compute convergence information
    leftvectors = let U = basis(fact, :U)
        [U*v for v in cols(Pv)]
    end
    rightvectors = let V = basis(fact, :V)
        [V*v for v in cols(Qv')]
    end
    residuals = let r = residual(fact)
        [r*last(v) for v in cols(Qv')]
    end
    normresiduals = let f = f
        map(i->abs(f[i]), 1:howmany)
    end
    if alg.verbosity > 0
        if converged < howmany
            @warn """GKL svdsolve finished without convergence after $numiter iterations:
             *  $converged singular values converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        else
            @info """GKL svdsolve finished after $numiter iterations:
             *  $converged singular values converged
             *  norm of residuals = $((normresiduals...,))
             *  number of operations = $numops"""
        end
    end

    return values, leftvectors, rightvectors,
            ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)
end

svdfun(A::AbstractMatrix) = (x,flag) -> flag ? A'*x : A*x
svdfun((f,fadjoint)::Tuple{Any,Any}) = (x,flag) -> flag ? fadjoint(x) : f(x)
svdfun(f) = f
