"""
    svdsolve(A::AbstractMatrix, [howmany = 1, which = :LR, T = eltype(A)]; kwargs...)
    svdsolve(f, m::Int, n::Int, [howmany = 1, which = :LR, T = Float64]; kwargs...)
    svdsolve(f, x₀, y₀, [howmany = 1, which = :LM]; kwargs...)
    svdsolve(f, x₀, y₀, howmany, which, algorithm)

Compute `howmany` singular values from the linear map encoded in the matrix `A` or by the
function `f`. Return singular values, left and right singular vectors and a
`ConvergenceInfo` structure.

### Arguments:
The linear map can be an `AbstractMatrix` (dense or sparse) or a general function or callable
object. Since both the action of the linear map and its adjoint are required in order to compute
singular values, `f` can either be a tuple of two callable objects (each accepting a single argument),
representing the linear map and its adjoint respectively, or, `f` can be a single callable object
that accepts two input arguments, and returns a length two tuple containing the action of the
linear map on the first input argument, and the action of the adjoint of the linear map on the
second argument. The latter form still combines well with the `do` block syntax of Julia, as in
```julia
vals, lvecs, rvecs, info = svdsolve(x₀, y₀, howmany, which; kwargs...) do (x,y)
    # x′ = compute action of linear map on x
    # y′ = compute action of adjoint map on y
    return (x′, y′)
end
```

For a general linear map encoded using either the tuple or the two-argument form, the best approach
is to provide starting vectors `x₀` (in the domain of the linear map) and `y₀` (in the domain of
the adjoint of the linear map). Alternatively, one can specify the size of the linear map using
integers `m` and `n`, in which case `x₀ = rand(T, n)` and `y₀ = rand(T, m)` are used, where the
default value of `T` is `Float64`, unless specified differently. If an `AbstractMatrix` is used,
starting vectors `x₀` and `y₀` do not need to be provided; they are chosen as `rand(T, size(A,2))`
and `rand(T, size(A,1))`.

The next arguments are optional, but should typically be specified. `howmany` specifies how many
singular values and vectors should be computed; `which` specifies which singular values should
be targetted. Valid specifications of `which` are
*   `LM` or `LR`: largest singular values
*   `SR`: smallest singular values
However, the current implementation based on the circulant matrix (see below) is only suitable
for targetting largest singular values.

### Return values:
The return value is always of the form `vals, lvecs, rvecs, info = svdsolve(...)` with
*   `vals`: a `Vector{<:Real}` containing the singular values, of length at least `howmany`,
    but could be longer if more singular values were converged at the same cost.
*   `lvecs`: a `Vector` of corresponding left singular vectors, of the same length as `vals`.
*   `rvecs`: a `Vector` of corresponding right singular vectors, of the same length as `vals`.
    Note that singular vectors are not returned as a matrix, as the linear map could act on any
    custom Julia type with vector like behavior, i.e. the elements of the lists `lvecs` (`rvecs`)
    are objects that are typically similar to the starting guess `y₀` (`x₀`), up to a possibly
    different `eltype`. When the linear map is a simple `AbstractMatrix`, `lvecs` and `rvecs`
    will be `Vector{Vector{<:Number}}`.
*   `info`: an object of type [`ConvergenceInfo`], which has the following fields
    -   `info.converged::Int`: indicates how many singular values and vectors were actually
        converged to the specified tolerance `tol` (see below under keyword arguments)
    -   `info.residual::Vector`: a list of the same length as `vals` containing the residuals
        as a [`RecursiveVec`](@ref) with two components, such that

        -   `first(info.residual[i]) = (A' * lvecs[i] - vals[i] * rvecs[i]) / √2`
        -   `last( info.residual[i]) = (A  * rvecs[i] - vals[i] * lvecs[i]) / √2`
    -   `info.normres::Vector{<:Real}`: list of the same length as `vals` containing the norm
        of the residual `info.normres[i] = norm(info.residual[i])`
    -   `info.numops::Int`: number of times the linear map was applied, i.e. number of times
        `f` was called, or a vector was multiplied with `A`
    -   `info.numiter::Int`: number of times the Krylov subspace was restarted (see below)
!!! warning "Check for convergence"
    No warning is printed if not all requested eigenvalues were converged, so always check
    if `info.converged >= howmany`.

### Keyword arguments:
Keyword arguments and their default values are given by:
*   `krylovdim`: the maximum dimension of the Krylov subspace that will be constructed.
    Note that the dimension of the vector space is not known or checked, e.g. `x₀` should not
    necessarily support the `Base.length` function. If you know the actual problem dimension
    is smaller than the default value, it is useful to reduce the value of `krylovdim`, though
    in principle this should be detected.
*   `tol`: the requested accuracy according to `normres` as defined above. If you work in e.g.
    single precision (`Float32`), you should definitely change the default value.
*   `maxiter`: the number of times the Krylov subspace can be rebuilt; see below for further
    details on the algorithms.

### Algorithm
Currently, singular values are computed as eigenvalues of the hermitian matrix ``[0 A'; A 0]``
using the [`eigsolve`](@ref) routine with the [`Lanczos`](@ref) algorithm, without forming this
matrix explicitly. For this, [`RecursiveVec`](@ref) is employed to capture both blocks of the
eigenvalue problem. The left and right singular vectors are afterwards simply extracted from
the first and last component of the corresponding eigenvector (up to normalization). The last method,
without default values and keyword arguments, is the one that is finally called, and can also
be used directly. Here, one specifies the algorithm explicitly as last argument, but only [`Lanczos`](@ref)
is currently accepted.
"""
function svdsolve end

function svdsolve(A::AbstractMatrix, howmany::Int = 1, which::Selector = :LR, T::Type = eltype(A); kwargs...)
    svdsolve(A, rand(T, size(A,2)), rand(T, size(A,1)), howmany, which; kwargs...)
end
function svdsolve(f, m::Int, n::Int, howmany::Int = 1, which::Selector = :LR, T::Type = Float64; kwargs...)
    svdsolve(f, rand(T, n), rand(T, m), howmany, which; kwargs...)
end
function svdsolve(f, x₀, y₀, howmany::Int = 1, which::Symbol = :LR; kwargs...)
    which == :LM || which == :LR || error("only largest singular values are currently supported; use which = :LR")
    alg = Lanczos(;kwargs...)
    svdsolve(f, x₀, y₀, howmany, alg)
end

function svdsolve(f, x₀, y₀, howmany::Int, alg::Lanczos)
    g = x -> RecursiveVec(reverse(svdfun(f)(x[1], x[2])))
    vals, vecs, info = eigsolve(g, RecursiveVec((x₀, y₀)), howmany, :LR, alg)
    # check if all returned eigenvalues correspond to actual singular values
    i = howmany
    @inbounds while i <= length(vals)
        vals[i] < 0 && break # negative values are necessarily discarded
        if vals[i] < 10*max(info.normres[i], eps(eltype(vals))) # carefully check small positive eigenvalues
            # if this eigenvalue is an actual singular value, both the first and second component
            # of the corresponding vec should have norm approximately 1/√2;
            # otherwise it is a zero eigenvalue due to a rectangular shape of the linear map
            if !(norm(first(vecs[i])) ≈ norm(last(vecs[i])))
                break
            end
        end
        i += 1
    end
    i -= 1
    # prepare output
    if i < length(vals)
        vals = resize!(vals, i)
        lvecs = rmul!.(last.(view(vecs,1:i)), sqrt(2))
        rvecs = rmul!.(first.(view(vecs,1:i)), sqrt(2))

        info = ConvergenceInfo(min(info.converged, i), resize!(info.residual, i),
                                resize!(info.normres, i), info.numops, info.numiter)
    else
        lvecs = rmul!.(last.(vecs), sqrt(2))
        rvecs = rmul!.(first.(vecs), sqrt(2))
    end
    return vals, lvecs, rvecs, info
end

svdfun(A::AbstractMatrix) = (x,y) -> (A*x, A'*y)
svdfun(f::Tuple{Any,Any}) = (x,y) -> (f[1](x), f[2](x))
svdfun(f) = f
