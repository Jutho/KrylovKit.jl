## Introduction
The high level interface of KrylovKit is provided by the following functions:
*   [`linsolve`](@ref): solve linear systems
*   [`eigsolve`](@ref): find a few eigenvalues and corresponding eigenvectors
*   [`svdsolve`](@ref): find a few singular values and corresponding left and right singular vectors
*   [`exponentiate`](@ref): apply the exponential of a linear map to a vector

They all follow the standard format
```julia
results..., info = problemsolver(A, args...; kwargs...)
```
where `problemsolver` is one of the functions above. Here, `A` is the linear map in the problem,
which could be an instance of `AbstractMatrix`, or any function or callable object that encodes
the action of the linear map on a vector. In particular, one can write the linear map using
Julia's `do` block syntax as
```julia
results..., info = method(args...; kwargs...) do x
    # implement linear map on x
end
```
Furthermore, `args` is a set of additional arguments to specify the problem. The keyword arguments
`kwargs` contain information about the linear map (`issymmetric`, `ishermitian`, `isposdef`) and
about the solution strategy (`tol`, `krylovdim`, `maxiter`). A suitable algorithm for the problem
is then chosen. The return value contains one or more entries that define the solution, and a final
entry `info` of type [`ConvergeInfo`](@ref) that encodes information about the solution, i.e. wether it
has converged, the residual(s) and the norm thereof, the number of operations used.

There is also an expert interface where the user specifies the algorithm that should be used
explicitly, i.e.
```julia
results..., info = method(A, args..., algorithm)
```
