# Linear problems

Linear systems are of the form `A*x=b` where `A` should be a linear map that has the same
type of output as input, i.e. the solution `x` should be of the same type as the right hand
side `b`. They can be solved using the function `linsolve`:

```@docs
linsolve
```

Currently supported algorithms are [`CG`](@ref) and [`GMRES`](@ref).
