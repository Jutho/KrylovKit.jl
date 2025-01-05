# Least squares problems

Least square problems take the form of finding `x` that minimises `norm(b - A*x)` where
`A` should be a linear map. As opposed to linear systems, the input and output of the linear
map do not need to be the same, so that `x` (input) and `b` (output) can live in different
vector spaces. Such problems can be solved using the function `lssolve`:

```@docs
lssolve
```
