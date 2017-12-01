if VERSION < v"0.7.0-DEV.2005"
    const Test = Base.Test
else
    import Test
end

using KrylovKit
using Test
const n = 10
const N = 100

include("linalg.jl")
include("factorize.jl")
include("eigsolve.jl")
include("gmres.jl")
include("exponentiate.jl")
