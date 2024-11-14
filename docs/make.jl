using Documenter
using KrylovKit

makedocs(; modules=[KrylovKit],
         sitename="KrylovKit.jl",
         authors="Jutho Haegeman and collaborators",
         pages=["Home" => "index.md",
                "Manual" => ["man/intro.md",
                             "man/linear.md",
                             "man/eig.md",
                             "man/svd.md",
                             "man/matfun.md",
                             "man/reallinear.md",
                             "man/algorithms.md",
                             "man/implementation.md"]],
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"))

deploydocs(; repo="github.com/Jutho/KrylovKit.jl.git")
