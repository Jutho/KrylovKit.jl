using Documenter
using KrylovKit

makedocs(
    modules=[KrylovKit],
    sitename = "KrylovKit.jl",
    authors = "Jutho Haegeman",
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "man/intro.md",
            "man/linear.md",
            "man/eig.md",
            "man/svd.md",
            "man/matfun.md",
            "man/algorithms.md",
            "man/implementation.md"
        ]
    ],
)

deploydocs(repo = "github.com/Jutho/KrylovKit.jl.git")
