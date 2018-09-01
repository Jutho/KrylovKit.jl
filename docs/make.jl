using Documenter
using KrylovKit

makedocs(modules=[KrylovKit],
            format=:html,
            sitename="KrylovKit.jl",
            pages = [
                "Home" => "index.md",
                "Manual" => [
                    "man/linear.md",
                    "man/eig.md",
                    "man/svd.md",
                    "man/matfun.md",
                    "man/algorithms.md",
                    "man/implementation.md"
                ]
                # "Linear systems" => "linear.md",
                # "Eigenvalues and singular values" => "eigsvd.md",
                # "Matrix functions" => "matfun.md",
                # "Available algorithms" => "algorithms.md",
                # "Implementation" => "implementation.md"
            ])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    deps = nothing,
    make = nothing,
    target = "build",
    repo = "github.com/Jutho/KrylovKit.jl.git",
    julia = "1.0"
)
