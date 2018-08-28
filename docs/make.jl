using Documenter, KrylovKit

makedocs(modules=[KrylovKit],
            format=:html,
            sitename="KrylovKit.jl",
            pages = [
                "Home" => "index.md",
                "Linear systems and least square problems" => "linear.md",
                "Eigenvalues and singular values" => "eigsvd.md",
                "Matrix functions" => "matfun.md",
                "Available algorithms" => "algorithms.md",
                "Implementation" => "implementation.md"
            ])
