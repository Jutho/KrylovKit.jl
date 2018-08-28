using Documenter, KrylovKit

makedocs(modules=[KrylovKit],
            format=:html,
            sitename="KrylovKit.jl",
            pages = [
                "Home" => "index.md",
                "Linear Problems" => "linsolve.md",
                "Eigenvalue Problems" => "eigsolve.md",
                "Matrix exponential" => "exponentiate.md",
            ])
