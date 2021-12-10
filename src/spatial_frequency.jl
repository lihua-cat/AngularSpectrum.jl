function spatial_frequency(X, Y, Nx, Ny, shift::Bool = true)
    dx = 1 / X
    dy = 1 / Y
    νx = collect(range(zero(dx), dx * (Nx - 1), length = Nx))
    νy = collect(range(zero(dy), dy * (Ny - 1), length = Ny))
    if shift
        νx = νx .- dx * (Nx ÷ 2)
        νy = νy .- dy * (Ny ÷ 2)
    end
    return (; x = νx, y = νy)
end

function spatial_frequency(x::AbstractVector, y::AbstractVector, shift::Bool = true)
    isuniform(x) || error("x is not uniform")
    isuniform(y) || error("y is not uniform")
    dx, dy = x[2] - x[1], y[2] - y[1]
    Nx, Ny = length(x), length(y)
    X, Y = (Nx, Ny) .* (dx, dy)
    spatial_frequency(X, Y, Nx, Ny, shift)
end