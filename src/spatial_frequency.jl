function spatial_frequency(X::Real, Y::Real, Nx::Signed, Ny::Signed, shift::Bool = true)
    dx = 1 / X
    dy = 1 / Y
    νx = collect(range(zero(dx), dx * (Nx-1), length = Nx))
    νy = collect(range(zero(dy), dy * (Ny-1), length = Ny))
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


## test
# X, Y = 8.0, 4.0
# Nx, Ny = 8, 7
# x = collect(0:Nx-1) * X / Nx
# y = collect(0:Ny-1) * Y / Ny
# ν = spatial_frequency(X, Y, Nx, Ny)
# ν2 = spatial_frequency(x, y)
# νs = spatial_frequency(X, Y, Nx, Ny, true)
# νs2 = spatial_frequency(x, y, true)