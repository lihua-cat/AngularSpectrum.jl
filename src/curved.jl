function phase_shift(xs, ys, radius, λ)
    Nx, Ny = length(xs), length(ys)
    yts = transpose(ys)
    δ = isinf(radius) ? 
        ones(Nx, Ny) .* zero(radius) : 
        @. radius - √(radius^2 - xs^2 - yts^2)
    δps = @. cispi(-2 / λ * 2 * δ)
end