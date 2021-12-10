function propagation_func(νx::AbstractVector, νy::AbstractVector, λ, d, evanescent::Bool = true)
    isshift(νx) || error("νx is not shift to center")
    isshift(νy) || error("νy is not shift to center")
    α, β = λ * νx, λ * νy
    βt = transpose(β)
    circ = @. α^2 + βt^2
    circ_mask = circ .< 1
    trans = Matrix{ComplexF64}(undef, length(νx), length(νy))
    @. trans[circ_mask] = cispi(2 * d / λ * √(1 - circ[circ_mask]) + 0.0)
    @. trans[!circ_mask] = evanescent ? exp(-2π * d / λ * √(circ[!circ_mask] - 1)) : 0
    return ifftshift(trans)
end

function propagation_func(X, Y, Nx, Ny, λ, d, evanescent::Bool = true)
    νx, νy = spatial_frequency(X, Y, Nx, Ny)
    return propagation_func(νx, νy, λ, d, evanescent)
end