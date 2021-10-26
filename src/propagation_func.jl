function propagation_func(νx::AbstractVector, νy::AbstractVector, λ::Real, d::Real, evanescent::Bool = true)
    isuniform(νx) || error("νx is not shift to center") 
    isuniform(νy) || error("νy is not shift to center") 
    α, β = λ * νx, λ * νy
    βt = transpose(β)
    circ = @. α^2 + βt^2
    circ_mask = circ .< 1
    trans = Matrix{Complex{eltype(float.(νx))}}(undef, length(νx), length(νy))
    @. trans[circ_mask] = cispi(2 * d / λ * √(1 - circ[circ_mask]))
    @. trans[!circ_mask] = evanescent ? exp(-2π * d / λ * √(circ[!circ_mask] - 1)) : 0
    return ifftshift(trans)
end