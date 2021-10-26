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

function propagation_func(X::Real, Y::Real, Nx::Signed, Ny::Signed, λ::Real, d::Real, evanescent::Bool = true)
    νx, νy = spatial_frequency(X, Y, Nx, Ny)
    return propagation_func(νx, νy, λ, d, evanescent)
end

# function propagation_func(x::AbstractVector, y::AbstractVector, λ::Real, d::Real, evanescent::Bool = true)
#     νx, νy = spatial_frequency(x, y)
#     return propagation_func(νx, νy, λ, d, evanescent)
# end

function propagation_func(X::Unitful.Length, Y::Unitful.Length, Nx::Signed, Ny::Signed, λ::Unitful.Length, d::Unitful.Length, evanescent::Bool = true)
    uu = unit(X)
    X_val, Y_val, λ_val, d_val = [ustrip(uu, i) for i in (X, Y, λ, d)]
    return propagation_func(X_val, Y_val, Nx, Ny, λ_val, d_val, evanescent)
end