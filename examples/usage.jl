## Usage
using AngularSpectrum
using FFTW
FFTW.set_num_threads(8)
import LinearAlgebra: norm
# using MKL
using CUDA
using Unitful
using GLMakie
## 1. given a initial uniform 2D optical field distribution
X, Y = 8.0u"cm", 8.0u"cm"
Nx, Ny = 1024, 1024
x = collect(0:Nx-1) * X / Nx
y = collect(0:Ny-1) * Y / Ny
ap = (x = 6.0u"cm", y = 6.0u"cm")
ap_mask = (abs.(x .- X/2) .< ap.x/2) .* transpose(abs.(y .- Y/2) .< ap.y/2)
ap_mask = ap_mask .* reverse(ap_mask)
u0 = ones(ComplexF64, Nx, Ny) .* ap_mask
## 2. compute propagation function of angular spectrum
λ = 1.315u"μm"
d = 1.5u"m"
trans = propagation_func(X, Y, Nx, Ny, λ, d)
## 3. angular spectrum approach
function Fox_Li_method(u0::AbstractMatrix, trans::AbstractMatrix, ap::AbstractMatrix{Bool}, N::Signed, precision::Type{<:AbstractFloat}; gpu::Bool)
    if gpu
        u = CuArray{Complex{precision}}(u0)
        trans = CuArray{Complex{precision}}(trans)
        plan, iplan = plan_fft!(u), plan_ifft!(u)
        ap = CuArray(ap)
    else
        u = copy(convert(Matrix{Complex{precision}}, u0))
        trans = convert(Matrix{Complex{precision}}, trans)
        plan, iplan = plan_fft!(u, flags=FFTW.MEASURE), plan_ifft!(u, flags=FFTW.MEASURE)
    end
    t = @elapsed for _ in 1:N
        free_propagate!(u, trans, plan, iplan)
        u .*= ap
    end
    return Array(u), t
end

N = 1000

u_h_F32, t_h_F32 = Fox_Li_method(u0, trans, ap_mask, N, Float32, gpu = false)
u_h_F64, t_h_F64 = Fox_Li_method(u0, trans, ap_mask, N, Float64, gpu = false)
u_d_F32, t_d_F32 = Fox_Li_method(u0, trans, ap_mask, N, Float32, gpu = true)
u_d_F64, t_d_F64 = Fox_Li_method(u0, trans, ap_mask, N, Float64, gpu = true)

error_F32 = norm(u_h_F32 - u_d_F32) ./ norm(u_h_F32)
error_F64 = norm(u_h_F64 - u_d_F64) ./ norm(u_h_F64)

println("\n- FP32 ")
print("Consistency between CPU and GPU : ")
u_h_F32 ≈ u_d_F32 ? println("Pass") : println("Fail")
print("Symmetry about y-axis : ")
@views u_h_F32[1:Nx÷2, :] ≈ u_h_F32[end:-1:Nx÷2+1, :] ? println("Pass") : println("Fail")
print("Symmetry about x-axis : ")
@views u_h_F32[:, 1:Ny÷2] ≈ u_h_F32[:, end:-1:Ny÷2+1] ? println("Pass") : println("Fail")
println("Relative error : $error_F32")
println("\n- FP64 ")
print("Consistency between CPU and GPU : ")
u_h_F64 ≈ u_d_F64 ? println("Pass") : println("Fail")
print("Symmetry about y-axis : ")
@views u_h_F64[1:Nx÷2, :] ≈ u_h_F64[end:-1:Nx÷2+1, :] ? println("Pass") : println("Fail")
print("Symmetry about x-axis : ")
@views u_h_F64[:, 1:Ny÷2] ≈ u_h_F64[:, end:-1:Ny÷2+1] ? println("Pass") : println("Fail")
println("Relative error : $error_F64")
## 4. visualization
let
    uu = u"mm"
    Xv, Yv = ustrip(uu, X), ustrip(uu, Y)
    ap_v = (; x = ustrip(uu, ap.x), y = ustrip(uu, ap.y))
    apx1, apy1 = @. ((Xv, Yv) - (ap_v.x, ap_v.y)) / 2
    apx2, apy2 = @. (Xv, Yv) - (apx1, apy1)
    ap_poly = Point2f0[(apx1, apy1), (apx1, apy2), (apx2, apy2), (apx2, apy1)]
    x = collect(0:Nx-1) * Xv / Nx
    y = collect(0:Ny-1) * Yv / Ny
    intensity = Array{AbstractFloat, 3}(undef, Nx, Ny, 5)
    intensity[:, :, 1] = abs2.(u0)
    intensity[:, :, 2] = abs2.(u_h_F32)
    intensity[:, :, 3] = abs2.(u_d_F32)
    intensity[:, :, 4] = abs2.(u_h_F64)
    intensity[:, :, 5] = abs2.(u_d_F64)
    max_value = maximum(intensity)

    fig = Figure(resolution = (1100, 750), backgroundcolor = :Wheat, fontsize = 20)
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 1, Relative(0.27))
    
    title = ["initial field", 
             "CPU, elapsed: $(round(t_h_F32, digits=2)) s",
             "GPU, elapsed: $(round(t_d_F32, digits=2)) s",
             "CPU, elapsed: $(round(t_h_F64, digits=2)) s", 
             "GPU, elapsed: $(round(t_d_F64, digits=2)) s"]

    ax1 = Axis(gl1[1, 1], aspect = AxisAspect(Xv/Yv), title = title[1])
    ax2 = [Axis(gl2[i, j], aspect = AxisAspect(Xv/Yv), title = title[2(i-1)+j+1]) for i in 1:2, j in 1:2]
    ax = [ax1, ax2...]

    Label(fig[0, :], "Nx = $Nx, Ny = $Ny, $N loops", textsize = 30)
    Label(gl2[1, 3], "FP32", textsize = 24, rotation = -pi/2, tellheight = false)
    Label(gl2[2, 3], "FP64", textsize = 24, rotation = -pi/2, tellheight = false)
    
    for i in 1:5
        poly!(ax[i], ap_poly, color = :transparent, strokecolor = :grey, strokewidth = 1)
        heatmap!(ax[i], x, y, intensity[:, :, i], colormap = :plasma, colorrange=(0, max_value))
    end
    Colorbar(gl2[:, 4], limits = (0, max_value), height = Relative(1))

    # save("examples/usage.png", fig)
    fig
end
