using AngularSpectrum
using CUDA
using FFTW
using GLMakie
using Printf
using ProgressLogging

function ap_makie(ap, X, Y)
    apx1, apy1 = @. ((X, Y) - (ap.x, ap.y)) ./ 2
    apx2, apy2 = @. (X, Y) .- (apx1, apy1)
    ap_makie = Point2f0[(apx1, apy1), (apx1, apy2), (apx2, apy2), (apx2, apy1)]
end

function paras(Nx, Ny, random)
    x = collect(0:Nx-1) * X / Nx
    y = collect(0:Ny-1) * Y / Ny
    xs, ys = x .- X/2, y .- Y/2
    yts = transpose(ys)
    ap_mask = @. (abs(xs) < ap.x/2) * (abs(yts) < ap.y/2)
    δps = phase_shift(xs, ys, radius, λ)
    νx, νy = spatial_frequency(X, Y, Nx, Ny)
    trans = propagation_func(νx, νy, λ, d)
    u = zeros(ComplexF64, Nx, Ny)
    u[ap_mask] .= 1
    if random
        u .*= cispi.((rand(Nx, Ny) .- 1/2) * 2)
        u = (u + reverse(u, dims=2)) / 2
    end
    return x, y, ap_mask, δps, trans, u
end

##
X, Y = 8e-2, 2e-2
ap = (x = 6e-2, y = 1.8e-2)
λ = 1.315e-6
d = 1.5
radius = 10.0

ap_poly = ap_makie(ap, X, Y)

Nx1, Ny1 = 1024, 256
Nx2, Ny2 = (Nx1, Ny1) .* 4

random = true
x1, y1, ap1_mask, δps1, trans1, u1 = paras(Nx1, Ny1, random)
x2, y2, ap2_mask, δps2, trans2, u2 = paras(Nx2, Ny2, random)

##
PRECISION = Float32
u1_d = CuArray{Complex{PRECISION}}(u1)
plan1_d, iplan1_d = plan_fft!(u1_d), plan_ifft!(u1_d)
trans1_d = CuArray{Complex{PRECISION}}(trans1)
ap1_mask_d = CuArray{Complex{PRECISION}}(ap1_mask)
δps1_d = CuArray{Complex{PRECISION}}(δps1)

u2_d = CuArray{Complex{PRECISION}}(u2)
plan2_d, iplan2_d = plan_fft!(u2_d), plan_ifft!(u2_d)
trans2_d = CuArray{Complex{PRECISION}}(trans2)
ap2_mask_d = CuArray{Complex{PRECISION}}(ap2_mask)
δps2_d = CuArray{Complex{PRECISION}}(δps2)

i_node = Node(1)
loss_node = Node(100f0)
u1d_node = Node(abs2.(Array(u1_d)))
u2d_node = Node(abs2.(Array(u2_d)))
fig = Figure(resolution = (1200, 800), fontsize = 24)
ax1 = Axis(fig[1, 1], aspect = AxisAspect(X/Y), title = "gpu1")
ax2 = Axis(fig[2, 1], aspect = AxisAspect(X/Y), title = "gpu2")
poly!(ax1, ap_poly, color = :transparent, strokecolor = :cyan, strokewidth = 1)
poly!(ax2, ap_poly, color = :transparent, strokecolor = :cyan, strokewidth = 1)
heatmap!(ax1, x1, y1, u1d_node, colormap = :plasma)
heatmap!(ax2, x2, y2, u2d_node, colormap = :plasma)
Label(fig[0, :], text = @lift("trip = $($i_node), loss = $($loss_node)%"), tellwidth = false)
display(fig)


e_gpu1 = 0.0
e_gpu2 = 0.0

tic = time_ns()

@progress for i in 1:10000
    p1 = sum(abs2.(Array(u1_d)))
    if iseven(i)
        u1_d .*= δps1_d
        u2_d .*= δps2_d
    end
    elapsed_gpu1 = CUDA.@elapsed free_propagate!(u1_d, trans1_d, plan1_d, iplan1_d) 
    elapsed_gpu2 = CUDA.@elapsed free_propagate!(u2_d, trans2_d, plan2_d, iplan2_d) 
    global e_gpu1 += elapsed_gpu1
    global e_gpu2 += elapsed_gpu2
    u1_d .*= ap1_mask_d
    u2_d .*= ap2_mask_d
    p2 = sum(abs2.(Array(u1_d)))

    if (i <= 100 && i % 2 == 0) || (i <= 1000 && i % 100 == 0) || (i <= 10000 && i % 1000 == 0)
        i_node[] = i
        u1d_node[] = abs2.(Array(u1_d))
        u2d_node[] = abs2.(Array(u2_d))
        loss_node[] = round((p1 - p2) / p1 * 100, sigdigits = 4)
        sleep(0.01)
    end

end

toc = time_ns()

@printf("gpu1 elapsed: %10.3f s\n", e_gpu1)
@printf("gpu2 elapsed: %10.3f s\n", e_gpu2)
@printf("tot elapsed: %10.3f s\n", (toc - tic) / 1e9)
