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

##
X, Y = 8e-2, 2e-2
ap = (x = 6e-2, y = 1.6e-2)
λ = 1.315e-5
d = 1.5
radius = 10.0

Nx, Ny = 8192, 2048

x = collect(0:Nx-1) * X / Nx
y = collect(0:Ny-1) * Y / Ny
xs, ys = x .- X/2, y .- Y/2
yts = transpose(ys)
ap_mask = @. (abs(xs) < ap.x/2) * (abs(yts) < ap.y/2)
δps = phase_shift(xs, ys, radius, λ)

u0 = zeros(ComplexF64, Nx, Ny)
u0[ap_mask] .= 1
# u0 .*= cispi.((rand(Nx, Ny) .- 1/2) * 2)
# u0 = (u0 + reverse(u0, dims=2)) / 2

νx, νy = spatial_frequency(X, Y, Nx, Ny)
trans = propagation_func(νx, νy, λ, d)
plan, iplan = plan_fft!(u0), plan_ifft!(u0)

ap_poly = ap_makie(ap, X, Y)
##
u = copy(u0)
PRECISION = Float32
u_d = CuArray{Complex{PRECISION}}(u0)
plan_d, iplan_d = plan_fft!(u_d), plan_ifft!(u_d)
trans_d = CuArray{Complex{PRECISION}}(trans)
ap_mask_d = CuArray{Complex{PRECISION}}(ap_mask)
δps_d = CuArray{Complex{PRECISION}}(δps)

i_node = Node(1)
loss_node = Node(100.0)
u2_node = Node(abs2.(u))
u2d_node = Node(abs2.(Array(u_d)))
fig = Figure(resolution = (1200, 800), fontsize = 24)
ax1 = Axis(fig[1, 1], aspect = AxisAspect(X/Y), title = "cpu")
ax2 = Axis(fig[2, 1], aspect = AxisAspect(X/Y), title = "gpu")
poly!(ax1, ap_poly, color = :transparent, strokecolor = :cyan, strokewidth = 1)
poly!(ax2, ap_poly, color = :transparent, strokecolor = :cyan, strokewidth = 1)
heatmap!(ax1, x, y, u2_node, colormap = :plasma)
heatmap!(ax2, x, y, u2d_node, colormap = :plasma)
Label(fig[0, :], text = @lift("trip = $($i_node), loss = $($loss_node)%"), tellwidth = false)
display(fig)


e_cpu = 0.0
e_gpu = 0.0

tic = time_ns()

@progress for i in 1:100
    p1 = sum(abs2.(u))
    if iseven(i)
        u .*= δps
        u_d .*= δps_d
    end
    elapsed_cpu = @elapsed free_propagate!(u, trans, plan, iplan) 
    elapsed_gpu = CUDA.@elapsed free_propagate!(u_d, trans_d, plan_d, iplan_d) 
    global e_cpu += elapsed_cpu
    global e_gpu += elapsed_gpu
    u .*= ap_mask
    u_d .*= ap_mask_d
    p2 = sum(abs2.(u))

    if i <= 100 || (i <= 1000 && i % 100 == 0) || (i <= 10000 && i % 1000 == 0)
        i_node[] = i
        u2_node[] = abs2.(u)
        u2d_node[] = abs2.(Array(u_d))
        loss_node[] = round((p1 - p2) / p1 * 100, sigdigits = 4)
        sleep(0.01)
    end

end

toc = time_ns()

@printf("cpu elapsed: %10.3f s\n", e_cpu)
@printf("gpu elapsed: %10.3f s\n", e_gpu)
@printf("tot elapsed: %10.3f s\n", (toc - tic) / 1e9)
