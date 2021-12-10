using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20000
using CUDA
using CpuId
using DataFrames

cpu = (name = cpubrand(), cores = cpucores(), threads = cputhreads())

println("---- System info ----")
println("CPU $(cpu.name) $(cpu.cores)C$(cpu.threads)T")
println("GPU $(name(device()))")

Nx, Ny = 1024, 1024

## FP64
t1 = @belapsed u1 .*= u2 setup = (u1 = rand(ComplexF64, Nx, Ny); u2 = rand(ComplexF64, Nx, Ny));
t2 = @belapsed (CUDA.@sync u1 .*= u2) setup = (u1 = CUDA.rand(ComplexF64, Nx, Ny); u2 = CUDA.rand(ComplexF64, Nx, Ny));

df_F64 = DataFrame(case = ["CPU", "GPU"],
    time_ms = [t1, t2] * 1000,
    performance = t1 ./ [t1, t2])
println(df_F64)

## FP32
t1 = @belapsed u1 .*= u2 setup = (u1 = rand(ComplexF32, Nx, Ny); u2 = rand(ComplexF32, Nx, Ny));
t2 = @belapsed (CUDA.@sync u1 .*= u2) setup = (u1 = CUDA.rand(ComplexF32, Nx, Ny); u2 = CUDA.rand(ComplexF32, Nx, Ny));

df_F32 = DataFrame(case = ["CPU", "GPU"],
    time_ms = [t1, t2] * 1000,
    performance = t1 ./ [t1, t2])
println(df_F32)
# show(df_F32, allrows = true)