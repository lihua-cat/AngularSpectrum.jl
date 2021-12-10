using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20000
using CUDA
using FFTW
using CpuId
using DataFrames

cpu = (name = cpubrand(), cores = cpucores(), threads = cputhreads())

println("---- System info ----")
println("CPU $(cpu.name) $(cpu.cores)C$(cpu.threads)T")
println("GPU $(name(device()))")

Nx, Ny = 1024, 1024

## FP64
u_F64 = rand(ComplexF64, Nx, Ny)

FFTW.set_num_threads(1)
p1 = plan_fft!(u_F64, flags=FFTW.MEASURE)
FFTW.set_num_threads(cpu.cores)
p2 = plan_fft!(u_F64, flags=FFTW.MEASURE)
FFTW.set_num_threads(cpu.threads)
p3 = plan_fft!(u_F64, flags=FFTW.MEASURE)
p4 = plan_fft!(CuArray(u_F64))

t1 = @belapsed p1 * u setup=(u = rand(ComplexF64, Nx, Ny));
t2 = @belapsed p2 * u setup=(u = rand(ComplexF64, Nx, Ny));
t3 = @belapsed p3 * u setup=(u = rand(ComplexF64, Nx, Ny));
t4 = @belapsed (CUDA.@sync p4 * u) setup=(u = CUDA.rand(ComplexF64, Nx, Ny));

df_F64 = DataFrame(case = ["CPU 1T", "CPU $(cpu.cores)T", "CPU $(cpu.threads)T", "GPU"],
               time_ms = [t1, t2, t3, t4] * 1000, 
               performance = t2 ./ [t1, t2, t3, t4])
println("FP64:")
println(df_F64)

## FP32
u_F32 = rand(ComplexF32, Nx, Ny)

FFTW.set_num_threads(1)
p1 = plan_fft!(u_F32, flags=FFTW.MEASURE)
FFTW.set_num_threads(cpu.cores)
p2 = plan_fft!(u_F32, flags=FFTW.MEASURE)
FFTW.set_num_threads(cpu.threads)
p3 = plan_fft!(u_F32, flags=FFTW.MEASURE)
p4 = plan_fft!(CuArray(u_F32))

t1 = @belapsed p1 * u setup=(u = rand(ComplexF32, Nx, Ny));
t2 = @belapsed p2 * u setup=(u = rand(ComplexF32, Nx, Ny));
t3 = @belapsed p3 * u setup=(u = rand(ComplexF32, Nx, Ny));
t4 = @belapsed (CUDA.@sync p4 * u) setup=(u = CUDA.rand(ComplexF32, Nx, Ny));

df_F32 = DataFrame(case = ["CPU 1T", "CPU $(cpu.cores)T", "CPU $(cpu.threads)T", "GPU"],
                   time_ms = [t1, t2, t3, t4] * 1000, 
                   performance = t2 ./ [t1, t2, t3, t4])
println("\nFP32:")
println(df_F32)
