function free_propagate(u::AbstractMatrix, trans::AbstractMatrix)
    ifft(fft(u) .* trans)
end

function free_propagate!(u, trans, plan, iplan)
    plan * u
    u .*= trans
    iplan * u
end

function plan_as(u)
    p, ip = plan_fft!(u), plan_ifft!(u)
    return (p, ip)
end