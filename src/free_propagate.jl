function free_propagate(u::AbstractMatrix, trans::AbstractMatrix)
    ifft(fft(u) .* trans)
end

function free_propagate!(u, trans, plan, iplan)
    plan * u
    u .*= trans
    iplan * u
end