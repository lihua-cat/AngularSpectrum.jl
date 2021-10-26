function isuniform(v::AbstractVector)
    dv = v[2:end] - v[1:end-1]
    return all(dv .≈ dv[1])
end

function isshift(v::AbstractVector)
    isuniform(v) || error("v is not uniform")
    N = length(v)
    return v[N÷2+1] == zero(eltype(v))
end