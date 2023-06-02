
function gaussian_kernel(σ, n=ceil(Int, (6σ-1)/2))
    kernel = @. exp(-(-n:n)^2 / (2σ^2))
    return kernel ./ sum(kernel)
end

const SSIM_K = (0.01, 0.03)

const SSIM_KERNEL = let
    k = gaussian_kernel(1.5, 5)
    (k*k')[:,:,:,:]
end

function ssim(x::T, y::T; peakval=1.0, K=SSIM_K, crop=true) where {T}
    groups = size(x, 3)
    kernel = repeat(T(SSIM_KERNEL), 1, 1, 1, groups)

    C₁, C₂ = @. (peakval * K)^2

    x, y = crop ? (x, y) : pad_reflect.((x, y), size(kernel,1) ÷ 2) 
    μx  = conv(x, kernel; groups=groups)
    μy  = conv(y, kernel; groups=groups)
    μx² = μx.^2
    μy² = μy.^2
    μxy = μx.*μy
    σx² = conv(x.^2, kernel; groups=groups) .- μx²
    σy² = conv(y.^2, kernel; groups=groups) .- μy²
    σxy = conv(x.*y, kernel; groups=groups) .- μxy

    ssim_map = @. (2μxy + C₁)*(2σxy + C₂)/((μx² + μy² + C₁)*(σx² + σy² + C₂))
    return mean(ssim_map)
end

    
