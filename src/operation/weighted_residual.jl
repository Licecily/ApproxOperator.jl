module WeightedResidual

using ..ApproxOperator: AbstractElement
# ===== 一维加权残余法算子 =====
# 1. Kuu: k ∫ Nᵢ ∂Nⱼ/∂t dt
function ∫kNṄdt(ap::T,Kuu::AbstractMatrix) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        N  = ξ[:𝝭]       
        Ṅ  = ξ[:∂𝝭∂y]    
        𝑤  = ξ.𝑤          
        kᶜ = ξ.k          
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                Kuu[I,J] += kᶜ * N[i] * Ṅ[j] * 𝑤
            end
        end
    end
end

# 2. Kuv: k ∫ Nᵢ Nⱼ dt 
function ∫kNNdt(ap::T,Kuv::AbstractMatrix) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        N  = ξ[:𝝭]
        𝑤  = ξ.𝑤
        kᶜ = ξ.k
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                Kuv[I,J] -= kᶜ * N[i] * N[j] * 𝑤
            end
        end
    end
end

# 3. Kvv: ∫ Nᵢ ∂Nⱼ/∂t dt
function ∫mNṄdt(ap::T,Kvv::AbstractMatrix) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        Ṅ = ξ[:∂𝝭∂y]
        𝑤  = ξ.𝑤
        mᶜ = ξ.m       
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                Kvv[I,J] += mᶜ * N[i] * Ṅ[j] * 𝑤
            end
        end
    end
end
# ===== 二维（时空）加权残余法算子 =====
# 二维坐标 (x,y) 中 y 为时间方向
# 1. Kuu: k ∫∫ Nᵢ ∂Nⱼ/∂t dxdt
function ∫∫kNṄdxdt(ap::T, k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        Ṅ = ξ[:∂𝝭∂y]
        kᶜ = ξ.k
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += kᶜ * N[i] * Ṅ[j] * 𝑤
            end
        end
    end
end

# 2. Kuv: k ∫∫ Nᵢ Nⱼ dxdt 
function ∫∫kNNdxdt(ap::T, k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        kᶜ = ξ.k
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] -= kᶜ * N[i] * N[j] * 𝑤
            end
        end
    end
end

# 3. Kvv: ∫∫ Nᵢ ∂Nⱼ/∂t dxdt
function ∫∫NṄdxdt(ap::T, k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        N = ξ[:𝝭]
        Ṅ = ξ[:∂𝝭∂y]
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += N[i] * Ṅ[j] * 𝑤
            end
        end
    end
end

# 4. Kvu: c² ∫∫ (∂Nᵢ/∂x ∂Nⱼ/∂x) dxdt
function ∫∫c²B₁B₁dxdt(ap::T, k::AbstractMatrix{Float64}) where T<:AbstractElement
    𝓒 = ap.𝓒; 𝓖 = ap.𝓖
    for ξ in 𝓖
        B₁ = ξ[:∂𝝭∂x]
        c = ξ.c
        𝑤 = ξ.𝑤
        for (i,xᵢ) in enumerate(𝓒)
            I = xᵢ.𝐼
            for (j,xⱼ) in enumerate(𝓒)
                J = xⱼ.𝐼
                k[I,J] += c^2 * B₁[i] * B₁[j] * 𝑤
            end
        end
    end
end
end