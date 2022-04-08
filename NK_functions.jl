###################################################
# Julia code for Section 5 of "Simple Models and Biased Forecasts," by Pooya Molavi (2022)
# The code is licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
###################################################
# This file contains functions used to compute the equilibirum of the NK model
# as well is its response to conventional monetary shocks and forward guidance.
# It is called by "NK.jl."
###################################################

########################
@with_kw mutable struct calibrated_parameters
    ########################
    # deep parameters
    β::Float64 = 0.99
    σ::Float64 = 1
    δ::Float64 = 3/4
    φ::Float64 = 5
    ϵ::Float64 = 9
    α::Float64 = 1/4
    Θ::Float64 = (1-α)/(1-α+α*ϵ)
    λ::Float64 = (1-δ)*(1-β*δ)/δ*Θ
    κ::Float64 = λ*(σ+(φ+α)/(1-α))
end
########################

########################
mutable struct parameters
    # deep parameters
    β::Float64
    σ::Float64
    δ::Float64
    κ::Float64

    # shocks
    a::Float64
    p::Array{Float64,1}
    q::Array{Float64,1}
    Γ₀::Array{Float64,2}

    # composite parameters
    vx::Array{Float64,1}
    vπ::Array{Float64,1}

    # endogenous parameters
    γx::Float64
    γπ::Float64

    # the mapping from fₜ to yₜ
    T::Array{Float64,2}
end
########################

########################
# calibrate a, p, and q given time series of o, i, and π
########################
function estimate_using_data(data; L = 30)
    ########################
    # compute autocovariance matrices
    ########################
    Γ = zeros(L,n_f,n_f)
    for l=1:L
       for y1 ∈ f_vars
          for y2 ∈ f_vars
             Γ[l,f_dict[y1],f_dict[y2]] = Statistics.cov(data[l:T,f_dict[y1]],data[1:T-(l-1),f_dict[y2]])
          end
       end
    end
    ########################

    ########################
    # compute autocorrelation matrices
    ########################
    C = zeros(L,n_f,n_f)
    for l=1:L
       C[l,:,:] = Γ[1,:,:]^(-1/2)*(Γ[l,:,:]+transpose(Γ[l,:,:]))/2*Γ[1,:,:]^(-1/2)
    end
    ########################

    ########################
    a,η,~ = best_1_state_model_Ipopt(Γ)

    if η > 1e-10
        return "not exponentially ergodic"
    else
        eig_C = eigen(C[2,:,:])
        ~, λ_max_idx = findmax(abs.(eig_C.values))
        u_max = real.(eig_C.vectors[:,λ_max_idx])
        p = -Γ[1,:,:]^(-1/2)*u_max
        q = -Γ[1,:,:]^(1/2)*u_max

        return a,p,q,Γ[1,:,:],Γ[2,:,:],C
    end
end
########################

########################
function parameters(data)
    x = calibrated_parameters()

    a,p,q,Γ₀,~,~ = estimate_using_data(data)

    γx = a*(q[f_dict["x"]]-x.σ*q[f_dict["π"]])
    γπ = a*x.β*q[f_dict["π"]]

    vx = zeros(n_f)
    vx[f_dict["x"]] = (1-x.β)/x.β + 1 - γx*p[f_dict["x"]]
    vx[y_dict["π"]] = -x.σ/x.β - γx*p[f_dict["π"]]
    vx[f_dict["i"]] = -γx*p[f_dict["i"]]

    vπ = zeros(n_f)
    vπ[f_dict["x"]] = - γπ*p[f_dict["x"]]
    vπ[f_dict["π"]] = (1-x.δ)/x.δ + 1 - γπ*p[f_dict["π"]]
    vπ[f_dict["i"]] =  - γπ*p[f_dict["i"]]

    # the matrix that maps fₜ to yₜ ## NOT TO BE CONSUFED with the duration of FG ##
    T = zeros(n_y,n_f)
    T[y_dict["x"],f_dict["x"]] = 1
    T[y_dict["π"],f_dict["π"]] = 1
    T[y_dict["i"],f_dict["i"]] = 1

    T[y_dict["n"],f_dict["x"]] = (1/x.σ)*(1-γx*p[f_dict["x"]])
    T[y_dict["n"],f_dict["π"]] = (1/x.σ)*(-γx*p[f_dict["π"]])
    T[y_dict["n"],f_dict["i"]] = (1/x.σ)*(x.σ-γx*p[f_dict["i"]])

    T[y_dict["μ"],f_dict["x"]] = -x.κ - γπ*p[f_dict["x"]]
    T[y_dict["μ"],f_dict["π"]] = 1-γπ*p[f_dict["π"]]
    T[y_dict["μ"],f_dict["i"]] = -γπ*p[f_dict["i"]]

    return parameters(x.β,x.σ,x.δ,x.κ,a,p,q,Γ₀,vx,vπ,γx,γπ, T)
end
########################


########################
function forward_guidance_constants(x::parameters, T::Int64)
    # coordinate vector
    eᵢ = zeros(3)
    eᵢ[f_dict["i"]] = 1


    ########################
    # compute Σωω
    ########################
    Σωω = zeros(n_f+T, n_f+T)

    # upper left block
    Σωω[1:n_f,1:n_f] = x.Γ₀

    # upper block
    for τ=1:T
        Σωω[1:n_f,n_f+τ] = x.a^τ*x.q[f_dict["i"]]*x.Γ₀*x.p
    end

    # lower left block
    for τ=1:T
        Σωω[n_f+τ,1:n_f] = x.a^τ*x.q[f_dict["i"]]*x.p'*x.Γ₀
    end

    # diagonals
    for τ=1:T
        Σωω[n_f+τ,n_f+τ] = eᵢ'*x.Γ₀*eᵢ
    end

    # off diagonals
    for s=1:T
        for τ=1:T
            Σωω[n_f+s,n_f+τ] = x.a^abs(s-τ)*x.q[f_dict["i"]]*x.p'*x.Γ₀*eᵢ
        end
    end
    ########################

    ########################
    # χx = \sum_{s=1}^\infty \beta^s \Sigma_{f_s\omega_T}
    χx = zeros(n_f,n_f+T)
    χx[:,1:n_f] = x.a*x.β*x.q*x.p'*x.Γ₀/(1-x.a*x.β)
    for τ=1:T
        χx[:,n_f+τ] = x.β^τ*x.Γ₀*eᵢ+x.a*x.β^(τ+1)*x.q*x.p'*x.Γ₀*eᵢ/(1-x.a*x.β)+(x.a^τ*x.β-x.a*x.β^τ)/(x.a-x.β)*x.q[f_dict["i"]]*x.Γ₀*x.p
    end

    ########################
    # χπ = \sum_{s=1}^\infty (\beta\delta)^s \Sigma_{f_s\omega_T}
    χπ = zeros(n_f,n_f+T)
    χπ[:,1:n_f] = x.a*x.β*x.δ*x.q*x.p'*x.Γ₀/(1-x.a*x.β*x.δ)
    for τ=1:T
        χπ[:,n_f+τ] = (x.β*x.δ)^τ*x.Γ₀*eᵢ+x.a*(x.β*x.δ)^(τ+1)*x.q*x.p'*x.Γ₀*eᵢ/(1-x.a*x.β*x.δ)+(x.a^τ*x.β*x.δ-x.a*(x.β*x.δ)^τ)/(x.a-x.β*x.δ)*x.q[f_dict["i"]]*x.Γ₀*x.p
    end

    ########################
    ψx = (x.vx'*χx*inv(Σωω))[1:n_f]
    ψπ = (x.vπ'*χπ*inv(Σωω))[1:n_f]
    ψxi_fg = (x.vx'*χx*inv(Σωω))[n_f+1:n_f+T]
    ψπi_fg = (x.vπ'*χπ*inv(Σωω))[n_f+1:n_f+T]
    return ψx, ψπ, ψxi_fg, ψπi_fg
end
########################

########################
function forward_guidance(x::parameters, i, T)
    outcome = zeros(n_f, T+1)
    Φ = zeros(n_f, n_f)
    ω = zeros(n_f)
    for t=1:T+1
        ψx, ψπ, ψxi_fg, ψπi_fg = forward_guidance_constants(x,T-t+1)

        # x
        Φ[f_dict["x"],f_dict["x"]] = ψx[f_dict["x"]]
        Φ[f_dict["x"],f_dict["π"]] = ψx[f_dict["π"]]
        Φ[f_dict["x"],f_dict["i"]] = -x.σ + ψx[f_dict["i"]]
        ω[f_dict["x"]] = ψxi_fg'*i[t+1:T+1]

        # π
        Φ[f_dict["π"],f_dict["x"]] = x.κ + ψπ[f_dict["x"]]
        Φ[f_dict["π"],f_dict["π"]] = ψπ[f_dict["π"]]
        Φ[f_dict["π"],f_dict["i"]] = ψπ[f_dict["i"]]
        ω[f_dict["π"]] = ψπi_fg'*i[t+1:T+1]

        # i
        ω[f_dict["i"]] = i[t]

        outcome[:,t]=inv(I-Φ)*ω
    end
    return outcome
end
