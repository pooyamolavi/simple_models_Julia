########################
# Julia code for Section 7 of "Simple Models and Biased Forecasts," by Pooya Molavi (2022)
# The code is licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
########################
# This file contains functions used to compute the equilibirum of the DMP model
# as well is its response to labor productivity and separation rate shocks.
########################

########################
# dictionaries
########################

########################
ψ_dict = Dict()
for (idx, name_idx) in enumerate(ψ_vars)
   ψ_dict[name_idx] = idx
end

γ_dict = Dict()
for (idx, name_idx) in enumerate(γ_vars)
   γ_dict[name_idx] = idx
end

f_dict = Dict()
for (idx, name_idx) in enumerate(f_vars)
   f_dict[name_idx] = idx
end

ω_dict = Dict()
for (idx, name_idx) in enumerate(ω_vars)
   ω_dict[name_idx] = idx
end

x_dict = Dict()
for (idx, name_idx) in enumerate(x_vars)
   x_dict[name_idx] = idx
end

ϵ_dict = Dict()
for (idx, name_idx) in enumerate(ϵ_vars)
   ϵ_dict[name_idx] = idx
end
########################

########################
# types
########################

########################
@with_kw mutable struct calibrated_parameters
    ########################
    # deep parameters
    β::Float64 = 0.99
    s::Float64 = 0.035
    b::Float64 = 0.4
    α::Float64 = 0.72
    δ::Float64 = 0.72

    # steady state targets
    auto_corr_a::Float64 = 0.96
    auto_corr_s::Float64 = 0.90
    SD_a::Float64 = 1
    SD_s::Float64 = 1/10
    corr_a_s::Float64 = -0.4
    p::Float64 = 0.4  # steady-state job-finding rate
end
########################

########################
mutable struct exogenous_parameters
    # deep parameters
    β::Float64
    s::Float64
    b::Float64
    α::Float64
    δ::Float64

    # steady state values
    p::Float64
    J::Float64
    w::Float64

    # composite parameters
    ζ::Float64
    χ::Float64

    # shock process
    F::Array{Float64}
    Σ::Array{Float64}
end
########################

########################
function exogenous_parameters(x::calibrated_parameters)
    # steady state values
    w = (x.δ*(1-x.β*(1-x.s-x.p))+(1-x.δ)*(1-x.β*(1-x.s))*x.b)/(1-x.β*(1-x.s-x.δ*x.p))
    J = (1-w)/(1-x.β*(1-x.s))

    # composite parameters
    ζ = x.β*x.s*(1-w)/(1-x.β*(1-x.s))
    χ = x.β*(1-x.δ)*(w-x.b)/(1-x.β*(1-x.s-x.p))

    # the shock process
    F = zeros(n_ϵ, n_ϵ)
    Σ = zeros(n_ϵ, n_ϵ)
    F[ϵ_dict["a"],ϵ_dict["a"]] = x.auto_corr_a
    F[ϵ_dict["s"],ϵ_dict["s"]] = x.auto_corr_s
    Σ[ϵ_dict["a"],ϵ_dict["a"]] = (1-x.auto_corr_a^2)*x.SD_a^2
    Σ[ϵ_dict["a"],ϵ_dict["s"]] = x.corr_a_s*(1-x.auto_corr_a*x.auto_corr_s)*x.SD_a*x.SD_s
    Σ[ϵ_dict["s"],ϵ_dict["a"]] = x.corr_a_s*(1-x.auto_corr_a*x.auto_corr_s)*x.SD_a*x.SD_s
    Σ[ϵ_dict["s"],ϵ_dict["s"]] = (1-x.auto_corr_s^2)*x.SD_s^2

    return exogenous_parameters(x.β,x.s,x.b,x.α,x.δ,x.p,J,w,ζ,χ,F,Σ)
end
########################

########################
function exogenous_parameters()
    return exogenous_parameters(calibrated_parameters())
end
########################

########################
mutable struct endogenous_parameters
    ψθu::Float64
    ψθa::Float64
    ψθs::Float64
end

function endogenous_parameters(e_vec)
    endogenous_parameters(e_vec[1],e_vec[2],e_vec[3])
end
########################

########################
# functions
########################

########################
# the rational-expectations equilibrium
function REE(x::exogenous_parameters)
    ########################
    # REE variables
    # solution is   fₜ = Ffₜ₋₁ + Lϵₜ                  where   ϵₜ ~ N(0,Σ)
    ########################
    Φ = zeros(n_γ, n_γ)
    ω = zeros(n_γ)
    ########################
    F = zeros(n_f, n_f)
    L = zeros(n_f, n_ϵ)
    ########################
    # θa
    ω[γ_dict["θa"]] = x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.b)/(1-x.β*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s))/(x.α*x.J)
    Φ[γ_dict["θa"],γ_dict["wa"]] = -x.F[ϵ_dict["a"],ϵ_dict["a"]]/(1-x.β*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s))/(x.α*x.J)
    ########################
    # θs
    ω[γ_dict["θs"]] = -x.F[ϵ_dict["s"],ϵ_dict["s"]]*x.ζ/(1-x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s))/(x.α*x.J)
    Φ[γ_dict["θs"],γ_dict["ws"]] = -x.F[ϵ_dict["s"],ϵ_dict["s"]]/(1-x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s))/(x.α*x.J)
    ########################
    # wa
    ω[γ_dict["wa"]] += x.δ*(1-x.b)
    Φ[γ_dict["wa"],γ_dict["θa"]] += x.p*x.χ*(1-x.α)
    ω[γ_dict["wa"]] += x.β*x.δ*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s)*(1-x.b)/(1-x.β*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s))
    Φ[γ_dict["wa"],γ_dict["wa"]] += -x.β*x.δ*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s)/(1-x.β*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s))
    Φ[γ_dict["wa"],γ_dict["θa"]] += x.β*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s-x.p)/(1-x.β*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s-x.p))*x.p*x.χ*(1-x.α)
    Φ[γ_dict["wa"],γ_dict["wa"]] += -x.β*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s-x.p)/(1-x.β*x.F[ϵ_dict["a"],ϵ_dict["a"]]*(1-x.s-x.p))*(1-x.δ)
    ########################
    # ws
    ω[γ_dict["ws"]] += x.s*x.χ-x.δ*x.ζ
    Φ[γ_dict["ws"],γ_dict["θs"]] += x.p*x.χ*(1-x.α)
    ω[γ_dict["ws"]] += -x.β*x.δ*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s)*x.ζ/(1-x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s))
    Φ[γ_dict["ws"],γ_dict["ws"]] += -x.β*x.δ*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s)/(1-x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s))
    Φ[γ_dict["ws"],γ_dict["θs"]] += x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s-x.p)/(1-x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s-x.p))*x.p*x.χ*(1-x.α)
    ω[γ_dict["ws"]] += x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s-x.p)/(1-x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s-x.p))*x.s*x.χ
    Φ[γ_dict["ws"],γ_dict["ws"]] += -x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s-x.p)/(1-x.β*x.F[ϵ_dict["s"],ϵ_dict["s"]]*(1-x.s-x.p))*(1-x.δ)
    ########################
    γ = inv(I-Φ)*ω
    ########################
    T = zeros(n_x,n_f)
    T[x_dict["u"],f_dict["u"]]=1.0
    T[x_dict["a"],f_dict["a"]]=1.0
    T[x_dict["s"],f_dict["s"]]=1.0
    T[x_dict["θ"],f_dict["a"]]=γ[γ_dict["θa"]]
    T[x_dict["θ"],f_dict["s"]]=γ[γ_dict["θs"]]
    T[x_dict["w"],f_dict["a"]]=γ[γ_dict["wa"]]
    T[x_dict["w"],f_dict["s"]]=γ[γ_dict["ws"]]
    T[x_dict["p"],f_dict["a"]]=(1-x.α)*γ[γ_dict["θa"]]
    T[x_dict["p"],f_dict["s"]]=(1-x.α)*γ[γ_dict["θs"]]
    T[x_dict["v"],f_dict["u"]]=T[x_dict["θ"],f_dict["u"]]+T[x_dict["u"],f_dict["u"]]
    T[x_dict["v"],f_dict["a"]]=T[x_dict["θ"],f_dict["a"]]+T[x_dict["u"],f_dict["a"]]
    T[x_dict["v"],f_dict["s"]]=T[x_dict["θ"],f_dict["s"]]+T[x_dict["u"],f_dict["s"]]
    ########################
    # u
    F[f_dict["u"],f_dict["u"]] = 1-x.s-x.p
    F[f_dict["u"],f_dict["a"]] = -(1-x.α)*x.p*γ[γ_dict["θa"]]
    F[f_dict["u"],f_dict["s"]] = x.p-(1-x.α)*x.p*γ[γ_dict["θs"]]
    ########################
    # a
    F[f_dict["a"],f_dict["a"]] = x.F[ϵ_dict["a"],ϵ_dict["a"]]
    L[f_dict["a"],ϵ_dict["a"]] = 1
    ########################
    # s
    F[f_dict["s"],f_dict["s"]] = x.F[ϵ_dict["s"],ϵ_dict["s"]]
    L[f_dict["s"],ϵ_dict["s"]] = 1
    ########################
    return T, F, L
end
########################

########################
# temporary equilibrium
function temporary_equilibrium(x::exogenous_parameters, e::endogenous_parameters; tol=1e-12, max_time=1e6)
    ########################
    # CREE
    # model is      fₜ = Ffₜ + Lϵₜ
    # ########################
    F = zeros(n_f, n_f)
    L = zeros(n_f, n_ϵ)
    ########################
    # u
    F[f_dict["u"],f_dict["u"]] = 1-x.s-x.p-(1-x.α)*x.p*e.ψθu
    F[f_dict["u"],f_dict["a"]] = -(1-x.α)*x.p*e.ψθa
    F[f_dict["u"],f_dict["s"]] = x.p-(1-x.α)*x.p*e.ψθs
    ########################
    # a
    F[f_dict["a"],f_dict["a"]] = x.F[ϵ_dict["a"],ϵ_dict["a"]]
    F[f_dict["a"],f_dict["s"]] = x.F[ϵ_dict["a"],ϵ_dict["s"]]
    L[f_dict["a"],ϵ_dict["a"]] = 1
    ########################
    # s
    F[f_dict["s"],f_dict["s"]] = x.F[ϵ_dict["s"],ϵ_dict["s"]]
    F[f_dict["s"],f_dict["a"]] = x.F[ϵ_dict["s"],ϵ_dict["a"]]
    L[f_dict["s"],ϵ_dict["s"]] = 1
    ########################
    Σ = L*x.Σ*L'

    a,η,p,q,~ = best_1_state_full_info_model(F,Σ; tol=tol,max_time=max_time)
    ########################
    Φ = zeros(n_ψ, n_ψ)
    ω = zeros(n_ψ)
    ########################
    # θu
    ω[ψ_dict["θu"]] += a*p[f_dict["u"]]/(1-a*x.β*(1-x.s))*(1-x.b)/(x.α*x.J)*q[f_dict["a"]]
    Φ[ψ_dict["θu"],ψ_dict["wa"]] += -a*p[f_dict["u"]]/(1-a*x.β*(1-x.s))/(x.α*x.J)*q[f_dict["a"]]
    Φ[ψ_dict["θu"],ψ_dict["wu"]] += -a*p[f_dict["u"]]/(1-a*x.β*(1-x.s))/(x.α*x.J)*q[f_dict["u"]]
    ω[ψ_dict["θu"]] += -a*p[f_dict["u"]]/(1-a*x.β*(1-x.s))*x.ζ/(x.α*x.J)*q[f_dict["s"]]
    Φ[ψ_dict["θu"],ψ_dict["ws"]] += -a*p[f_dict["u"]]/(1-a*x.β*(1-x.s))/(x.α*x.J)*q[f_dict["s"]]
    ########################
    # θa
    ω[ψ_dict["θa"]] += a*p[f_dict["a"]]/(1-a*x.β*(1-x.s))*(1-x.b)/(x.α*x.J)*q[f_dict["a"]]
    Φ[ψ_dict["θa"],ψ_dict["wa"]] += -a*p[f_dict["a"]]/(1-a*x.β*(1-x.s))/(x.α*x.J)*q[f_dict["a"]]
    Φ[ψ_dict["θa"],ψ_dict["wu"]] += -a*p[f_dict["a"]]/(1-a*x.β*(1-x.s))/(x.α*x.J)*q[f_dict["u"]]
    ω[ψ_dict["θa"]] += -a*p[f_dict["a"]]/(1-a*x.β*(1-x.s))*x.ζ/(x.α*x.J)*q[f_dict["s"]]
    Φ[ψ_dict["θa"],ψ_dict["ws"]] += -a*p[f_dict["a"]]/(1-a*x.β*(1-x.s))/(x.α*x.J)*q[f_dict["s"]]
    ########################
    # θs
    ω[ψ_dict["θs"]] += a*p[f_dict["s"]]/(1-a*x.β*(1-x.s))*(1-x.b)/(x.α*x.J)*q[f_dict["a"]]
    Φ[ψ_dict["θs"],ψ_dict["wa"]] += -a*p[f_dict["s"]]/(1-a*x.β*(1-x.s))/(x.α*x.J)*q[f_dict["a"]]
    Φ[ψ_dict["θs"],ψ_dict["wu"]] += -a*p[f_dict["s"]]/(1-a*x.β*(1-x.s))/(x.α*x.J)*q[f_dict["u"]]
    ω[ψ_dict["θs"]] += -a*p[f_dict["s"]]/(1-a*x.β*(1-x.s))*x.ζ/(x.α*x.J)*q[f_dict["s"]]
    Φ[ψ_dict["θs"],ψ_dict["ws"]] += -a*p[f_dict["s"]]/(1-a*x.β*(1-x.s))/(x.α*x.J)*q[f_dict["s"]]
    ########################
    # wu
    Φ[ψ_dict["wu"],ψ_dict["θu"]] += x.p*x.χ*(1-x.α)
    ω[ψ_dict["wu"]] += a*x.β*x.δ*(1-x.s)*p[f_dict["u"]]/(1-a*x.β*(1-x.s))*(1-x.b)*q[f_dict["a"]]
    Φ[ψ_dict["wu"],ψ_dict["wa"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["u"]]/(1-a*x.β*(1-x.s))*q[f_dict["a"]]
    Φ[ψ_dict["wu"],ψ_dict["wu"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["u"]]/(1-a*x.β*(1-x.s))*q[f_dict["u"]]
    ω[ψ_dict["wu"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["u"]]/(1-a*x.β*(1-x.s))*x.ζ*q[f_dict["s"]]
    Φ[ψ_dict["wu"],ψ_dict["ws"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["u"]]/(1-a*x.β*(1-x.s))*q[f_dict["s"]]
    Φ[ψ_dict["wu"],ψ_dict["θa"]] += a*x.β*(1-x.s-x.p)*p[f_dict["u"]]/(1-a*x.β*(1-x.s-x.p))*x.p*x.χ*(1-x.α)*q[f_dict["a"]]
    Φ[ψ_dict["wu"],ψ_dict["wa"]] += -a*x.β*(1-x.s-x.p)*p[f_dict["u"]]/(1-a*x.β*(1-x.s-x.p))*(1-x.δ)*q[f_dict["a"]]
    Φ[ψ_dict["wu"],ψ_dict["θu"]] += a*x.β*(1-x.s-x.p)*p[f_dict["u"]]/(1-a*x.β*(1-x.s-x.p))*x.p*x.χ*(1-x.α)*q[f_dict["u"]]
    Φ[ψ_dict["wu"],ψ_dict["wu"]] += -a*x.β*(1-x.s-x.p)*p[f_dict["u"]]/(1-a*x.β*(1-x.s-x.p))*(1-x.δ)*q[f_dict["u"]]
    Φ[ψ_dict["wu"],ψ_dict["θs"]] += a*x.β*(1-x.s-x.p)*p[f_dict["u"]]/(1-a*x.β*(1-x.s-x.p))*x.p*x.χ*(1-x.α)*q[f_dict["s"]]
    Φ[ψ_dict["wu"],ψ_dict["ws"]] += -a*x.β*(1-x.s-x.p)*p[f_dict["u"]]/(1-a*x.β*(1-x.s-x.p))*(1-x.δ)*q[f_dict["s"]]
    ω[ψ_dict["wu"]] += a*x.β*(1-x.s-x.p)*p[f_dict["u"]]/(1-a*x.β*(1-x.s-x.p))*x.s*x.χ*q[f_dict["s"]]
    ########################
    # wa
    ω[ψ_dict["wa"]] += x.δ*(1-x.b)
    Φ[ψ_dict["wa"],ψ_dict["θa"]] += x.p*x.χ*(1-x.α)
    ω[ψ_dict["wa"]] += a*x.β*x.δ*(1-x.s)*p[f_dict["a"]]/(1-a*x.β*(1-x.s))*(1-x.b)*q[f_dict["a"]]
    Φ[ψ_dict["wa"],ψ_dict["wa"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["a"]]/(1-a*x.β*(1-x.s))*q[f_dict["a"]]
    Φ[ψ_dict["wa"],ψ_dict["wu"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["a"]]/(1-a*x.β*(1-x.s))*q[f_dict["u"]]
    ω[ψ_dict["wa"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["a"]]/(1-a*x.β*(1-x.s))*x.ζ*q[f_dict["s"]]
    Φ[ψ_dict["wa"],ψ_dict["ws"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["a"]]/(1-a*x.β*(1-x.s))*q[f_dict["s"]]
    Φ[ψ_dict["wa"],ψ_dict["θa"]] += a*x.β*(1-x.s-x.p)*p[f_dict["a"]]/(1-a*x.β*(1-x.s-x.p))*x.p*x.χ*(1-x.α)*q[f_dict["a"]]
    Φ[ψ_dict["wa"],ψ_dict["wa"]] += -a*x.β*(1-x.s-x.p)*p[f_dict["a"]]/(1-a*x.β*(1-x.s-x.p))*(1-x.δ)*q[f_dict["a"]]
    Φ[ψ_dict["wa"],ψ_dict["θu"]] += a*x.β*(1-x.s-x.p)*p[f_dict["a"]]/(1-a*x.β*(1-x.s-x.p))*x.p*x.χ*(1-x.α)*q[f_dict["u"]]
    Φ[ψ_dict["wa"],ψ_dict["wu"]] += -a*x.β*(1-x.s-x.p)*p[f_dict["a"]]/(1-a*x.β*(1-x.s-x.p))*(1-x.δ)*q[f_dict["u"]]
    Φ[ψ_dict["wa"],ψ_dict["θs"]] += a*x.β*(1-x.s-x.p)*p[f_dict["a"]]/(1-a*x.β*(1-x.s-x.p))*x.p*x.χ*(1-x.α)*q[f_dict["s"]]
    Φ[ψ_dict["wa"],ψ_dict["ws"]] += -a*x.β*(1-x.s-x.p)*p[f_dict["a"]]/(1-a*x.β*(1-x.s-x.p))*(1-x.δ)*q[f_dict["s"]]
    ω[ψ_dict["wa"]] += a*x.β*(1-x.s-x.p)*p[f_dict["a"]]/(1-a*x.β*(1-x.s-x.p))*x.s*x.χ*q[f_dict["s"]]
    ########################
    # ws
    ω[ψ_dict["ws"]] += x.s*x.χ-x.δ*x.ζ
    Φ[ψ_dict["ws"],ψ_dict["θs"]] += x.p*x.χ*(1-x.α)
    ω[ψ_dict["ws"]] += a*x.β*x.δ*(1-x.s)*p[f_dict["s"]]/(1-a*x.β*(1-x.s))*(1-x.b)*q[f_dict["a"]]
    Φ[ψ_dict["ws"],ψ_dict["wa"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["s"]]/(1-a*x.β*(1-x.s))*q[f_dict["a"]]
    Φ[ψ_dict["ws"],ψ_dict["wu"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["s"]]/(1-a*x.β*(1-x.s))*q[f_dict["u"]]
    ω[ψ_dict["ws"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["s"]]/(1-a*x.β*(1-x.s))*x.ζ*q[f_dict["s"]]
    Φ[ψ_dict["ws"],ψ_dict["ws"]] += -a*x.β*x.δ*(1-x.s)*p[f_dict["s"]]/(1-a*x.β*(1-x.s))*q[f_dict["s"]]
    Φ[ψ_dict["ws"],ψ_dict["θa"]] += a*x.β*(1-x.s-x.p)*p[f_dict["s"]]/(1-a*x.β*(1-x.s-x.p))*x.p*x.χ*(1-x.α)*q[f_dict["a"]]
    Φ[ψ_dict["ws"],ψ_dict["wa"]] += -a*x.β*(1-x.s-x.p)*p[f_dict["s"]]/(1-a*x.β*(1-x.s-x.p))*(1-x.δ)*q[f_dict["a"]]
    Φ[ψ_dict["ws"],ψ_dict["θu"]] += a*x.β*(1-x.s-x.p)*p[f_dict["s"]]/(1-a*x.β*(1-x.s-x.p))*x.p*x.χ*(1-x.α)*q[f_dict["u"]]
    Φ[ψ_dict["ws"],ψ_dict["wu"]] += -a*x.β*(1-x.s-x.p)*p[f_dict["s"]]/(1-a*x.β*(1-x.s-x.p))*(1-x.δ)*q[f_dict["u"]]
    Φ[ψ_dict["ws"],ψ_dict["θs"]] += a*x.β*(1-x.s-x.p)*p[f_dict["s"]]/(1-a*x.β*(1-x.s-x.p))*x.p*x.χ*(1-x.α)*q[f_dict["s"]]
    Φ[ψ_dict["ws"],ψ_dict["ws"]] += -a*x.β*(1-x.s-x.p)*p[f_dict["s"]]/(1-a*x.β*(1-x.s-x.p))*(1-x.δ)*q[f_dict["s"]]
    ω[ψ_dict["ws"]] += a*x.β*(1-x.s-x.p)*p[f_dict["s"]]/(1-a*x.β*(1-x.s-x.p))*x.s*x.χ*q[f_dict["s"]]
    ########################
    ψ = inv(I-Φ)*ω
    ########################
    T = zeros(n_x,n_f)
    T[x_dict["u"],f_dict["u"]]=1.0
    T[x_dict["a"],f_dict["a"]]=1.0
    T[x_dict["s"],f_dict["s"]]=1.0
    T[x_dict["θ"],f_dict["u"]]=ψ[ψ_dict["θu"]]
    T[x_dict["θ"],f_dict["a"]]=ψ[ψ_dict["θa"]]
    T[x_dict["θ"],f_dict["s"]]=ψ[ψ_dict["θs"]]
    T[x_dict["w"],f_dict["u"]]=ψ[ψ_dict["wu"]]
    T[x_dict["w"],f_dict["a"]]=ψ[ψ_dict["wa"]]
    T[x_dict["w"],f_dict["s"]]=ψ[ψ_dict["ws"]]
    T[x_dict["p"],f_dict["u"]]=(1-x.α)*ψ[ψ_dict["θu"]]
    T[x_dict["p"],f_dict["a"]]=(1-x.α)*ψ[ψ_dict["θa"]]
    T[x_dict["p"],f_dict["s"]]=(1-x.α)*ψ[ψ_dict["θs"]]
    T[x_dict["v"],f_dict["u"]]=T[x_dict["θ"],f_dict["u"]]+T[x_dict["u"],f_dict["u"]]
    T[x_dict["v"],f_dict["a"]]=T[x_dict["θ"],f_dict["a"]]+T[x_dict["u"],f_dict["a"]]
    T[x_dict["v"],f_dict["s"]]=T[x_dict["θ"],f_dict["s"]]+T[x_dict["u"],f_dict["s"]]
    ########################
    return F, L, T, a, η, p, q
end

########################

########################
function fixed_point_violation(x::exogenous_parameters, e::endogenous_parameters; instability_penalty=1e50, tol=1e-12, max_time=1e6)
    ########################
    F = zeros(n_f, n_f)
    L = zeros(n_f, n_ϵ)
    F = zeros(n_f, n_f)
    L = zeros(n_f, n_ϵ)
    ########################
    # u
    F[f_dict["u"],f_dict["u"]] = 1-x.s-x.p-(1-x.α)*x.p*e.ψθu
    F[f_dict["u"],f_dict["a"]] = -(1-x.α)*x.p*e.ψθa
    F[f_dict["u"],f_dict["s"]] = x.p-(1-x.α)*x.p*e.ψθs
    ########################
    # a
    F[f_dict["a"],f_dict["a"]] = x.F[ϵ_dict["a"],ϵ_dict["a"]]
    F[f_dict["a"],f_dict["s"]] = x.F[ϵ_dict["a"],ϵ_dict["s"]]
    ########################
    # a
    F[f_dict["s"],f_dict["s"]] = x.F[ϵ_dict["s"],ϵ_dict["s"]]
    F[f_dict["s"],f_dict["a"]] = x.F[ϵ_dict["s"],ϵ_dict["a"]]
    ########################
    if maximum(abs.(eigvals(F)))>1
        return instability_penalty*exp(maximum(abs.(eigvals(F)))-1)
    else
        ~, ~, T, ~, η, ~, ~ = temporary_equilibrium(x,e)
        ψθu = T[x_dict["θ"],f_dict["u"]]
        ψθa = T[x_dict["θ"],f_dict["a"]]
        ψθs = T[x_dict["θ"],f_dict["s"]]

        return sqrt(η^2+(ψθu-e.ψθu)^2+(ψθa-e.ψθa)^2+(ψθs-e.ψθs)^2)
    end
end
########################

########################
function fixed_point_iteration(x::exogenous_parameters, e_init::endogenous_parameters; new_weight=0.01, fixed_iter=1000, fixed_tol=1e-10, inner_loop_tol=1e-12, inner_loop_max_cpu_time=1.0)
    ########################
    F = zeros(n_f, n_f)
    L = zeros(n_f, n_ϵ)
    e_new = e_init
    for i=1:fixed_iter
        e_old = e_new
        ########################
        # u
        F[f_dict["u"],f_dict["u"]] = 1-x.s-x.p-(1-x.α)*x.p*e_new.ψθu
        F[f_dict["u"],f_dict["a"]] = -(1-x.α)*x.p*e_new.ψθa
        F[f_dict["u"],f_dict["s"]] = x.p-(1-x.α)*x.p*e_new.ψθs
        ########################
        # a
        F[f_dict["a"],f_dict["a"]] = x.F[ϵ_dict["a"],ϵ_dict["a"]]
        F[f_dict["a"],f_dict["s"]] = x.F[ϵ_dict["a"],ϵ_dict["s"]]
        ########################
        # a
        F[f_dict["s"],f_dict["s"]] = x.F[ϵ_dict["s"],ϵ_dict["s"]]
        F[f_dict["s"],f_dict["a"]] = x.F[ϵ_dict["s"],ϵ_dict["a"]]
        ########################
        F, L, T = temporary_equilibrium(x, e_new)
        if maximum(abs.(eigvals(F)))>1
            return false, e_new
        else
            ~, ~, T, ~, η, ~, ~ = temporary_equilibrium(x,e_new)
            ψθu = T[x_dict["θ"],f_dict["u"]]
            ψθa = T[x_dict["θ"],f_dict["a"]]
            ψθs = T[x_dict["θ"],f_dict["s"]]
            change = sqrt((ψθu-e_old.ψθu)^2+(ψθa-e_old.ψθa)^2+(ψθs-e_old.ψθs)^2)
            println("i=", i, ";  change=", change)
            if change<fixed_tol
                return true, e_new
            end

            e_new = endogenous_parameters(ψθu*new_weight+e_old.ψθu*(1-new_weight),ψθa*new_weight+e_old.ψθa*(1-new_weight),ψθs*new_weight+e_old.ψθs*(1-new_weight))
        end
    end
    return false, e_new
end
########################

########################
# compute the CREE given exogenous parameters and shocks
function CREE(x::exogenous_parameters; e_vec_lb=[-5.0,-5.0,-5.0], e_vec_ub=[5.0,5.0,5.0], max_time=5*60, algorithm=:adaptive_de_rand_1_bin_radiuslimited, instability_penalty=1e50, tol=1e-10, inner_loop_tol=1e-12, inner_loop_max_cpu_time=1.0)
    function fp_violation(e_vec)
        e = endogenous_parameters(e_vec)
        return fixed_point_violation(x,e; instability_penalty=instability_penalty,tol=inner_loop_tol,max_time=inner_loop_max_cpu_time)
    end

    e_vec_bnd=Tuple{Float64,Float64}[]
    for i=1:3
        new_bnd = (e_vec_lb[i], e_vec_ub[i])
        e_vec_bnd = vcat(e_vec_bnd, new_bnd)
    end

    # optimize
    bboptim_prob = bbsetup(fp_violation;  Method=algorithm, SearchRange=e_vec_bnd, TargetFitness = 0.0, FitnessTolerance=tol)
    bboptim_sol = bboptimize(bboptim_prob, MaxTime=max_time)
    e_vec = best_candidate(bboptim_sol)
    return e_vec, bboptim_prob
end
########################
