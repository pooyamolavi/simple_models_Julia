###################################################
# Julia code for Section 7 of "Simple Models and Biased Forecasts," by Pooya Molavi (2022)
# The code is licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
###################################################
# This file generates Figures 4 and 5 from the paper.
###################################################
# It calls "multi_variate_functions.jl" and "DMP_functions.jl."
###################################################

using Parameters
using SolveDSGE
using LinearAlgebra
using BlackBoxOptim
using JuMP
using Ipopt
using MatrixEquations
using Optim
using Plots

########################
# variables
########################
ψ_vars = ["θu","θa","θs","wu","wa","ws"]
γ_vars = ["θa","θs","wa","ws"]
f_vars = ["u","a","s"]
ω_vars = ["θ","w","p","v"]
x_vars = [f_vars; ω_vars]
ϵ_vars = ["a","s"]

n_ψ = size(ψ_vars,1)
n_γ = size(γ_vars,1)
n_f = size(f_vars,1)
n_ω = size(ω_vars,1)
n_x = size(x_vars,1)
n_ϵ = size(ϵ_vars, 1)
########################

include("multi_variate_functions.jl")
include("DMP_functions.jl")

ex_params = exogenous_parameters()

########################
# find the CREE
########################
e_vec, bboptim_prob = CREE(ex_params; max_time=10*3600)
bboptim_sol = bboptimize(bboptim_prob, MaxTime=16*3600)
e_vec = best_candidate(bboptim_sol)
en_params = endogenous_parameters(e_vec)

# en_params = endogenous_parameters(rand(3))
# ~, en_params = fixed_point_iteration(ex_params, en_params; fixed_iter=200)

# en_params = endogenous_parameters([-2.245167110286566, 0.02819113961411014, -0.4900798062729814])

F_CREE, L_CREE, T_CREE, a, η, p, q = temporary_equilibrium(ex_params, en_params)
Σ_CREE = L_CREE*ex_params.Σ*L_CREE'
V_CREE = lyapd(F_CREE,Σ_CREE)
Cov_CREE = T_CREE*V_CREE*T_CREE'
########################

########################
# find the REE
########################
T_REE, F_REE, L_REE = REE(ex_params)
Σ_REE = L_REE*ex_params.Σ*L_REE'
V_REE = lyapd(F_REE,Σ_REE)
Cov_REE = T_REE*V_REE*T_REE'
########################

########################
# Compute the impulse response functions to productivity shocks.
T = 36
IRF_a_RE = zeros(n_x,T)
IRF_a = zeros(n_x,T)
for t=1:T
    IRF_a_RE[:,t] = T_REE*F_REE^(t-1)*L_REE[:,ϵ_dict["a"]]
    IRF_a[:,t] = T_CREE*F_CREE^(t-1)*L_CREE[:,ϵ_dict["a"]]
end
########################

########################
# Compute the impulse response functions to separation shocks.
IRF_s_RE = zeros(n_x,T)
IRF_s = zeros(n_x,T)
for t=1:T
    IRF_s_RE[:,t] = T_REE*F_REE^(t-1)*L_REE[:,ϵ_dict["s"]]
    IRF_s[:,t] = T_CREE*F_CREE^(t-1)*L_CREE[:,ϵ_dict["s"]]
end
########################

########################
# Plot the impulse response functions to productivity shocks.
plot(IRF_a[x_dict["a"],:])
plot!(IRF_a_RE[x_dict["a"],:])

plot(IRF_a[x_dict["p"],:])
plot!(IRF_a_RE[x_dict["p"],:])

plot(IRF_a[x_dict["w"],:])
plot!(IRF_a_RE[x_dict["w"],:])

plot(IRF_a[x_dict["v"],:])
plot!(IRF_a_RE[x_dict["v"],:])

plot(IRF_a[x_dict["u"],:])
plot!(IRF_a_RE[x_dict["u"],:])

plot(p[f_dict["a"]]*IRF_a[x_dict["a"],:]+p[f_dict["s"]]*IRF_a[x_dict["s"],:]+p[f_dict["u"]]*IRF_a[x_dict["u"],:])
########################

########################
# Plot the impulse response functions to separation shocks.
plot(IRF_s[x_dict["s"],:])
plot!(IRF_s_RE[x_dict["s"],:])

plot(IRF_s[x_dict["p"],:])
plot!(IRF_s_RE[x_dict["p"],:])

plot(IRF_s[x_dict["w"],:])
plot!(IRF_s_RE[x_dict["w"],:])

plot(IRF_s[x_dict["v"],:])
plot!(IRF_s_RE[x_dict["v"],:])

plot(IRF_s[x_dict["u"],:])
plot!(IRF_s_RE[x_dict["u"],:])

plot(p[f_dict["a"]]*IRF_s[x_dict["a"],:]+p[f_dict["s"]]*IRF_s[x_dict["s"],:]+p[f_dict["u"]]*IRF_s[x_dict["u"],:])
########################
