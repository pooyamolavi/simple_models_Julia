###################################################
# Julia code for Section 6 of "Simple Models and Biased Forecasts," by Pooya Molavi (2022)
# The code is licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
###################################################
# This file generates Figure 3 from the paper.
###################################################
# It calls "multi_variate_functions.jl" and "RBC_functions.jl."
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
RE_vars = ["o","a","k","n","w","r","c","i"]
f_vars = ["k","a"]
ω_vars = ["o","n","w","r","c","i"]
x_vars = [f_vars; ω_vars]
ϵ_vars = ["ϵ"]

n_RE = size(RE_vars,1)
n_f = size(f_vars,1)
n_ω = size(ω_vars,1)
n_x = size(x_vars,1)
n_ϵ = size(ϵ_vars, 1)
########################

include("multi_variate_functions.jl")
include("RBC_functions.jl")

ex_params = exogenous_parameters()

########################
# find the CREE
########################
e_vec, bboptim_prob = CREE(ex_params; max_time=10*3600)
bboptim_sol = bboptimize(bboptim_prob, MaxTime=10*3600)
e_vec = best_candidate(bboptim_sol)
en_params = endogenous_parameters(e_vec)
# fixed_point_iteration(ex_params, en_params)
# en_params = endogenous_parameters([0.7969401733810653 0.04447178234769557])
F_CREE, L_CREE, T_CREE = temporary_equilibrium(ex_params, en_params)
a,η,p,q,~ = best_1_state_full_info_model(F_CREE,L_CREE*ex_params.Σ*L_CREE')
Σ_CREE = L_CREE*ex_params.Σ*L_CREE'
V_CREE = lyapd(F_CREE,Σ_CREE)
Cov_CREE = T_CREE*V_CREE*T_CREE'
########################

########################
# find the REE
~, F_REE, L_REE = REE(ex_params)
Σ_REE = L_REE*ex_params.Σ*L_REE'
Cov_REE = lyapd(F_REE,Σ_REE)
########################

########################
# Compute the impulse response functions.
T = 51
IRF_RE = zeros(n_RE,T)
IRF = zeros(n_x,T)
for t=1:T
    IRF_RE[:,t] = F_REE^(t-1)*L_REE
    IRF[:,t] = T_CREE*F_CREE^(t-1)*L_CREE
end
########################

########################
plot(IRF[x_dict["a"],:])
plot!(IRF_RE[RE_dict["a"],:])

plot(IRF[x_dict["o"],:])
plot!(IRF_RE[RE_dict["o"],:])

plot(IRF[x_dict["n"],:])
plot!(IRF_RE[RE_dict["n"],:])

plot(IRF[x_dict["i"],:])
plot!(IRF_RE[RE_dict["i"],:])

plot(IRF[x_dict["r"],:])
plot!(IRF_RE[RE_dict["r"],:])

plot(IRF[x_dict["w"],:])
plot!(IRF_RE[RE_dict["w"],:])

plot(IRF[x_dict["k"],:])
plot!(IRF_RE[RE_dict["k"],:])

plot(IRF[x_dict["c"],:])
plot!(IRF_RE[RE_dict["c"],:])

plot(p[f_dict["a"]]*IRF[x_dict["a"],:]+p[f_dict["k"]]*IRF[x_dict["k"],:])
########################
