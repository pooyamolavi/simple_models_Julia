###################################################
# Julia code for Section 5 of "Simple Models and Biased Forecasts," by Pooya Molavi (2022)
###################################################
# This file generates Figures 1, 2, and C.1 from the paper.
###################################################
# It calls "multi_variate_functions.jl" and "NK_functions.jl"
# and uses data from files "GDPC1.xlsx," "GDPPOT.xlsx," "GDPDEF.xlsx," and
# "FEDFUNDS.xlsx."
###################################################

import XLSX
using Statistics
using Parameters
using LinearAlgebra
using BlackBoxOptim
using JuMP
using Ipopt
using Plots

########################
# variables
########################
y_vars = ["x","π","i","n","μ"]
f_vars = ["x","π","i"]

n_y = size(y_vars, 1)
n_f = size(f_vars, 1)

y_dict = Dict()
for (idx, name_idx) in enumerate(y_vars)
   y_dict[name_idx] = idx
end

f_dict = Dict()
for (idx, name_idx) in enumerate(f_vars)
   f_dict[name_idx] = idx
end
########################

include("multi_variate_functions.jl")
include("NK_functions.jl")

########################
# read the data
########################
# date range: 1955Q1 - 2008Q4
T = 216
data = zeros(T,n_f)
########################

########################
# Output Gap
########################
excel_file = XLSX.readxlsx("GDPC1.xlsx")
excel_data = excel_file["FRED Graph"]
RealGDP = excel_data["B44:B259"]

excel_file = XLSX.readxlsx("GDPPOT.xlsx")
excel_data = excel_file["FRED Graph"]
PotentialGDP = excel_data["B36:B251"]

data[:,f_dict["x"]] = (RealGDP-PotentialGDP)./PotentialGDP*100
########################
# GDP Deflator
########################
excel_file = XLSX.readxlsx("GDPDEF.xlsx")
excel_data = excel_file["FRED Graph"]
data[:,f_dict["π"]] = excel_data["B40:B255"]
########################
# Fed Funds Rate
########################
excel_file = XLSX.readxlsx("FEDFUNDS.xlsx")
excel_data = excel_file["FRED Graph"]
data[:,f_dict["i"]] = excel_data["B14:B229"]
########################

# ########################
plot(data[:,f_dict["x"]])
plot!(data[:,f_dict["π"]])
plot!(data[:,f_dict["i"]])
# ########################

########################
# Find the pseudo-true model
params = parameters(data)
a,p,q,Γ₀,Γ₁,C = estimate_using_data(data; L=50)
########################

########################
# Compute the spectral radius of autocorrelations
T = 31
ρ = zeros(T)
ρ_bound = zeros(T)
for l=1:T
   eig_C = eigen(C[l,:,:])
   ~, λ_max_idx = findmax(abs.(eig_C.values))
   ρ[l] = eig_C.values[λ_max_idx]
   ρ_bound[l] = ρ[2]^(l-1)
end
########################

########################
plot(ρ)
plot!(ρ_bound)
########################

########################
# Compute the impulse response to an accommodative monetary policy shock

# serial correlation of interest rate
ρi = Γ₁[f_dict["i"],f_dict["i"]]/Γ₀[f_dict["i"],f_dict["i"]]

T = 51
IRF = zeros(n_f,T)
for t=1:T
    IRF[f_dict["i"],t] = ρi^(t-1)*(-1)
    IRF[f_dict["x"],t] = 1/(1-p[f_dict["x"]]*params.γx-p[f_dict["π"]]*(params.γπ+params.κ*params.γx))*(params.γx*p[f_dict["i"]]-params.σ*(1-params.γπ*p[f_dict["π"]]))*IRF[f_dict["i"],t]
    IRF[f_dict["π"],t] = 1/(1-p[f_dict["x"]]*params.γx-p[f_dict["π"]]*(params.γπ+params.κ*params.γx))*((params.γπ+params.κ*params.γx)*p[f_dict["i"]]-params.σ*(params.κ+params.γπ*p[f_dict["x"]]))*IRF[f_dict["i"],t]
end
########################

########################
plot(IRF[f_dict["i"],:])
plot(IRF[f_dict["x"],:])
plot(IRF[f_dict["π"],:])
########################

########################
# Compute the response of output and inflation to forward guidance

T=21
FG_outcomes = forward_guidance(params, -1*ones(T), T-1)
FG = zeros(n_f,T)
for t=0:T-1
    FG[f_dict["x"],t+1] = FG_outcomes[f_dict["x"],T-t]
    FG[f_dict["π"],t+1] = FG_outcomes[f_dict["π"],T-t]
end
########################

########################
plot(0:1:T-1,FG[f_dict["x"],:])
plot(0:1:T-1,FG[f_dict["π"],:])
########################
