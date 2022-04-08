############################
# Julia code for Section 6 of "Simple Models and Biased Forecasts," by Pooya Molavi (2022)
# The code is licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
############################
# This file contains functions used to compute the equilibirum of the RBC model
# as well is its response to TFP shocks.
############################
# This code uses SolveDSGE v0.2
########################

########################
# dictionaries
########################

########################
RE_dict = Dict()
for (idx, name_idx) in enumerate(RE_vars)
   RE_dict[name_idx] = idx
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
    ϕ::Float64 = 1.0
    σ::Float64 = 1.0
    δ::Float64 = 0.012
    α::Float64 = 0.30
    ρ::Float64 = 0.95
    Σ::Float64 = 1.0
end
########################

########################
mutable struct exogenous_parameters
    # deep parameters
    β::Float64
    ϕ::Float64
    σ::Float64
    δ::Float64
    α::Float64
    ρ::Float64
    Σ::Float64

    # steady state values
    r::Float64
    c_over_k::Float64
    y_over_i::Float64
    c_over_i::Float64

    # composite parameters
    χ::Float64
    ζ::Float64
    v::Array{Float64,1}
end
########################

########################
function exogenous_parameters(x::calibrated_parameters)
    # steady state values
    r = 1/x.β + x.δ - 1
    c_over_k = r/x.α - x.δ
    y_over_i = r/(x.α*x.δ)
    c_over_i = c_over_k/x.δ

    # composite parameters
    χ::Float64 = (1-x.β)/((1-x.α)*r/(x.α*x.σ*x.ϕ)+c_over_k)
    ζ::Float64 = (1-x.α)*(1+x.ϕ)*r/(x.α*x.ϕ)
    v = zeros(n_x)
    v[x_dict["r"]] = χ-x.β*x.σ
    v[x_dict["w"]] = χ*ζ

    return exogenous_parameters(x.β,x.ϕ,x.σ,x.δ,x.α,x.ρ,x.Σ,r,c_over_k,y_over_i,c_over_i,χ,ζ,v)
end
########################

########################
function exogenous_parameters()
    return exogenous_parameters(calibrated_parameters())
end
########################

########################
mutable struct endogenous_parameters
    γk::Float64
    γa::Float64
end

function endogenous_parameters(e_vec)
    endogenous_parameters(e_vec[1],e_vec[2])
end
########################

########################
# functions
########################

########################
# the rational-expectations equilibrium
function REE(x::exogenous_parameters; tol=1e-10)
    ########################
    # REE variables
    # model is      fₜ = Φfₜ + Afₜ₋₁ + BEₜfₜ₊₁ + Cϵₜ    where   ϵₜ ~ N(0,Σ)
    # solution is   fₜ = Ffₜ₋₁ + Lϵₜ                  where   ϵₜ ~ N(0,Σ)
    ########################
    Φ = zeros(n_RE,n_RE)
    A = zeros(n_RE,n_RE)
    B = zeros(n_RE,n_RE)
    C = zeros(n_RE,n_ϵ)
    Σ = zeros(n_ϵ, n_ϵ)
    ########################
    # y
    Φ[RE_dict["o"],RE_dict["a"]] = 1.0
    Φ[RE_dict["o"],RE_dict["k"]] = x.α
    Φ[RE_dict["o"],RE_dict["n"]] = 1.0-x.α
    ########################
    # w
    Φ[RE_dict["w"],RE_dict["a"]] = 1.0
    Φ[RE_dict["w"],RE_dict["k"]] = x.α
    Φ[RE_dict["w"],RE_dict["n"]] = -x.α
    ########################
    # r
    Φ[RE_dict["r"],RE_dict["a"]] = x.r
    Φ[RE_dict["r"],RE_dict["n"]] = (1.0-x.α)*x.r
    Φ[RE_dict["r"],RE_dict["k"]] = -(1.0-x.α)*x.r
    ########################
    # n
    Φ[RE_dict["n"],RE_dict["w"]] = 1.0/x.ϕ
    Φ[RE_dict["n"],RE_dict["c"]] = -1.0/(x.σ*x.ϕ)
    ########################
    # k
    A[RE_dict["k"],RE_dict["k"]] = 1.0-x.δ
    A[RE_dict["k"],RE_dict["i"]] = x.δ
    ########################
    # i
    Φ[RE_dict["i"],RE_dict["o"]] = x.y_over_i
    Φ[RE_dict["i"],RE_dict["c"]] = -x.c_over_i
    ########################
    # a
    A[RE_dict["a"],RE_dict["a"]] = x.ρ
    C[RE_dict["a"],ϵ_dict["ϵ"]] = 1.0
    ########################
    # c
    B[RE_dict["c"],RE_dict["c"]] = 1.0
    B[RE_dict["c"],RE_dict["r"]] = -x.σ*x.β
    ########################
    # variance of the shock
    Σ[ϵ_dict["ϵ"],ϵ_dict["ϵ"]] = x.Σ
    ########################
    cutoff = 1.0
    model = Binder_Pesaran_Form(I-Φ, A, B, C, Σ)
    soln = solve_re(model, cutoff, tol)

    if soln.soln_type == "determinate"
        F = soln.p
        L = soln.k
    else
        println(soln.soln_type)
    end
    return soln.soln_type, F, L
end
########################

########################
# temporary equilibrium
function temporary_equilibrium(x::exogenous_parameters, e::endogenous_parameters)
    ########################
    # CREE
    # model is      fₜ = Ffₜ + Lϵₜ
    ########################
    # the T matrix
    T = zeros(n_x,n_f)
    Φωω = zeros(n_ω, n_ω)
    Φωf = zeros(n_ω, n_f)
    ########################
    # y
    Φωf[ω_dict["o"],f_dict["a"]] = 1.0
    Φωf[ω_dict["o"],f_dict["k"]] = x.α
    Φωω[ω_dict["o"],ω_dict["n"]] = 1.0-x.α
    ########################
    # w
    Φωf[ω_dict["w"],f_dict["a"]] = 1.0
    Φωf[ω_dict["w"],f_dict["k"]] = x.α
    Φωω[ω_dict["w"],ω_dict["n"]] = -x.α
    ########################
    # r
    Φωf[ω_dict["r"],f_dict["a"]] = x.r
    Φωω[ω_dict["r"],ω_dict["n"]] = (1.0-x.α)*x.r
    Φωf[ω_dict["r"],f_dict["k"]] = -(1.0-x.α)*x.r
    ########################
    # n
    Φωω[ω_dict["n"],ω_dict["w"]] = 1.0/x.ϕ
    Φωω[ω_dict["n"],ω_dict["c"]] = -1.0/(x.σ*x.ϕ)
    ########################
    # i
    Φωω[ω_dict["i"],ω_dict["o"]] = x.y_over_i
    Φωω[ω_dict["i"],ω_dict["c"]] = -x.c_over_i
    ########################
    # c
    Φωf[ω_dict["c"],f_dict["k"]] = x.χ/x.β+e.γk
    Φωω[ω_dict["c"],ω_dict["r"]] = x.χ
    Φωω[ω_dict["c"],ω_dict["w"]] = x.χ*x.ζ
    Φωf[ω_dict["c"],f_dict["a"]] = e.γa
    ########################
    Φωf_inv = inv(I-Φωω)*Φωf
    for f_idx ∈ f_vars
        T[x_dict[f_idx],f_dict[f_idx]]=1.0
    end
    for row_idx ∈ ω_vars
        for col_idx ∈ f_vars
            T[x_dict[row_idx],f_dict[col_idx]] = Φωf_inv[ω_dict[row_idx],f_dict[col_idx]]
        end
    end
    ########################
    # the ψ vectors
    ψk = T[x_dict["i"],f_dict["k"]]
    ψa = T[x_dict["i"],f_dict["a"]]
    ########################
    F = zeros(n_f, n_f)
    L = zeros(n_f, n_ϵ)
    ########################
    # k
    F[f_dict["k"],f_dict["k"]] = 1-x.δ+x.δ*ψk
    F[f_dict["k"],f_dict["a"]] = x.δ*ψa
    ########################
    # a
    F[f_dict["a"],f_dict["a"]] = x.ρ
    L[f_dict["a"],ϵ_dict["ϵ"]] = 1
    ########################
    return F, L, T
end
########################

########################
function fixed_point_violation(x::exogenous_parameters, e::endogenous_parameters; instability_penalty=1e50, tol=1e-12, max_time=1e6)
    F, L, T = temporary_equilibrium(x, e)
    if maximum(abs.(eigvals(F)))>1
        return instability_penalty*exp(maximum(abs.(eigvals(F)))-1)
    else
        Σ = L*x.Σ*L'

        a,η,p,q,~ = best_1_state_full_info_model(F,Σ; tol=tol,max_time=max_time)
        γk = a*x.β/(1-a*x.β)*x.v'*T*q*p[f_dict["k"]]
        γa = a*x.β/(1-a*x.β)*x.v'*T*q*p[f_dict["a"]]

        return sqrt(η^2+(γk-e.γk)^2+(γa-e.γa)^2)
    end
end
########################

########################
function fixed_point_iteration(x::exogenous_parameters, e_init::endogenous_parameters; new_weight=0.01, fixed_iter=1000, fixed_tol=1e-10, inner_loop_tol=1e-12, inner_loop_max_cpu_time=1.0)
    e_new = e_init
    for i=1:fixed_iter
        e_old = e_new
        F, L, T = temporary_equilibrium(x, e_new)
        if maximum(abs.(eigvals(F)))>1
            return false, e_new
        else
            Σ = L*x.Σ*L'
            a,η,p,q,~ = best_1_state_full_info_model(F,Σ; tol=inner_loop_tol,max_time=inner_loop_max_cpu_time)
            γk = a*x.β/(1-a*x.β)*x.v'*T*q*p[f_dict["k"]]
            γa = a*x.β/(1-a*x.β)*x.v'*T*q*p[f_dict["a"]]
            change = sqrt((γk-e_old.γk)^2+(γa-e_old.γa)^2)
            println("i=", i, ";  change=", change)
            if change<fixed_tol
                return true, e_new
            end

            e_new = endogenous_parameters(γk*new_weight+e_old.γk*(1-new_weight),γa*new_weight+e_old.γa*(1-new_weight))
        end
    end
    return false, e_new
end
########################

########################
# compute the CREE given exogenous parameters and shocks
function CREE(x::exogenous_parameters; e_vec_lb=[-10.0,-10.0], e_vec_ub=[10.0,10.0], max_time=5*60, algorithm=:adaptive_de_rand_1_bin_radiuslimited, instability_penalty=1e50, tol=1e-10, inner_loop_tol=1e-12, inner_loop_max_cpu_time=1.0)
    function fp_violation(e_vec)
        e = endogenous_parameters(e_vec)
        return fixed_point_violation(x,e; instability_penalty=instability_penalty,tol=inner_loop_tol,max_time=inner_loop_max_cpu_time)
    end

    e_vec_bnd=Tuple{Float64,Float64}[]
    for i=1:2
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
