###################################################
# Julia code for "Simple Models and Biased Forecasts," by Pooya Molavi (2022)
# The code is licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
###################################################
# This file contains functions that compute pseudo-true models. It is called by
# the main file in each of the applications.
###################################################

###################################################
# The optimization algorithm can be any of the following
# algorithm ∈ {:Ipopt, :adaptive_de_rand_1_bin_radiuslimited, :dxnes, ...}
###################################################

###################################################
########## 1-state Model + General DGP ###########
###################################################

###################################################
# the Kullback-Leibler divergence of 1-state state-space model (a,η) from DGP with auto-covariance matrices Γ
function H_1(Γ,n,a,η,sum_upper_bound)
    C = fill(0.0, n, n)
    for l = 1:sum_upper_bound
        C = C + a^l*η^(l-1)*pinv(Γ[1,:,:])*(Γ[l+1,:,:]+transpose(Γ[l+1,:,:]))/2
    end
    eig_C = eigen(C)
    λ_max, ~ = findmax(real.(eig_C.values))

    return -a^2*(1-η)^2/(1-a^2*η^2)+2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*λ_max
end
###################################################

###################################################
# the derivative of the Kullback-Leibler divergence of 1-state state-space model (a,η) with auto-covariance matrices Γ
function DH_1(Γ,n,a,η,sum_upper_bound)
    C = fill(0.0, n, n)
    dC_da = fill(0.0, n, n)
    dC_dη = fill(0.0, n, n)
    for l = 1:sum_upper_bound
        dC_da = dC_da + l*a^(l-1)*η^(l-1)*(Γ[l+1,:,:]+transpose(Γ[l+1,:,:]))/2
        dC_dη = dC_dη + (l-1)*a^l*η^(l-2)*(Γ[l+1,:,:]+transpose(Γ[l+1,:,:]))/2
        C = C + a^l*η^(l-1)*pinv(Γ[1,:,:])*(Γ[l+1,:,:]+transpose(Γ[l+1,:,:]))/2
    end
    eig_C = eigen(C)
    λ_max, λ_max_idx = findmax(real.(eig_C.values))
    p_max = real.(eig_C.vectors[:,λ_max_idx])
    p_max = p_max/sqrt(p_max'*Γ[1,:,:]*p_max)

    D = fill(0.0, 2)
    D[1] = -2*a*(1-η)^2/(1-a^2*η^2)^2
    D[1] = D[1] - 4*a*η*(1-η)^2/(1-a^2*η^2)^2*λ_max
    D[1] = D[1] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_da*p_max

    D[2] = 2*a^2*(1-η)*(1-a^2*η)/(1-a^2*η^2)^2
    D[2] = D[2] - 2*(1+a^4*η^2+a^2*(1-4*η+η^2))/(1-a^2*η^2)^2*λ_max
    D[2] = D[2] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_dη*p_max

    return D
end
###################################################

###################################################
function best_1_state_model_Ipopt(Γ;n=size(Γ,2),x_lb=[-1.0;0.0],x_ub=[1.0;1.0],sum_upper_bound=size(Γ,1)-1,print_level=5, tol=1e-8, max_time=1e6)
    # the KL divergence
    function H(a,η)
        return H_1(Γ,n,a,η,sum_upper_bound)
    end

    function ∇H(D,a,η)
        C = fill(0.0, n, n)
        dC_da = fill(0.0, n, n)
        dC_dη = fill(0.0, n, n)
        for l = 1:sum_upper_bound
            dC_da = dC_da + l*a^(l-1)*η^(l-1)*(Γ[l+1,:,:]+transpose(Γ[l+1,:,:]))/2
            dC_dη = dC_dη + (l-1)*a^l*η^(l-2)*(Γ[l+1,:,:]+transpose(Γ[l+1,:,:]))/2
            C = C + a^l*η^(l-1)*pinv(Γ[1,:,:])*(Γ[l+1,:,:]+transpose(Γ[l+1,:,:]))/2
        end
        eig_C = eigen(C)
        λ_max, λ_max_idx = findmax(real.(eig_C.values))
        p_max = real.(eig_C.vectors[:,λ_max_idx])
        p_max = p_max/sqrt(p_max'*Γ[1,:,:]*p_max)

        D[1] = -2*a*(1-η)^2/(1-a^2*η^2)^2
        D[1] = D[1] - 4*a*η*(1-η)^2/(1-a^2*η^2)^2*λ_max
        D[1] = D[1] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_da*p_max

        D[2] = 2*a^2*(1-η)*(1-a^2*η)/(1-a^2*η^2)^2
        D[2] = D[2] - 2*(1+a^4*η^2+a^2*(1-4*η+η^2))/(1-a^2*η^2)^2*λ_max
        D[2] = D[2] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_dη*p_max
    end
    # optimize
    model = Model(Ipopt.Optimizer)
    set_optimizer_attributes(model, "print_level" => print_level)
    set_optimizer_attribute(model, "max_cpu_time", max_time)
    set_optimizer_attribute(model, "tol", tol)
    @variable(model, x_lb[1] <= a_opt <= x_ub[1])
    @variable(model, x_lb[2] <= η_opt <= x_ub[2])
    set_start_value(a_opt, .5)
    set_start_value(η_opt, .5)
    register(model, :H, 2, H, ∇H)
    @NLobjective(model, Max, H(a_opt,η_opt))
    optimize!(model)

    # return optimization results
    a = JuMP.value.(a_opt)
    η = JuMP.value.(η_opt)
    return a,η,model
end
###################################################

###################################################
function best_1_state_model_bboptim(Γ;n=size(Γ,2),x_lb=[-1.0;0],x_ub=[1.0;1.0],sum_upper_bound=size(Γ,1)-1,max_time=10*60,algorithm=:adaptive_de_rand_1_bin_radiuslimited)
    # the KL divergence
    function H(X)
        a = X[1]
        η = X[2]
        return -H_1(Γ,n,a,η,sum_upper_bound)
    end

    x_bnd=Tuple{Float64,Float64}[]
    for i=1:2
        new_bnd = (x_lb[i], x_ub[i])
        x_bnd = vcat(x_bnd, new_bnd)
    end

    # optimize
    bboptim_prob = bbsetup(H;  Method=algorithm, SearchRange=x_bnd)
    bboptim_sol = bboptimize(bboptim_prob, MaxTime=max_time)
    X = best_candidate(bboptim_sol)

    # return optimization results
    a = X[1]
    η = X[2]
    return a,η,bboptim_prob,bboptim_sol
end
##################################################

###################################################
function best_1_state_model(Γ;n=size(Γ,2),x_lb=[-1.0;0],x_ub=[1.0;1.0],sum_upper_bound=size(Γ,1)-1,tol=1e-8,max_time=1e6,algorithm=:Ipopt,print_level=2)
    # optimize
    if algorithm == :Ipopt
        return best_1_state_model_Ipopt(Γ;n=n,x_lb=x_lb,x_ub=x_ub,sum_upper_bound=sum_upper_bound,print_level=print_level, tol=tol, max_time=max_time)
    elseif algorithm == :BlackBoxOptim
        return best_1_state_model_bboptim(Γ;n=n,x_lb=x_lb,x_ub=x_ub,sum_upper_bound=sum_upper_bound,max_time=max_time,algorithm=:adaptive_de_rand_1_bin_radiuslimited)
    else
        return best_1_state_model_bboptim(Γ;n=n,x_lb=x_lb,x_ub=x_ub,sum_upper_bound=sum_upper_bound,max_time=max_time,algorithm=algorithm)
    end
end
###################################################

###################################################
########## 1-state Model + VAR DGP ###############
###################################################

###################################################
# the Kullback-Leibler divergence of 1-state state-space model (a,η) from ACF Ξ implied by auto-covariance matrices Γ
# where Γₗ = H'FˡVH
function H_1(F,H,V,Γ₀_pinv,m,n,a,η)
    II = Matrix{Float64}(I, m, m)
    C = Γ₀_pinv*H'*(a*F*inv(II-a*η*F)*V+a*V*F'*inv(II-a*η*F'))*H/2
    eig_C = eigen(C)
    λ_max, ~ = findmax(real.(eig_C.values))

    return -a^2*(1-η)^2/(1-a^2*η^2)+2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*λ_max
end
###################################################

###################################################
# the derivative of the Kullback-Leibler divergence of 1-state state-space model (a,η) from ACF Ξ implied by auto-covariance matrices Γ
# where Γₗ = H'FˡVH
function DH_1(F,H,V,Γ₀_pinv,m,n,a,η)
    II = Matrix{Float64}(I, m, m)
    C = Γ₀_pinv*H'*(a*F*inv(II-a*η*F)*V+a*V*F'*inv(II-a*η*F'))*H/2
    dC_da = H'*(F*(inv(II-a*η*F))^2*V+V*F'*(inv(II-a*η*F'))^2)*H/2
    dC_dη = H'*(a^2*F^2*(inv(II-a*η*F))^2*V+a^2*V*(F')^2*(inv(II-a*η*F'))^2)*H/2
    eig_C = eigen(C)
    λ_max, λ_max_idx = findmax(real.(eig_C.values))
    p_max = real.(eig_C.vectors[:,λ_max_idx])
    p_max = p_max/sqrt(p_max'*H'*V*H*p_max)

    D = fill(0.0, 2)
    D[1] = -2*a*(1-η)^2/(1-a^2*η^2)^2
    D[1] = D[1] - 4*a*η*(1-η)^2/(1-a^2*η^2)^2*λ_max
    D[1] = D[1] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_da*p_max

    D[2] = 2*a^2*(1-η)*(1-a^2*η)/(1-a^2*η^2)^2
    D[2] = D[2] - 2*(1+a^4*η^2+a^2*(1-4*η+η^2))/(1-a^2*η^2)^2*λ_max
    D[2] = D[2] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_dη*p_max

    return D
end
###################################################

###################################################
function best_1_state_model_Ipopt(F,H,V,Γ₀_pinv;m=size(F,1),n=size(H,2),x_lb=[-1.0;0.0],x_ub=[1.0;1.0],print_level=5,tol=1e-8, max_time=1e6)
    # the KL divergence
    function H_obj(a,η)
        return H_1(F,H,V,Γ₀_pinv,m,n,a,η)
    end

    function ∇H_obj(D,a,η)
        II = Matrix{Float64}(I, m, m)
        C = Γ₀_pinv*H'*(a*F*inv(II-a*η*F)*V+a*V*F'*inv(II-a*η*F'))*H/2
        dC_da = H'*(F*(inv(II-a*η*F))^2*V+V*F'*(inv(II-a*η*F'))^2)*H/2
        dC_dη = H'*(a^2*F^2*(inv(II-a*η*F))^2*V+a^2*V*(F')^2*(inv(II-a*η*F'))^2)*H/2
        eig_C = eigen(C)
        λ_max, λ_max_idx = findmax(real.(eig_C.values))
        p_max = real.(eig_C.vectors[:,λ_max_idx])
        p_max = p_max/sqrt(p_max'*H'*V*H*p_max)

        D[1] = -2*a*(1-η)^2/(1-a^2*η^2)^2
        D[1] = D[1] - 4*a*η*(1-η)^2/(1-a^2*η^2)^2*λ_max
        D[1] = D[1] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_da*p_max

        D[2] = 2*a^2*(1-η)*(1-a^2*η)/(1-a^2*η^2)^2
        D[2] = D[2] - 2*(1+a^4*η^2+a^2*(1-4*η+η^2))/(1-a^2*η^2)^2*λ_max
        D[2] = D[2] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_dη*p_max
    end
    # optimize
    model = Model(Ipopt.Optimizer)
    set_optimizer_attributes(model, "print_level" => print_level)
    set_optimizer_attribute(model, "max_cpu_time", max_time)
    set_optimizer_attribute(model, "tol", tol)
    @variable(model, x_lb[1] <= a_opt <= x_ub[1])
    @variable(model, x_lb[2] <= η_opt <= x_ub[2])
    # set_start_value(a_opt, .5)
    # set_start_value(η_opt, .5)
    register(model, :H_obj, 2, H_obj, ∇H_obj)
    @NLobjective(model, Max, H_obj(a_opt,η_opt))
    optimize!(model)

    # return optimization results
    a = JuMP.value.(a_opt)
    η = JuMP.value.(η_opt)
    return a,η,model
end
###################################################

###################################################
function best_1_state_model_bboptim(F,H,V,Γ₀_pinv;m=size(F,1),n=size(H,2),x_lb=[-1.0;0],x_ub=[1.0;1.0],max_time=10*60,algorithm=:adaptive_de_rand_1_bin_radiuslimited)
    # the KL divergence
    function H_obj(X)
        a = X[1]
        η = X[2]
        return -H_1(F,H,V,Γ₀_pinv,m,n,a,η)
    end

    x_bnd=Tuple{Float64,Float64}[]
    for i=1:2
        new_bnd = (x_lb[i], x_ub[i])
        x_bnd = vcat(x_bnd, new_bnd)
    end

    # optimize
    bboptim_prob = bbsetup(H_obj;  Method=algorithm, SearchRange=x_bnd)
    bboptim_sol = bboptimize(bboptim_prob, MaxTime=max_time)
    X = best_candidate(bboptim_sol)

    # return optimization results
    a = X[1]
    η = X[2]
    return a,η,bboptim_prob
end
##################################################

###################################################
# the pseudo-true 1-state model given the true process
# fₜ = F*fₜ₋₁+ϵ               where     ϵ~N(0,Σ)
# yₜ = H'*fₜ
###################################################
function best_1_state_model(F,H,Σ;m=size(F,1),n=size(H,2),x_lb=[-1.0;0],x_ub=[1.0;1.0],tol=1e-8,max_time=1e6,algorithm=:Ipopt,print_level=2,split_a_range=true)
    # optimize
    V = lyapd(F,Σ)
    Γ₀_pinv = pinv(H'*V*H)
    if algorithm == :Ipopt && split_a_range == false
        a,η,model = best_1_state_model_Ipopt(F,H,V,Γ₀_pinv;m=m,n=n,x_lb=x_lb,x_ub=x_ub,print_level=print_level,tol=tol,max_time=max_time)
    elseif algorithm == :Ipopt && split_a_range == true
        a_pos,η_pos,model_pos = best_1_state_model_Ipopt(F,H,V,Γ₀_pinv;m=m,n=n,x_lb=[0.0,x_lb[2]],x_ub=x_ub,print_level=print_level,tol=tol,max_time=max_time)
        a_neg,η_neg,model_neg = best_1_state_model_Ipopt(F,H,V,Γ₀_pinv;m=m,n=n,x_lb=x_lb,x_ub=[0.0, x_ub[2]],print_level=print_level,tol=tol,max_time=max_time)
        if H_1(F,H,V,Γ₀_pinv,m,n,a_pos,η_pos) >= H_1(F,H,V,Γ₀_pinv,m,n,a_neg,η_neg)
            a,η,model =  a_pos,η_pos,model_pos
        else
            a,η,model = a_neg,η_neg,model_neg
        end
    elseif algorithm == :BlackBoxOptim
        a,η,model = best_1_state_model_bboptim(F,H,V,Γ₀_pinv;m=m,n=n,x_lb=x_lb,x_ub=x_ub,max_time=max_time,algorithm=:adaptive_de_rand_1_bin_radiuslimited)
    else
        a,η,model = best_1_state_model_bboptim(F,H,V,Γ₀_pinv;m=m,n=n,x_lb=x_lb,x_ub=x_ub,sum_upper_bound=sum_upper_bound,max_time=max_time,algorithm=algorithm)
    end

    # compute the attention and sensitivity vectors
    II = Matrix{Float64}(I, m, m)
    C = Γ₀_pinv*H'*(a*F*inv(II-a*η*F)*V+a*V*F'*inv(II-a*η*F'))*H/2
    eig_C = eigen(C)
    ~, λ_max_idx = findmax(real.(eig_C.values))
    p = real.(eig_C.vectors[:,λ_max_idx])
    p = p/sqrt(p'*H'*V*H*p)
    q = H'*V*H*p

    return a,η,p,q,model
end
###################################################

###################################################
########## 1-state Full Information ##############
###################################################

###################################################
# the Kullback-Leibler divergence of 1-state state-space model (a,η) from ACF Ξ implied by auto-covariance matrices Γ
# where Γₗ = FˡV
function H_1(F,V,Γ₀_pinv,m,n,a,η)
    II = Matrix{Float64}(I, m, m)
    C = Γ₀_pinv*(a*F*inv(II-a*η*F)*V+a*V*F'*inv(II-a*η*F'))/2
    eig_C = eigen(C)
    λ_max, ~ = findmax(real.(eig_C.values))

    return -a^2*(1-η)^2/(1-a^2*η^2)+2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*λ_max
end
###################################################

###################################################
# the derivative of the Kullback-Leibler divergence of 1-state state-space model (a,η) from ACF Ξ implied by auto-covariance matrices Γ
# where Γₗ = H'FˡVH
function DH_1(F,V,Γ₀_pinv,m,n,a,η)
    II = Matrix{Float64}(I, m, m)
    C = Γ₀_pinv*(a*F*inv(II-a*η*F)*V+a*V*F'*inv(II-a*η*F'))/2
    dC_da = (F*(inv(II-a*η*F))^2*V+V*F'*(inv(II-a*η*F'))^2)/2
    dC_dη = (a^2*F^2*(inv(II-a*η*F))^2*V+a^2*V*(F')^2*(inv(II-a*η*F'))^2)/2
    eig_C = eigen(C)
    λ_max, λ_max_idx = findmax(real.(eig_C.values))
    p_max = real.(eig_C.vectors[:,λ_max_idx])
    p_max = p_max/sqrt(p_max'*V*p_max)

    D = fill(0.0, 2)
    D[1] = -2*a*(1-η)^2/(1-a^2*η^2)^2
    D[1] = D[1] - 4*a*η*(1-η)^2/(1-a^2*η^2)^2*λ_max
    D[1] = D[1] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_da*p_max

    D[2] = 2*a^2*(1-η)*(1-a^2*η)/(1-a^2*η^2)^2
    D[2] = D[2] - 2*(1+a^4*η^2+a^2*(1-4*η+η^2))/(1-a^2*η^2)^2*λ_max
    D[2] = D[2] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_dη*p_max

    return D
end
###################################################

###################################################
function best_1_state_model_Ipopt(F,V,Γ₀_pinv;m=size(F,1),n=size(H,2),x_lb=[-1.0;0.0],x_ub=[1.0;1.0],print_level=5,tol=1e-8,max_time=1e6)
    # the KL divergence
    function H_obj(a,η)
        return H_1(F,V,Γ₀_pinv,m,n,a,η)
    end

    function ∇H_obj(D,a,η)
        II = Matrix{Float64}(I, m, m)
        C = Γ₀_pinv*(a*F*inv(II-a*η*F)*V+a*V*F'*inv(II-a*η*F'))/2
        dC_da = (F*(inv(II-a*η*F))^2*V+V*F'*(inv(II-a*η*F'))^2)/2
        dC_dη = (a^2*F^2*(inv(II-a*η*F))^2*V+a^2*V*(F')^2*(inv(II-a*η*F'))^2)/2
        eig_C = eigen(C)
        λ_max, λ_max_idx = findmax(real.(eig_C.values))
        p_max = real.(eig_C.vectors[:,λ_max_idx])
        p_max = p_max/sqrt(p_max'*V*p_max)

        D[1] = -2*a*(1-η)^2/(1-a^2*η^2)^2
        D[1] = D[1] - 4*a*η*(1-η)^2/(1-a^2*η^2)^2*λ_max
        D[1] = D[1] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_da*p_max

        D[2] = 2*a^2*(1-η)*(1-a^2*η)/(1-a^2*η^2)^2
        D[2] = D[2] - 2*(1+a^4*η^2+a^2*(1-4*η+η^2))/(1-a^2*η^2)^2*λ_max
        D[2] = D[2] + 2*(1-η)*(1-a^2*η)/(1-a^2*η^2)*p_max'*dC_dη*p_max
    end
    # optimize
    model = Model(Ipopt.Optimizer)
    set_optimizer_attributes(model, "print_level" => print_level)
    set_optimizer_attribute(model, "max_cpu_time", max_time)
    set_optimizer_attribute(model, "tol", tol)
    @variable(model, x_lb[1] <= a_opt <= x_ub[1])
    @variable(model, x_lb[2] <= η_opt <= x_ub[2])
    # set_start_value(a_opt, .5)
    # set_start_value(η_opt, .5)
    register(model, :H_obj, 2, H_obj, ∇H_obj)
    @NLobjective(model, Max, H_obj(a_opt,η_opt))
    optimize!(model)

    # return optimization results
    a = JuMP.value.(a_opt)
    η = JuMP.value.(η_opt)
    return a,η,model
end
###################################################

###################################################
function best_1_state_model_bboptim(F,V,Γ₀_pinv;m=size(F,1),n=size(H,2),x_lb=[-1.0;0],x_ub=[1.0;1.0],max_time=10*60,algorithm=:adaptive_de_rand_1_bin_radiuslimited)
    # the KL divergence
    function H_obj(X)
        a = X[1]
        η = X[2]
        return -H_1(F,V,Γ₀_pinv,m,n,a,η)
    end

    x_bnd=Tuple{Float64,Float64}[]
    for i=1:2
        new_bnd = (x_lb[i], x_ub[i])
        x_bnd = vcat(x_bnd, new_bnd)
    end

    # optimize
    bboptim_prob = bbsetup(H_obj;  Method=algorithm, SearchRange=x_bnd)
    bboptim_sol = bboptimize(bboptim_prob, MaxTime=max_time)
    X = best_candidate(bboptim_sol)

    # return optimization results
    a = X[1]
    η = X[2]
    return a,η,bboptim_prob
end
##################################################

###################################################
# the pseudo-true 1-state model given the true process
# fₜ = F*fₜ₋₁+ϵ               where     ϵ~N(0,Σ)
# yₜ = fₜ
###################################################
function best_1_state_full_info_model(F,Σ;m=size(F,1),x_lb=[-1.0;0],x_ub=[1.0;1.0],tol=1e-8,max_time=1e6,algorithm=:Ipopt,print_level=2,split_a_range=true)
    # optimize
    V = lyapd(F,Σ)
    Γ₀_pinv = pinv(V)
    if algorithm == :Ipopt && split_a_range == false
        a,η,model = best_1_state_model_Ipopt(F,V,Γ₀_pinv;m=m,n=m,x_lb=x_lb,x_ub=x_ub,print_level=print_level,tol=tol,max_time=max_time)
    elseif algorithm == :Ipopt && split_a_range == true
        a_pos,η_pos,model_pos = best_1_state_model_Ipopt(F,V,Γ₀_pinv;m=m,n=m,x_lb=[0.0,x_lb[2]],x_ub=x_ub,print_level=print_level,tol=tol,max_time=max_time)
        a_neg,η_neg,model_neg = best_1_state_model_Ipopt(F,V,Γ₀_pinv;m=m,n=m,x_lb=x_lb,x_ub=[0.0, x_ub[2]],print_level=print_level,tol=tol,max_time=max_time)
        if H_1(F,V,Γ₀_pinv,m,m,a_pos,η_pos) >= H_1(F,V,Γ₀_pinv,m,m,a_neg,η_neg)
            a,η,model =  a_pos,η_pos,model_pos
        else
            a,η,model = a_neg,η_neg,model_neg
        end
    elseif algorithm == :BlackBoxOptim
        a,η,model = best_1_state_model_bboptim(F,V,Γ₀_pinv;m=m,n=m,x_lb=x_lb,x_ub=x_ub,max_time=max_time,algorithm=:adaptive_de_rand_1_bin_radiuslimited)
    else
        a,η,model = best_1_state_model_bboptim(F,V,Γ₀_pinv;m=m,n=m,x_lb=x_lb,x_ub=x_ub,sum_upper_bound=sum_upper_bound,max_time=max_time,algorithm=algorithm)
    end

    # compute the attention and sensitivity vectors
    II = Matrix{Float64}(I, m, m)
    C = Γ₀_pinv*(a*F*inv(II-a*η*F)*V+a*V*F'*inv(II-a*η*F'))/2
    eig_C = eigen(C)
    ~, λ_max_idx = findmax(real.(eig_C.values))
    p = real.(eig_C.vectors[:,λ_max_idx])
    p = p/sqrt(p'*V*p)
    q = V*p

    return a,η,p,q,model
end
###################################################
