using LinearAlgebra
using Plots
# using Zygote
# using Roots
using LaTeXStrings
using Interpolations
using DataFrames
using CSV
using Printf



# global constants...

const vF::Float64 = 1e8 # cm/s
const r_LA::Float64 = 0.021 # vS/vF
# note: in this code I use r = vS/vF and s = vD/vS

const L::Float64 = 13 # μm
const W::Float64 = 3 # μm
const ρ::Float64 = 7.63e-8 # g cm^-2 

const n_flavors::Int32 = 4

const ergs_per_eV::Float64 = 1.602e-12

const D::Float64 = 30 * ergs_per_eV # 30 eV, converted to ergs

const e_statC::Float64 = -4.80326e-10 # statC
const e_C::Float64 = -1.60217663e-19 # coulombs

const ħ_eV_s::Float64 = 6.582e-16 # eV*s
const kB_eV_K::Float64 = 8.617e-5 # eV/K

const cutoff_angle::Float64 = 2*π


# the device is 3 μm wide and 13 μm long.
# the contacts are 1 μm wide and there is 1 μm spacing between them,
# so the center-to-center distance is 2 μm.
# the centers of the contacts are, I think, 1.5, 3.5, 5.5, 7.5, 9.5, 11.5 μm.




############################



# get γs from csv file

function load_γs_df(T,n,s)
    T_str = @sprintf("%1.2e",T)
    n_str = @sprintf("%1.2e",n)
    s_str = @sprintf("%1.2e",s)
    return DataFrame(CSV.File("./amp_rates_csv/set_1/" * replace("T="*T_str*"_n="*n_str*"_s="*s_str, "."=>"-") * ".csv"))
end

function γs_df_to_dict(γs_df)
    γs_dict = Dict{Tuple{Float64,Float64},Vector{Float64}}()
    for i=1:size(γs_df)[1]
        row = γs_df[i,:]
        qx,qy = row[1],row[2]
        γem,γabs = row[3],row[4]
        τinv = row[5]
        Γamp = row[6]
        γs_dict[qx,qy] = [γem,γabs,τinv,Γamp]
    end
    return γs_dict
end


# interpolate γs

function γ_interps(γs_dict)
    qx_dict_vec= union(sort([key[1] for key in keys(γs_dict)]))
    qx_dict_range = range(qx_dict_vec[1],qx_dict_vec[end],length=size(qx_dict_vec,1))
    qy_dict_vec= union(sort([key[1] for key in keys(γs_dict)]))
    qy_dict_range = range(qy_dict_vec[1],qy_dict_vec[end],length=size(qy_dict_vec,1))
    Γamp_vals = [γs_dict[qx,qy][4] for qx in qx_dict_range, qy in qy_dict_range]
    γem_vals = [γs_dict[qx,qy][1] for qx in qx_dict_range, qy in qy_dict_range]
    τinv_vals = [γs_dict[qx,qy][3] for qx in qx_dict_range, qy in qy_dict_range]
    Γamp_interp = linear_interpolation((qx_dict_range,qy_dict_range), Γamp_vals)
    γem_interp = linear_interpolation((qx_dict_range,qy_dict_range), γem_vals)
    τinv_interp = linear_interpolation((qx_dict_range,qy_dict_range), τinv_vals)
    return Γamp_interp, γem_interp, τinv_interp
end


# phonon population

function n_bose(β,qx,qy,r)
    minusE = -r*sqrt(qx^2 + qy^2)
    if minusE > 0
        return exp(β*minusE) / (-exp(β*minusE) + 1)
    else
        return 1 / (-1 + exp(-β*minusE))
    end
end

function n_phonons_right(qx,qy,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    Γamp = Γamp_interp(qx,qy)
    γem = γem_interp(qx,qy)
    τinv = τinv_interp(qx,qy)
    
    x_in_cm = x_in_μm * 1e-4

    n0 = n_bose(β,qx,qy,r)
    
    vq  = vF * r * abs(qx) / norm([qx,qy])
    exponential = exp(Γamp*x_in_cm/vq)
    term_1 = n0 * exponential
    term_2 = (τinv / Γamp) * (n0 + (γem / τinv)) * (exponential - 1)
    return term_1 + term_2
end

function n_phonons_left(qx,qy,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    Γamp = Γamp_interp(qx,qy)
    γem = γem_interp(qx,qy)
    τinv = τinv_interp(qx,qy)
    
    L_in_cm = L * 1e-4
    x_in_cm = x_in_μm * 1e-4

    n0 = n_bose(β,qx,qy,r)
    
    vq  = vF * r * abs(qx) / norm([qx,qy])
    exponential = exp(-Γamp * (x_in_cm - L_in_cm) / vq)
    term_1 = n0 * exponential
    term_2 = (τinv / Γamp) * (n0 + (γem / τinv)) * (exponential - 1)
    return term_1 + term_2
end

function n_phonons_vertical(qy,Γamp_interp,γem_interp,τinv_interp)
    qx = 0.0
    Γamp = Γamp_interp(qx,qy)
    γem = γem_interp(qx,qy)
    τinv = τinv_interp(qx,qy)
    n0 = n_bose(β,qx,qy,r)
    return (-n0*τinv - γem)/Γamp
end

function n_phonons(qx,qy,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    if qx > 0.0
        return n_phonons_right(qx,qy,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    elseif qx < 0.0
        return n_phonons_left(qx,qy,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    else # qx == 0.0
        return n_phonons_vertical(qy,Γamp_interp,γem_interp,τinv_interp)
    end
end




############################




# rhs of current balance equation


# drifting fermi-dirac distribution

function n_fermi_drifting(β,kx,ky,r,s)
    minusE = 1 + kx*r*s - sqrt(kx^2 + ky^2)
    if minusE > 0
        return exp(β*minusE) / (exp(β*minusE) + 1)
    else
        return 1 / (1 + exp(-β*minusE))
    end
end


# energy-conserving delta functions solutions & jacobian

kp_absorption_soln(k,θ,θp,r) = (k - k*(r^2)*cos(θ-θp) + sqrt((k^2)*(r^2)*(-1+cos(θ-θp))*(-2+(r^2)+(r^2)*cos(θ-θp)))) / (1-(r^2))
kp_emission_soln(k,θ,θp,r) = (k - k*(r^2)*cos(θ-θp) - sqrt((k^2)*(r^2)*(-1+cos(θ-θp))*(-2+(r^2)+(r^2)*cos(θ-θp)))) / (1-(r^2))

jacobian_for_absorption(k,θ,kp,θp,r) = -1 + r*(kp - k*cos(θ-θp)) / sqrt((k^2) + (kp^2) - 2*k*kp*cos(θ-θp))
jacobian_for_emission(k,θ,kp,θp,r) = -1 - r*(kp - k*cos(θ-θp)) / sqrt((k^2) + (kp^2) - 2*k*kp*cos(θ-θp))


function W_emission(k,θ,kp,θp,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    qx = k*cos(θ) - kp*cos(θp)
    qy = k*sin(θ) - kp*sin(θp)

    ε = 1e-12
    # if abs(qx) < ε && abs(qy) < ε
    if sqrt(qx^2 + qy^2) < ε
        return 0.0
    end

    n_q = 0.0
    if abs(θ-θp % π) > cutoff_angle && abs(θ-θp) < π - cutoff_angle
        n_q = n_bose(β,qx,qy,r)
    else
        n_q = n_phonons(qx,qy,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    end

    return sqrt(qx^2 + qy^2) * cos(abs(θ-θp)/2)^2 * (n_q + 1) / abs(jacobian_for_emission(k,θ,kp,θp,r))
end

function W_absorption(k,θ,kp,θp,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    qx = k*cos(θ) - kp*cos(θp)
    qy = k*sin(θ) - kp*sin(θp)

    ε = 1e-12
    # if abs(qx) < ε && abs(qy) < ε
    if sqrt(qx^2 + qy^2) < ε
        return 0.0
    end

    n_mq = 0.0
    if abs(θ-θp % π) > cutoff_angle && abs(θ-θp) < π - cutoff_angle
        n_mq = n_bose(β,-qx,-qy,r)
    else
        n_mq = n_phonons(-qx,-qy,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    end
    
    return sqrt(qx^2 + qy^2) * cos(abs(θ-θp)/2)^2 * n_mq / abs(jacobian_for_absorption(k,θ,kp,θp,r))
end

function F_emission(k,θ,kp,θp,β,r,s,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    kx, ky = k*cos(θ), k*sin(θ)
    kpx, kpy = kp*cos(θp), kp*sin(θp)

    fermi_factors = n_fermi_drifting(β,kx,ky,r,s) * (1 - n_fermi_drifting(β,kpx,kpy,r,s))
    
    # don't calculate W if the fermi factors aren't at least ε
    ε = 1e-12
    if fermi_factors < ε
        return 0.0
    end

    W_em = W_emission(k,θ,kp,θp,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)

    return fermi_factors * W_em
end

function F_absorption(k,θ,kp,θp,β,r,s,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    kx, ky = k*cos(θ), k*sin(θ)
    kpx, kpy = kp*cos(θp), kp*sin(θp)

    fermi_factors = n_fermi_drifting(β,kx,ky,r,s) * (1 - n_fermi_drifting(β,kpx,kpy,r,s))
    
    # don't calculate W if the fermi factors aren't at least ε
    ε = 1e-12
    if fermi_factors < ε
        return 0.0
    end

    W_abs = W_absorption(k,θ,kp,θp,β,r,x_in_μm,Γamp_interp,γem_interp,τinv_interp)

    return fermi_factors * W_abs
end

function integrate_F(β,r,s,x_in_μm,Γamp_interp,γem_interp,τinv_interp)
    kmin = 1 / (1 + r*s) - (12/β)
    kmax = 1 / (1 - r*s) + (12/β)
    k_steps = 200
    θ_steps = 200

    k_vals = range(kmin,kmax,length=k_steps)
    Δk = (kmax-kmin)/k_steps
    θ_vals = range(0,2*π,length=θ_steps+1)[1:end-1]
    Δθ = 2*π/θ_steps
    θp_vals = range(0,2*π,length=θ_steps+1)[1:end-1]
    Δθp = 2*π/θ_steps

    sum = 0.0
    for k in k_vals
        for θ in θ_vals
            for θp in θp_vals
                # note: cutoff angle is in F_emission and F_absorption

                kp_em = kp_emission_soln(k,θ,θp,r)
                kp_abs = kp_absorption_soln(k,θ,θp,r)

                emission_term = F_emission(k,θ,kp_em,θp,β,r,s,x_in_μm,Γamp_interp,γem_interp,τinv_interp) * k * kp_em * Δk * Δθ * Δθp
                absorption_term = F_absorption(k,θ,kp_abs,θp,β,r,s,x_in_μm,Γamp_interp,γem_interp,τinv_interp) * k * kp_abs * Δk * Δθ * Δθp

                sum += (emission_term + absorption_term) * (cos(θ) - cos(θp))
            end
        end
    end
    return sum
end



############################


# now the lhs of the current balance equation,
# see 13 july page of notes

ϵ(kx,ky,r,s) = sqrt(kx^2 + ky^2) - r*s*kx
function m∂f∂ϵ(β,kx,ky,r,s)
    exponential = exp(β * (ϵ(kx,ky,r,s) - 1))
    return β * exponential / (1 + exponential)^2
end
function ∂ϵ∂kx(kx,ky,r,s)
    return kx/sqrt(kx^2 + ky^2) - r*s
end

lhs_integrand(β,kx,ky,r,s) = kx/sqrt(kx^2 + ky^2) * m∂f∂ϵ(β,kx,ky,r,s) * ∂ϵ∂kx(kx,ky,r,s)

function integrate_lhs(β,r,s)
    kmin = max(1 / (1 + r*s) - (12/β), 0.0)
    kmax = 1 / (1 - r*s) + (12/β)

    k_steps = 2000
    θ_steps = 2000

    k_vals = range(kmin,kmax,length=k_steps)
    Δk = (kmax-kmin)/k_steps
    θ_vals = range(0,2*π,length=θ_steps+1)[1:end-1]
    Δθ = 2*π/θ_steps

    sum = 0.0
    for k in k_vals
        for θ in θ_vals
            kx, ky = k*cos(θ), k*sin(θ)
            sum += lhs_integrand(β,kx,ky,r,s) * k * Δk * Δθ
        end
    end
    return sum
end



############################



# now for the quotient to calculate the electric field
function compute_field(T,n,s,x)
    # note on units:
    # T is in K
    # n is in cm^-2
    # s is dimensionless
    # x is in μm
    # and this function returns the electric field in volts

    μ = carrier_density_to_chemical_potential(n)
    β = compute_β(T,μ)

    γs_df = load_γs_df(T,n,s)
    γs_dict = γs_df_to_dict(γs_df)
    Γamp_interp, γem_interp, τinv_interp = γ_interps(γs_dict)

    rhs = integrate_F(β,r_LA,s,x,Γamp_interp,γem_interp,τinv_interp)
    lhs = integrate_lhs(β,r_LA,s)

    # now for the other constants,
    vS = r_LA * vF
    kF = μ / (ħ_eV_s * vF) # so this is in cm^-1

    convert_to_SI = 29979.2458 # to convert to SI at the end... dyne/statC -> N/C (=V/m)

    constants = (n_flavors * D^2 * kF^3) / (4 * π * ρ * abs(e_statC) * vS * vF)
    
    return (rhs/lhs) * constants * convert_to_SI
end

function carrier_density_to_chemical_potential(n)
    # n should be in cm^-2
    # the chemical potential will be in eV
    return ħ_eV_s * vF * sqrt(π*n)
end

function compute_β(T,μ)
    # since we use β in units of μ^-1,
    # here we compute the value of β for a given temperature in K.
    # μ is in eV
    # T is in K
    # returns β in μ^-1.
    return μ / (kB_eV_K * T)
end




############################




# differential resistivity...

function drift_velocity_to_current(n,s)
    # n in cm^-2, vD in cm/s
    # returns current in amperes
    vD = r_LA*s*vF
    width_in_cm = W * 1e-4
    return n * width_in_cm * e_C * vD
end
    
function differential_resistivity(voltage_vals, current_vals)
    derivatives = zeros(Float64, size(current_vals,1))
    for i in eachindex(current_vals)
        if i==1 || i==size(current_vals,1)
            derivatives[i] = NaN
        else
            derivatives[i] = ((voltage_vals[i+1]-voltage_vals[i])/(current_vals[i+1]-current_vals[i]) + (voltage_vals[i]-voltage_vals[i-1])/(current_vals[i]-current_vals[i-1])) / 2.0
        end
    end
    return derivatives
end



############################




function write_fields_to_file(T,n,x,s_vals,field_vals)
    fields_df = DataFrame(s = s_vals, Fx = field_vals)
    T_str = @sprintf("%1.2e",T)
    n_str = @sprintf("%1.2e",n)
    x_str = @sprintf("%1.2e",x)
    CSV.write("./field_vals_csv/set_1/" * replace("T="*T_str*"_n="*n_str*"_x="*x_str, "."=>"-") * ".csv", fields_df)
end



begin
    T::Float64 = 2.0 # K
    n::Float64 = 1.4e12 # cm^-2

    x1::Float64 = 9.5 # μm
    x2::Float64 = 10.0
    x3::Float64 = 10.5
    x4::Float64 = 11.0
    x5::Float64 = 11.5
    # x6::Float64 = 12.0

    s_vals = 0.0:0.05:2.0

    field_vals_1 = Float64[]
    field_vals_2 = Float64[]
    field_vals_3 = Float64[]
    field_vals_4 = Float64[]
    field_vals_5 = Float64[]
    # field_vals_6 = Float64[]


    for s in s_vals
        println("s = ",s)

        @time field_1 = compute_field(T,n,s,x1)
        push!(field_vals_1, field_1)

        @time field_2 = compute_field(T,n,s,x2)
        push!(field_vals_2, field_2)

        @time field_3 = compute_field(T,n,s,x3)
        push!(field_vals_3, field_3)

        @time field_4 = compute_field(T,n,s,x4)
        push!(field_vals_4, field_4)

        @time field_5 = compute_field(T,n,s,x5)
        push!(field_vals_5, field_5)

        # @time field_6 = compute_field(T,n,s,x6)
        # push!(field_vals_6, field_6)
    end

    write_fields_to_file(T,n,x1,s_vals,field_vals_1)
    write_fields_to_file(T,n,x2,s_vals,field_vals_2)
    write_fields_to_file(T,n,x3,s_vals,field_vals_3)
    write_fields_to_file(T,n,x4,s_vals,field_vals_4)
    write_fields_to_file(T,n,x5,s_vals,field_vals_5)
    # write_fields_to_file(T,n,x6,s_vals,field_vals_6)
end