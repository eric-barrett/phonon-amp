using LinearAlgebra
using Plots
using Zygote
using Roots
using LaTeXStrings
# using Interpolations
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



####################################################################



# drifting fermi-dirac distribution
function n_fermi_drifting(β,kx,ky,r,s)
    minusE = 1 + kx*r*s - sqrt(kx^2 + ky^2)
    if minusE > 0
        return exp(β*minusE) / (exp(β*minusE) + 1)
    else
        return 1 / (1 + exp(-β*minusE))
    end
end

# indicator functions
χ_em(β,kx,ky,qx,qy,r,s) = n_fermi_drifting(β,kx,ky,r,s) * (1 - n_fermi_drifting(β,kx-qx,ky-qy,r,s))
χ_abs(β,kx,ky,qx,qy,r,s) = n_fermi_drifting(β,kx-qx,ky-qy,r,s) * (1 - n_fermi_drifting(β,kx,ky,r,s))

# ampltiude
# the integration has nice numbers without any of the prefactors, so im just gonna put them all in at the end, after the integration
matrix_element_squared(kx,ky,qx,qy) = sqrt(qx^2 + qy^2) * (1 + (kx^2 + ky^2 - kx*qx - ky*qy)/(sqrt(kx^2 + ky^2) * sqrt((kx-qx)^2 + (ky-qy)^2)))

# for integrating over kx: good for |qy|>|qx|
function ky1(kx,qx,qy,r)
    discriminant = (qx^2 + qy^2)^2 * r^2 * (r^2 - 1) * (-4*kx^2 + 4*kx*qx + (qx^2 + qy^2)*(r^2 - 1))
    if discriminant < 0
        return NaN
    end
    return (2*kx*qx*qy + qy*(qx^2 + qy^2)*(r^2 - 1) + sqrt(discriminant)) / (-2*qy^2 + 2*(qx^2 + qy^2)*r^2)
end
function ky2(kx,qx,qy,r)
    discriminant = (qx^2 + qy^2)^2 * r^2 * (r^2 - 1) * (-4*kx^2 + 4*kx*qx + (qx^2 + qy^2)*(r^2 - 1))
    if discriminant < 0
        return NaN
    end
    return (2*kx*qx*qy + qy*(qx^2 + qy^2)*(r^2 - 1) - sqrt(discriminant)) / (-2*qy^2 + 2*(qx^2 + qy^2)*r^2)
end
jacobian_for_ky(kx,ky,qx,qy) = abs(1/(ky/sqrt(kx^2 + ky^2) - (ky-qy)/sqrt((kx-qx)^2 + (ky-qy)^2)))

# if qy > 0, use ky2
# if qy < 0, use ky1

function emission_integrand_over_kx(β,kx,qx,qy,r,s)
    ky = ky1(kx,qx,qy,r)
    if qy > 0
        ky = ky2(kx,qx,qy,r)
    end
    return matrix_element_squared(kx,ky,qx,qy) * jacobian_for_ky(kx,ky,qx,qy) * χ_em(β,kx,ky,qx,qy,r,s)
end
function absorption_integrand_over_kx(β,kx,qx,qy,r,s)
    ky = ky1(kx,qx,qy,r)
    if qy > 0
        ky = ky2(kx,qx,qy,r)
    end
    return matrix_element_squared(kx,ky,qx,qy) * jacobian_for_ky(kx,ky,qx,qy) * χ_abs(β,kx,ky,qx,qy,r,s)
end

# for integrating over ky: good for |qx|>|qy|
function kx1(ky,qx,qy,r)
    discriminant = (qx^2 + qy^2)^2 * r^2 * (r^2 - 1) * (-4*ky^2 + 4*ky*qy + (qx^2 + qy^2)*(r^2 - 1))
    if discriminant < 0
        return NaN
    end
    return ((qx^3)*(r^2 - 1) + qx*qy*(2*ky + qy*(r^2 - 1)) + sqrt(discriminant)) / (-2*qx^2 + 2*(qx^2 + qy^2)*r^2)
end
function kx2(ky,qx,qy,r)
    discriminant = (qx^2 + qy^2)^2 * r^2 * (r^2 - 1) * (-4*ky^2 + 4*ky*qy + (qx^2 + qy^2)*(r^2 - 1))
    if discriminant < 0
        return NaN
    end
    return ((qx^3)*(r^2 - 1) + qx*qy*(2*ky + qy*(r^2 - 1)) - sqrt(discriminant)) / (-2*qx^2 + 2*(qx^2 + qy^2)*r^2)
end
jacobian_for_kx(kx,ky,qx,qy) = abs(1/(kx/sqrt(kx^2 + ky^2) - (kx-qx)/sqrt((kx-qx)^2 + (ky-qy)^2)))

# if qx > 0, use kx2
# if qx < 0, use kx1

function emission_integrand_over_ky(β,ky,qx,qy,r,s)
    kx = kx1(ky,qx,qy,r)
    if qx > 0
        kx = kx2(ky,qx,qy,r)
    end
    return matrix_element_squared(kx,ky,qx,qy) * jacobian_for_kx(kx,ky,qx,qy) * χ_em(β,kx,ky,qx,qy,r,s)
end
function absorption_integrand_over_ky(β,ky,qx,qy,r,s)
    kx = kx1(ky,qx,qy,r)
    if qx > 0
        kx = kx2(ky,qx,qy,r)
    end
    return matrix_element_squared(kx,ky,qx,qy) * jacobian_for_kx(kx,ky,qx,qy) * χ_abs(β,kx,ky,qx,qy,r,s)
end



####################################################################



function trapezoid(f,lower,upper,goal_digits)
    # goal_digits should be the number of digits of accuracy desired
    goal_ε = 10.0^(-goal_digits)
    sum = 0.0
    ε = goal_ε * 1000
    step = (upper-lower) * 0.1
    while true
        grid = lower:step:upper
        samples = f.(grid)
        if samples != zeros(size(grid,1))
            break
        else
            step *= 0.2
        end
        if step < 1e-7
            break
        end
    end
    while ε > goal_ε
        new_sum = 0.0
        grid = lower:step:upper
        samples = f.(grid)
        N = size(grid,1)
        for k=2:N
            area = (grid[k] - grid[k-1]) * (samples[k] + samples[k-1]) * 0.5
            new_sum += area
        end
        ε = abs(new_sum - sum)
        sum = new_sum
        step *= 0.2
    end
    return sum, ε
end



####################################################################



function find_maxima(f,lower_bound,upper_bound,number_of_maxima,offset)
    trimmed_points_at_maxima = []
    grid = NaN
    step = (upper_bound-lower_bound) * 0.1
    while size(trimmed_points_at_maxima,1) < number_of_maxima
        if step < (upper_bound-lower_bound)*1e-5
            break
        end

        points_at_maxima = []
        points_near_maxima = []
        grid = lower_bound:step:upper_bound
        samples = f.(grid) .- offset
        for i=1:size(grid,1)
            if samples[i] > 0
                push!(points_near_maxima, grid[i])
            end
        end
    
        g = x -> f'(x)
        derivatives_near_maxima = g.(points_near_maxima)
        pairs_closest_to_maxima = []
        for i in 1:size(points_near_maxima,1)-1
            if derivatives_near_maxima[i] * derivatives_near_maxima[i+1] < 0
                push!(pairs_closest_to_maxima, [points_near_maxima[i],points_near_maxima[i+1]])
            end
        end

        points_at_maxima = []
        for pair in pairs_closest_to_maxima
            point_at_maximum = find_zero(g,pair)
            push!(points_at_maxima, point_at_maximum)
        end

        trimmed_points_at_maxima = []
        for x in points_at_maxima
            if f(x) >= offset
                push!(trimmed_points_at_maxima,x)
            end
        end

        step *= 0.1
    end
    return trimmed_points_at_maxima
end



####################################################################



# an important note here: even if the kx(ky) or ky(kx) curve lies in a single indicator-function-region,
# it still will have a zero in the middle and two maxima;
# it just will just be smooth on both sides rather than a sharp drop to a zero-valued region
# due to the fermi factors

function find_two_support_regions(f,maxima,lower_bound,upper_bound,ε)
    left_max, right_max = minimum(maxima), maximum(maxima)

    step = (right_max - left_max) * 0.01
    left_edges = []
    sample_point = left_max
    while f(sample_point) > ε
        if sample_point < lower_bound
            sample_point = lower_bound
            break
        end
        sample_point -= step
    end
    push!(left_edges, sample_point)
    sample_point = left_max
    while f(sample_point) > ε
        if sample_point > right_max
            sample_point = left_max
            step *= 0.1
        end
        sample_point += step
    end
    push!(left_edges, sample_point)

    step = (right_max - left_max) * 0.01
    right_edges = []
    sample_point = right_max
    while f(sample_point) > ε
        if sample_point < left_max
            sample_point = right_max
            step *= 0.1
        end
        sample_point -= step
    end
    push!(right_edges, sample_point)
    sample_point = right_max
    while f(sample_point) > ε
        if sample_point > upper_bound
            sample_point = upper_bound
            break
        end
        sample_point += step
    end
    push!(right_edges, sample_point)

    return [left_edges, right_edges]
end

function find_one_support_region(f,maxima,lower_bound,upper_bound,ε)
    left_max, right_max = minimum(maxima), maximum(maxima)
    step = (right_max - left_max) * 0.1
    if step == 0.0 # this happens when there is only one maximum instead of 2
        step = 0.1
        while f(left_max - step) < ε && f(left_max + step) < ε
            step *= 0.5
        end
    end

    left_point = left_max
    while f(left_point) > ε
        if left_point < lower_bound
            left_point = lower_bound
            break
        end
        left_point -= step
    end
    right_point = right_max
    while f(right_point) > ε
        if right_point > upper_bound
            right_point = upper_bound
            break
        end
        right_point += step
    end
    return [[left_point, right_point]]
end

function find_support(f,lower_bound,upper_bound,ε)
    number_of_maxima = 2
    maxima = find_maxima(f,lower_bound,upper_bound,number_of_maxima,ε)

    if maxima == []
        return []
    elseif size(maxima,1) == 1
        return find_one_support_region(f,maxima,lower_bound,upper_bound,ε)
    end

    left_max, right_max = minimum(maxima), maximum(maxima)
    # println(left_max, right_max)

    step = (right_max - left_max) * 0.1

    grid = left_max:step:right_max
    samples = f.(grid)
    
    number_of_regions = 1
    below_ε_count = 0
    for sample in samples
        if sample < ε
            below_ε_count += 1
        end
    end
    if below_ε_count >= 2
        number_of_regions = 2
    end

    if number_of_regions == 1
        # println("one region")
        return find_one_support_region(f,maxima,lower_bound,upper_bound,ε)
    else
        # println("two regions")
        return find_two_support_regions(f,maxima,lower_bound,upper_bound,ε)
    end
end



####################################################################



function integration_wrapper(f,low,hi,ε,goal_digits)
    integral = 0.
    absf = x -> abs(f(x))
    support = find_support(absf,low,hi,ε)
    for interval in support
        # println("integrating over\t", interval)
        sub_integral, _ = trapezoid(f,interval[1],interval[2],goal_digits)
        integral += sub_integral
    end
    return integral
end

# Δγ_integrand_over_ky(β,ky,qx,qy,r,s) = emission_integrand_over_ky(β,ky,qx,qy,r,s) - absorption_integrand_over_ky(β,ky,qx,qy,r,s)
# Δγ_integrand_over_kx(β,kx,qx,qy,r,s) = emission_integrand_over_kx(β,kx,qx,qy,r,s) - absorption_integrand_over_kx(β,kx,qx,qy,r,s)

function integrate_over_ky(integrand,kymin,kymax,ε,accuracy_digits)
    # integrand should be a function of only ky
    integral = integration_wrapper(integrand, kymin, kymax, ε, accuracy_digits)
    return integral
end

function integrate_over_kx(integrand,kxmin,kxmax,ε,accuracy_digits)
    # integrand should be a function of only kx
    integral = integration_wrapper(integrand, kxmin, kxmax, ε, accuracy_digits)
    return integral
end

function integrate(qx,qy,kx_integrand,ky_integrand,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
    if qx == 0.0 && qy == 0.0
        return 0.0
    end
    integral = 0.0
    if abs(qx) >= abs(qy)
        integral = integrate_over_ky(ky_integrand,kymin,kymax,ε,accuracy_digits)
    else
        integral = integrate_over_kx(kx_integrand,kxmin,kxmax,ε,accuracy_digits)
    end
end

# function Δγ_integrate(β,qx,qy,r,s,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
#     kx_integrand = kx -> Δγ_integrand_over_kx(β,kx,qx,qy,r,s)
#     ky_integrand = ky -> Δγ_integrand_over_ky(β,ky,qx,qy,r,s)
#     integral = integrate(qx,qy,kx_integrand,ky_integrand,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
#     return integral
# end

function γem_integrate(β,qx,qy,r,s,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
    kx_integrand = kx -> emission_integrand_over_kx(β,kx,qx,qy,r,s)
    ky_integrand = ky -> emission_integrand_over_ky(β,ky,qx,qy,r,s)
    integral = integrate(qx,qy,kx_integrand,ky_integrand,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
    return integral
end

function γabs_integrate(β,qx,qy,r,s,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
    kx_integrand = kx -> absorption_integrand_over_kx(β,kx,qx,qy,r,s)
    ky_integrand = ky -> absorption_integrand_over_ky(β,ky,qx,qy,r,s)
    integral = integrate(qx,qy,kx_integrand,ky_integrand,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
    return integral
end

# function make_Δγ_dictionary_unscaled(β,qvals,s)
#     r = 0.021
#     ε = 1e-8
#     accuracy_digits = 6
#     kymin,kymax = -3,3
#     kxmin,kxmax = -3,3

#     number_of_values = size(qvals,1)

#     Δγ_list = []
#     for i=1:number_of_values
#         q = qvals[i]
#         qx,qy = q[1],q[2]
#         Δγ = Δγ_integrate(β,qx,qy,r,s,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
#         push!(Δγ_list, Δγ)
#     end

#     return Dict(qvals[i]=>Δγ_list[i] for i=1:number_of_values)
# end

function make_γem_dictionary_unscaled(β,qvals,s)
    r = 0.021
    ε = 1e-8
    accuracy_digits = 6
    kymin,kymax = -3,3
    kxmin,kxmax = -3,3

    number_of_values = size(qvals,1)
    γem_list = []
    for i=1:number_of_values
        q = qvals[i]
        qx,qy = q[1],q[2]
        γem = γem_integrate(β,qx,qy,r,s,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
        push!(γem_list, γem)
    end

    return Dict(qvals[i]=>γem_list[i] for i=1:number_of_values)
end

function make_γabs_dictionary_unscaled(β,qvals,s)
    r = 0.021
    ε = 1e-8
    accuracy_digits = 6
    kymin,kymax = -3,3
    kxmin,kxmax = -3,3

    number_of_values = size(qvals,1)
    γabs_list = []
    for i=1:number_of_values
        q = qvals[i]
        qx,qy = q[1],q[2]
        γabs = γabs_integrate(β,qx,qy,r,s,kxmin,kxmax,kymin,kymax,ε,accuracy_digits)
        push!(γabs_list, γabs)
    end

    return Dict(qvals[i]=>γabs_list[i] for i=1:number_of_values)
end

function rescale_γs(γ_dict, μ_in_eV)
    # now multiply by all the constants
    scale = 1 # 1e-9 # if scale=1e-9, units are now GHz intsead of Hz

    kF = μ_in_eV / (ħ_eV_s*vF) # in cm^-1
    ħ_erg_s = ħ_eV_s * ergs_per_eV
    vS = r_LA * vF # 21 km/s, but in cm/s

    prefactors = n_flavors * 1/(4*π) * kF^2 * D^2/(ρ*vS) * 1/(ħ_erg_s*vF)

    newdict = Dict(collect(keys(γ_dict))[i] => scale*prefactors*collect(values(γ_dict))[i] for i=1:length(γ_dict))
    return newdict
end
    

# function calculate_Δγ_dict(γem_dict, γabs_dict)
#     Δγ_dict = Dict{Tuple{Float64,Float64},Float64}()
#     for key in keys(γem_dict)
#         Δγ_dict[key] = γem_dict[key] - γabs_dict[key]
#     end
#     return Δγ_dict
# end


function τinv(qx,qy)
    return 1e8
end

function make_τinv_dict(qvals)
    τinv_list = Float64[]
    for q in qvals
        τinv_q = τinv(q[1],q[2])
        push!(τinv_list, τinv_q)
    end
    return Dict(qvals[i]=>τinv_list[i] for i=1:size(qvals,1))
end




####################################################################




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




####################################################################




function γs_dict_to_df(γem_dict, γabs_dict, τinv_dict)
    df = DataFrame(qx=Float64[], qy=Float64[], γem=Float64[], γabs=Float64[], τinv=Float64[], Γamp=Float64[])
    for key in keys(γem_dict)
        Γamp = γem_dict[key] - γabs_dict[key] - τinv_dict[key]
        push!(df, (key[1], key[2], γem_dict[key], γabs_dict[key], τinv_dict[key], Γamp))
    end
    return df
end






qx_vals = -2.5:0.1:2.5
qy_vals = -2.5:0.1:2.5
qvals = vcat([(qx_vals[i],qy_vals[j]) for i=1:size(qx_vals,1), j=1:size(qy_vals,1)]...)

# β = 100 # in units of μ^-1
s_vals = 0.0:0.05:2.0 # vD / vS
# μ = 165e-3 # in units of eV

T = 1 # Kelvin
n = 2e12 # cm^-2

T_str = @sprintf("%1.2e",T)
n_str = @sprintf("%1.2e",n)

μ = carrier_density_to_chemical_potential(n)
β = compute_β(T,μ)

for s in s_vals
    println("s = ", s)
    @time begin
        γem_dict_unscaled = make_γem_dictionary_unscaled(β,qvals,s)
        γabs_dict_unscaled = make_γabs_dictionary_unscaled(β,qvals,s)

        γem_dict_scaled = rescale_γs(γem_dict_unscaled,μ)
        γabs_dict_scaled = rescale_γs(γabs_dict_unscaled,μ)

        τinv_dict = make_τinv_dict(qvals)

        γs_df = γs_dict_to_df(γem_dict_scaled, γabs_dict_scaled, τinv_dict)
    end

    s_str = @sprintf("%1.2e",s)
    CSV.write("./amp_rates_csv/test_3/" * replace("T="*T_str*"_n="*n_str*"_s="*s_str, "."=>"-") * ".csv", γs_df)
end