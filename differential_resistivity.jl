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



function load_field_vals(T,n,x)
    T_str = @sprintf("%1.2e",T)
    n_str = @sprintf("%1.2e",n)
    x_str = @sprintf("%1.2e",x)
    field_df = DataFrame(CSV.File("./field_vals_csv/set_1/" * replace("T="*T_str*"_n="*n_str*"_x="*x_str, "."=>"-") * ".csv"))
    return field_df
end


x_vals_μm = 9.5:0.5:11.5 # μm
x_vals_m = collect(x_vals_μm) .* 1e-6 # meters

T::Float64 = 2.0 # K
n::Float64 = 1.4e12 # cm^-2

field_df_x_9_5 = load_field_vals(T,n,x_vals_μm[1])
field_df_x_10_0 = load_field_vals(T,n,x_vals_μm[2])
field_df_x_10_5 = load_field_vals(T,n,x_vals_μm[3])
field_df_x_11_0 = load_field_vals(T,n,x_vals_μm[4])
field_df_x_11_5 = load_field_vals(T,n,x_vals_μm[5])
# field_df_x_12_0 = load_field_vals(T,n,x_vals[6])

fields_x_9_5 = field_df_x_9_5[:,2]
fields_x_10_0 = field_df_x_10_0[:,2]
fields_x_10_5 = field_df_x_10_5[:,2]
fields_x_11_0 = field_df_x_11_0[:,2]
fields_x_11_5 = field_df_x_11_5[:,2]
# fields_x_12_0 = field_df_x_12_0[:,2]

s_vals = field_df_x_9_5[:,1]

fields_matrix = hcat(fields_x_9_5,fields_x_10_0,fields_x_10_5,fields_x_11_0,fields_x_11_5)
# row corresponds to s, column corresponds to x


function trapezoid(x_vals, y_vals)
    N = size(x_vals,1)
    sum = 0.0
    for k=2:N
        area = (x_vals[k] - x_vals[k-1]) * (y_vals[k] + y_vals[k-1]) * 0.5
        sum += area
    end
    return sum
end


function voltages(x_vals, fields_matrix)
    N = size(fields_matrix,1)
    voltages = Float64[]

    for k=1:N
        fields = fields_matrix[k,:]
        push!(voltages, trapezoid(x_vals,fields))
    end

    return voltages 
end


Vs = voltages(x_vals_m, fields_matrix)


function drift_velocity_to_current(n,s)
    # n in cm^-2, vD in cm/s
    # returns current in amperes
    vD = r_LA*s*vF
    width_in_cm = W * 1e-4
    return n * width_in_cm * e_C * vD
end



currents = Float64[]
for s in s_vals
    push!(currents, drift_velocity_to_current(2e12, s))
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


dVdIs = differential_resistivity(Vs, currents)


p = scatter(s_vals[2:end], -dVdIs[2:end], label=:none, xaxis=:log, yaxis=:log)
p = plot!(xlabel = "vD/vS", ylabel="dV/dI (Ω)")


