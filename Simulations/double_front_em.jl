using Pkg; Pkg.instantiate()
using ArgParse
using Oceananigans
using Oceananigans.Units
using Oceananigans: location
using Oceananigans.BuoyancyFormulations: g_Earth
using Printf, Random


###########-------- COMMAND LINE ARGUMENTS ----------------#############
@info "Parse command line arguments..."
# Returns a dictionary of command line arguments
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "casename"
            help = "Name of simulation case"
            required = true
            arg_type = String

        "--spinup"
            help = "Flag for spinup run"
            action = :store_true

        "--init_tracer"
            help = "Flag for option to initialize tracer"
            action = :store_true

        "--nday"
            help = "Number of days the simulation runs"
            arg_type = Float64

        "--outdir"
            help = "Path of directory to save outputs under"
            default = "/glade/derecho/scratch/zhihuaz/TracerInversion/Output"
            arg_type = String
    end
    return parse_args(settings)
end

args = parse_command_line_arguments()
for (arg,val) in args
    @info "    $arg => $val"
end

casename = args["casename"]
spinup   = args["spinup"]
outdir   = args["outdir"]
init_tracer = args["init_tracer"]


###########-------- SIMULATION PARAMETERS ----------------#############
@info "Load in simulation parameters..."
include("simparams.jl")
groupname = ifelse(startswith(casename, 'd'), "DoubleFront",
                   startswith(casename, 'c') ? "Channel" : "FrontalZone")
pm = getproperty(SimParams(), Symbol(groupname))
pm = enrich_parameters(pm, casename, spinup, init_tracer)

stop_time = args["nday"] == nothing ? pm.stop_time : args["nday"] * 1days
basename  = SubString(casename, 1:19)
ckpdir    = replace(outdir, "Output" => "Restart") * "/" * pm.ckp_group * "/" * basename


###########-------- GRID SET UP ----------------#############
@inline h(k)  = (k - 1) / pm.Nz
@inline ζ₀(k) = 1 + (h(k) - 1) / pm.z_refinement
#@inline ζₘ(k) = (1 - tanh((h(k) - 0.2) / 0.3)) / 10
@inline ζ₁(k) = (1 - exp(-pm.z_stretching * h(k))) / (1 - exp(-pm.z_stretching))
#@inline z_faces(k) = pm.Lz * ((ζ₀(k) + ζₘ(k)) * ζ₁(k) - 1)
@inline z_faces(k) = pm.Lz * (ζ₀(k) * ζ₁(k) - 1)

zgrid = ifelse(pm.use_stretched_z, z_faces, (-pm.Lz, 0))
grid = RectilinearGrid(GPU(),
                       size = (pm.Nx, pm.Ny, pm.Nz),
                       x = (-pm.Lx/2, pm.Lx/2),
                       y = (0, pm.Ly),
                       z = zgrid,
                       #halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))


###########-------- INITIAL & BOUNDARY CONDITIONS -----------------#############
@inline tracer_like_nutrient(x, y, z) = (1 - tanh(3 * (z + pm.Hm))) / 2
@inline b_vertical(z) = pm.N₁² * (z + pm.Lz) + (pm.N₀² - pm.N₁²) * max(z + pm.Hm, 0)
@inline heaviside(z)  = ifelse(z < 0, zero(z), one(z))
@inline filament_horizontal(x) = (tanh( 2 * (x + pm.Lp) / pm.Lf) - tanh(2 * (x - pm.Lp) / pm.Lf)) * pm.Lf / 2
@inline filament_vertical(z)   = (tanh(12 * (z + pm.Hm) / pm.Hm) + 1) / 2
@inline filament_horizontal_gradient(x) = (sech(2 * (x + pm.Lp) / pm.Lf))^2 - (sech(2 * (x - pm.Lp) / pm.Lf))^2
@inline filament_vertical_integral(z)   = (z + pm.Hm / 12 * log(cosh(12 * (z + pm.Hm) / pm.Hm))) / 2
@inline Vg(x, z) = pm.M² / pm.f * filament_horizontal_gradient(x) * (filament_vertical_integral(z) - filament_vertical_integral(-pm.Lz))

Random.seed!(pm.em)
@inline bᵢ(x, y, z) = pm.ϵb * randn() * exp(z / 10) + b_vertical(z) + pm.M² * filament_horizontal(x) * filament_vertical(z)
@inline vᵢ(x, y, z) = pm.ϵv * randn() * exp(z / 10) + Vg(x, z)
@inline uᵢ(x, y, z) = pm.ϵv * randn() * exp(z / 10)

# same gradient magnitude for all tracers
@inline tracer_horizontal(x, y, z, p) = sin(π * ((p.i + 1) * x / p.Lx + 1/2))# / (p.i + 1)
@inline tracer_vertical(x, y, z, p) = sin(π * (p.i/2 * z / p.Hm))# + sin(π * ((p.i - 1) * x / p.Lx + 1/2))
@inline function tracer_IC(x, y, z, p)
    return isodd(p.i) ? tracer_horizontal(x, y, z, p) : tracer_vertical(x, y, z, p)
end

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-pm.τ₀ˣ/pm.ρ₀),
                                bottom = FluxBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-pm.τ₀ʸ/pm.ρ₀),
                                bottom = FluxBoundaryCondition(0))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(pm.B₀),
                                bottom = GradientBoundaryCondition(pm.N₁²))


###########-------- SPONGE LAYER -----------------#############
#@inline target_v(x, y, z, t) = Vg(x, z)
#@inline target_b(x, y, z, t) = b_vertical(z)
target_b = LinearTarget{:z}(intercept=pm.N₁²*pm.Lz, gradient=pm.N₁²)
bottom_mask = GaussianMask{:z}(center=-pm.Lz, width=pm.sponge_σ)
uvw_sponge = Relaxation(rate=pm.damping_rate, mask=bottom_mask, target=0)
#v_sponge  = Relaxation(rate=pm.damping_rate, mask=bottom_mask, target=target_v)
b_sponge   = Relaxation(rate=pm.damping_rate, mask=bottom_mask, target=target_b)
sponge_forcing = (u=uvw_sponge, v=uvw_sponge, w=uvw_sponge, b=b_sponge)


###########-------- DEFINE MODEL ---------------#############
closure_sgs = SmagorinskyLilly()#nothing
passive_tracers = ifelse(spinup, [Symbol(:c, pm.n_tracers)], [Symbol(:c, i) for i in 1:pm.n_tracers])

model = NonhydrostaticModel(; grid,
                            coriolis = FPlane(f=pm.f),
                            buoyancy = BuoyancyTracer(),
                            tracers  = (passive_tracers..., :b),
                            boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                            forcing = sponge_forcing,
                            closure = closure_sgs,
                            advection = WENO(order=5))

if spinup
    @info "Start from noise...."
    set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

    @info "Set tracer c$(pm.n_tracers) distribution...."
    eval(Meta.parse("set!(model, c$(pm.n_tracers) = (x, y, z) -> tracer_like_nutrient(x, y, z))"))
else
    @info "Restart from checkpoint file...."
    all_ckp   = split(read(`ls $ckpdir -1v`, String))
    ckp_list  = filter(s -> startswith(s, pm.pickup_prefix), all_ckp)
    ckp_fpath = ckpdir * "/" * ckp_list[pm.pickup_idx]
    set!(model, ckp_fpath)

    if init_tracer
        @info "Initialize tracer...."
        for i in 1:(pm.n_tracers - 1)
            tracer_params = (; pm.Lx, pm.Ly, pm.Lz, pm.Lf, pm.Lp, pm.Hm, i)
            @info "Set tracer c$i distribution...."
            eval(Meta.parse("set!(model, c$i = (x, y, z) -> tracer_IC(x, y, z, $tracer_params))"))
        end
    else
        @info "Tracer already initialized...."
    end
end


###########-------- DEFINE SIMULATION ---------------#############
simulation = Simulation(model, Δt=5seconds, stop_time=stop_time)
conjure_time_step_wizard!(simulation, IterationInterval(2), cfl=pm.cfl, max_Δt=pm.max_Δt)

wall_clock = Ref(time_ns())
@inline function print_progress(sim)
    u, v, w = model.velocities
    progress = 100 * (time(sim) / sim.stop_time)
    elapsed = (time_ns() - wall_clock[]) / 1e9

    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            progress, iteration(sim), prettytime(sim), prettytime(elapsed),
            maximum(u), maximum(v), maximum(w), prettytime(sim.Δt))

    wall_clock[] = time_ns()

    return nothing
end
add_callback!(simulation, print_progress, IterationInterval(50))


###########-------- DIAGNOSTICS --------------#############
include("diagnostics.jl")
@inline bool2int(x) = typeof(x) == Bool ? Int(x) : x
global_attributes = Dict(pairs(map(bool2int, pm)))

if pm.save_mean
    outputs_field, outputs_mean = get_outputs(model, pm.save_mean)
    ixm = round(Int, pm.Nx/2)
    ixr = round(Int, pm.Nx/4*3)
    slicers = (; xmid  = (ixm, :, :),
                 xfcr  = (ixr, : ,:),
                 south = (:, 1, :),
                 mlb   = (:, :, 28),
                 top   = (:, :, 58))
    for side in keys(slicers)
        indices = slicers[side]
        simulation.output_writers[side] = NetCDFOutputWriter(model, outputs_field;
                                                             filename = pm.outfile_prefix * "_$(side).nc",
                                                             dir = outdir,
                                                             global_attributes = global_attributes,
                                                             schedule = TimeInterval(pm.save_out_interval),
                                                             overwrite_existing = true,
                                                             indices = indices)
    end

    simulation.output_writers[:averages] = NetCDFOutputWriter(model, outputs_mean;
                                                              filename = pm.outfile_prefix * "_averages.nc",
                                                              dir = outdir,
                                                              global_attributes = global_attributes,
                                                              schedule = TimeInterval(pm.save_out_interval),
                                                              overwrite_existing = true)

else
    outputs_field = get_outputs(model, pm.save_mean)
    simulation.output_writers[:state] = NetCDFOutputWriter(model, outputs_field;
                                                           filename = pm.outfile_prefix * "_state.nc",
                                                           dir = outdir,
                                                           global_attributes = global_attributes,
                                                           schedule = TimeInterval(pm.save_out_interval),
                                                           overwrite_existing = true)
end

#simulation.output_writers[:averages] = NetCDFOutputWriter(model, outputs_mean;
#                                                         filename = fn_prefix * "_averages.nc",
#                                                         dir = outdir,
#                                                         global_attributes = global_attributes,
#                                                         schedule = TimeInterval(pm.save_out_interval),
#                                                         overwrite_existing = true)

if pm.save_ckp 
    @info "Add checkpointer..."
    if ispath(ckpdir)
        for f in readdir("$ckpdir")
            if startswith(f, pm.ckp_prefix)
                rm(joinpath(ckpdir, f), force=true)
            end
        end
    else
        mkpath(ckpdir)
    end
    simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(pm.save_ckp_interval),
                                                            dir=ckpdir, prefix=pm.ckp_prefix)
end

@info "Running the simulation..."
run(`nvidia-smi`)
run!(simulation)
@info "Simulation completed in " * prettytime(simulation.run_wall_time)
