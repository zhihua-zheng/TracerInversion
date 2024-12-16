using Pkg; Pkg.instantiate()
using Printf
using Random
using Oceananigans
using Oceananigans.Units
using Oceananigans: location, TendencyCallsite
using Oceananigans.TurbulenceClosures: HorizontalFormulation

const Lx = 750kilometers # east-west extent
const Ly = 6750kilometers # north-south extent
const Lz = 1kilometers    # depth
const f = 1e-4
const n_tracers = 10
const restore_tracer = false
const n_per_set = ifelse(restore_tracer, (n_tracers - 2) / 2, NaN)
const tracer_restoring_rate = ifelse(restore_tracer, 1 / (10*24*3600), 0)
const N² = 1e-5 # [s⁻²] buoyancy frequency / stratification
const M² = 1e-7 # [s⁻²] horizontal buoyancy gradient
const Lf = 100kilometers # width of the region of the front
const ϵv = 1e-2 * M² / f * Lz # velocity noise amplitude
const ϵb = 1e-2 * M² * Lf # buoyancy noise amplitude
const save_fields_interval = 1hour
const casename = "c11_M010_Q000_W000_D000_St0_Ri10"
const outdir = "/glade/derecho/scratch/zhihuaz/TracerInversion/Output"
const ckpdir = replace(outdir, "Output" => "Restart") * "/Regular/" * casename
const Δx = Lx / 96
const κ₂z = 1e-4 # [m² s⁻¹] Laplacian vertical viscosity and diffusivity
const κ₄h = 1e-1 / day * Δx^4 # [m⁴ s⁻¹] biharmonic horizontal viscosity and diffusivity

grid = RectilinearGrid(GPU(),
                       size = (96, 864, 32),
                       x = (-Lx/2, Lx/2),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       halo = (5, 5, 5),
                       topology = (Bounded, Periodic, Bounded))

@inline function cmod(a, b)
    result = mod(a, b)
    return result == 0 ? b : result
end

@inline b_front(x, Lf) = Lf / 2 * tanh(2 * x / Lf)
@inline dbdx_front(x, Lf) = (sech(2 * x / Lf))^2

@inline tracer_alongfront(x, y, z, p) = sin((p.i+2) * π * y / p.Ly)#+ 2 * (z + p.Lz/2) / p.Lz) / 2
@inline tracer_gaussian(x, y, z, p) = exp(- (x / p.i / 2e4)^2 - ((z + p.Lz/2) / p.i / 100)^2) 
#@inline tracer_vertical(x, y, z, p) = tanh(2 * (z + p.i / 2 * Lz/4) / 100)
#@inline tracer_horizontal(x, y, z, p) = ifelse(p.i==1, exp(- (x / 2e5)^2 - ((z + p.Lz/2) / 100)^2),
#                                               sin((p.i - 1) / 2 * π * (x + p.Lx / 2) / p.Lx)/2 + (z + p.Lz/2) / p.Lz)
#@inline tracer_vertical(x, y, z, p) = tanh(2 * (z + p.i / 2 * Lz/4) / 100)
#@inline tracer_vertical(x, y, z, p) = ifelse(p.i==2, exp(- (x / 2e4)^2 - ((z + p.Lz/2) / 300)^2),
#                                             sin((p.i/2-1) * π * z/p.Lz)) 
@inline tracer_horizontal(x, y, z, p) = sin((p.i + 1)/2 * π * (x + p.Lx / 2) / p.Lx)
@inline tracer_vertical(x, y, z, p) = sin(p.i/2 * π * z / p.Lz)

@inline function tracer_IC(x, y, z, p)
    #p = merge(p, (; i=cmod(p.i, p.n_per_set)))
    return isodd(p.i) ? tracer_horizontal(x, y, z, p) : tracer_vertical(x, y, z, p)#cos(p.i/2 * π * (x+p.Lx/2)/p.Lx)
end

passive_tracers = [Symbol(:c,i) for i in 1:n_tracers]
for var in passive_tracers[1:end-2]
    @eval begin
        @inline function $(Symbol(var, "_forcing_func"))(x, y, z, t, $var, p)
            clambda = cld(p.i, p.n_per_set)
            lambda  = p.tracer_restoring_rate / clambda
            return -lambda * ($var - tracer_IC(x, y, z, p))
        end
    end
end

c_forcing = NamedTuple(var => Forcing(eval(Symbol(var, "_forcing_func")),
                                      parameters=(; tracer_restoring_rate, n_per_set, Lx, Ly, Lz, Lf, i),
                                      field_dependencies=var) for (i, var) in enumerate(passive_tracers[1:end-2]))
tracer_forcing = ifelse(restore_tracer, c_forcing, NamedTuple())

sgsh = ScalarBiharmonicDiffusivity(HorizontalFormulation(), ν=κ₄h, κ=κ₄h)
sgsz = VerticalScalarDiffusivity(ν=κ₂z, κ=κ₂z)
closure_sgs = (sgsh, sgsz)# SmagorinskyLilly()

model = HydrostaticFreeSurfaceModel(; grid,
                                    coriolis = FPlane(f=f),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = (passive_tracers..., :b, :bp),
                                    forcing = tracer_forcing,
                                    momentum_advection = WENO(order=9),
                                    tracer_advection = WENO(order=9))
                                    #closure = closure_sgs,
                                    #momentum_advection = Centered(order=6),
                                    #tracer_advection = Centered(order=6))

Random.seed!(45)
@inline bᵢ(x, y, z) = N² * (z + Lz) + M² * b_front(x, Lf) + ϵb * randn()
@inline vᵢ(x, y, z) = M² * (z + Lz) / f * dbdx_front(x, Lf) + ϵv * randn()
@inline uᵢ(x, y, z) = ϵv * randn()
@inline tracer_like_N²(x, y, z) = N² * (z + Lz) + ϵb * randn()
@inline tracer_like_M²(x, y, z) = M² * b_front(x, Lf) + ϵb * randn()
@inline function set_tracers(simulation)
    for i in 1:(n_tracers - 2)
        tracer_params = (; n_per_set, Lx, Ly, Lz, Lf, i)
        @info "Set tracer c$i distribution...."
        expr = Meta.parse("set!(simulation.model, c$i = (x, y, z) -> tracer_IC(x, y, z, $tracer_params))")
        eval(expr)
    end
    @info "Set tracer c$(n_tracers-1) distribution...."
    expr = Meta.parse("set!(simulation.model, c$(n_tracers-1) = (x, y, z) -> tracer_like_M²(x, y, z))")
    eval(expr)
    @info "Set tracer c$n_tracers distribution...."
    expr = Meta.parse("set!(simulation.model, c$n_tracers = (x, y, z) -> tracer_like_N²(x, y, z))")
    eval(expr)
    return nothing
end

@info "Initialize from checkpoint file...."
ckp_list  = split(read(`ls $ckpdir -1v`, String))
ckp_fpath = ckpdir * "/" * ckp_list[end]
set!(model, ckp_fpath)

#c_initial = NamedTuple(var => (x, y, z) -> tracer_IC(x, y, z, tracer_params) (eval(Symbol(var, "_forcing_func")),
#                                      parameters=(; tracer_restoring_rate, n_per_set, Lx, Ly, Lz, Lf, i),
#                                      field_dependencies=var) for (i, var) in enumerate(passive_tracers[1:end-2]))

for i in 1:(n_tracers - 2)
    tracer_params = (; n_per_set, Lx, Ly, Lz, Lf, i)
    @info "Set tracer c$i distribution...."
    expr = Meta.parse("set!(model, c$i = (x, y, z) -> tracer_IC(x, y, z, $tracer_params))")
    eval(expr)
end
@info "Set tracer c$(n_tracers-1) distribution...."
expr = Meta.parse("set!(model, c$(n_tracers-1) = (x, y, z) -> tracer_like_M²(x, y, z))")
eval(expr)
@info "Set tracer c$n_tracers distribution...."
expr = Meta.parse("set!(model, c$(n_tracers) = (x, y, z) -> tracer_like_N²(x, y, z))")
eval(expr)

simulation = Simulation(model, Δt=5minutes, stop_time=20days)
#simulation.callbacks[:set_tracers] = Callback(set_tracers, SpecifiedTimes(10days))
conjure_time_step_wizard!(simulation, IterationInterval(2), cfl=0.6, max_Δt=10minutes)

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


### Diagnostics ###
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators
using Oceananigans.AbstractOperations: @at, ∂x, ∂z
ccf_scratch = Field{Center, Center,  Face}(grid)
fff_scratch = Field{Face,   Face,    Face}(grid)
ccc_scratch = Field{Center, Center,  Center}(grid)
cnc_scratch = Field{Center, Nothing, Center}(grid)
cnf_scratch = Field{Center, Nothing, Face}(grid)
fnc_scratch = Field{Face,   Nothing, Center}(grid)
fnf_scratch = Field{Face,   Nothing, Face}(grid)
CellCenter  = (Center, Center, Center)
@inline Field_ccc(op) = Field((@at CellCenter op), data=ccc_scratch.data)
@inline function y_average(F; field=false)
    avg = Average(F, dims=2)
    if field
        LX, LY, LZ = location(F)
        if LX==Center && LZ==Center
            avg = Field(avg, data=cnc_scratch.data)

        elseif LX==Center && LZ==Face
            avg = Field(avg, data=cnf_scratch.data)

        elseif LX==Face && LZ==Center
            avg = Field(avg, data=fnc_scratch.data)

        elseif LX==Face && LZ==Face
            avg = Field(avg, data=fnf_scratch.data)
        end
    end
    return avg
end

u, v, w = model.velocities
b  = model.tracers.b
bp = model.tracers.bp
cs = Dict{Symbol, Any}(key => c for (key,c) in pairs(model.tracers) if key ∉ (:b, :bp))

u′  = u  - y_average(u,  field=true)
w′  = w  - y_average(w,  field=true)
b′  = b  - y_average(b,  field=true)
bp′ = bp - y_average(bp, field=true)
cs′ = Dict{Symbol, Any}(Symbol(key, "′") => c - y_average(c, field=true) for (key,c) in pairs(cs))

u′b′  = Field_ccc(u′*b′)
w′b′  = Field_ccc(w′*b′)
u′bp′ = Field_ccc(u′*bp′)
w′bp′ = Field_ccc(w′*bp′)
u′cs′ = Dict{Symbol, Any}(Symbol(:u′, key) => Field_ccc(u′*c′) for (key,c′) in pairs(cs′))
w′cs′ = Dict{Symbol, Any}(Symbol(:w′, key) => Field_ccc(w′*c′) for (key,c′) in pairs(cs′))

dbdz  = Field_ccc(∂z(b))
dbdx  = Field_ccc(∂x(b))
dbpdz = Field_ccc(∂z(bp))
dbpdx = Field_ccc(∂x(bp))
dcsdz = Dict{Symbol, Any}(Symbol(:d, key, :dz) => Field_ccc(∂z(c)) for (key,c) in pairs(cs))
dcsdx = Dict{Symbol, Any}(Symbol(:d, key, :dx) => Field_ccc(∂x(c)) for (key,c) in pairs(cs))

flow_state = Dict{Symbol, Any}(:u => u,
                               :v => v,
                               :w => w,
                               :b => b,
                               :bp => bp)
flow_derived = Dict{Symbol, Any}(:dbdz => dbdz,
                                 :dbdx => dbdx,
                                 :dbpdz => dbpdz,
                                 :dbpdx => dbpdx,
                                 :u′b′ => u′b′,
                                 :w′b′ => w′b′,
                                 :u′bp′ => u′bp′,
                                 :w′bp′ => w′bp′,
                                )
outputs_state = merge(flow_state, cs)
outputs_slice = merge(outputs_state, flow_derived,
                      u′cs′, w′cs′, dcsdx, dcsdz)
outputs_ymean = Dict{Symbol, Any}(Symbol(key, :_ym) => y_average(val, field=true) for (key,val) in pairs(outputs_slice))

global_attributes = Dict(:n_tracers => n_tracers, :n_per_set => n_per_set, :f => f, :M² => M², :N² => N²)
simulation.output_writers[:state] = NetCDFOutputWriter(model, outputs_state;
                                                       filename = casename * "_state.nc",
                                                       dir = outdir,
                                                       global_attributes = global_attributes,
                                                       schedule = TimeInterval(save_fields_interval),
                                                       overwrite_existing = true)
simulation.output_writers[:averages] = NetCDFOutputWriter(model, outputs_ymean;
                                                       filename = casename * "_averages.nc",
                                                       dir = outdir,
                                                       global_attributes = global_attributes,
                                                       schedule = TimeInterval(save_fields_interval),
                                                       overwrite_existing = true)
@info "Running the simulation..."
run(`nvidia-smi`)
run!(simulation)
@info "Simulation completed in " * prettytime(simulation.run_wall_time)
