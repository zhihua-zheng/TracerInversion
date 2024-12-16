using Pkg; Pkg.instantiate()
using Printf
using Random
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: HorizontalFormulation

const Lx = 750kilometers # east-west extent
const Ly = 6750kilometers # north-south extent
const Lz = 1kilometers    # depth
const f = 1e-4
const N² = 1e-5 # [s⁻²] buoyancy frequency / stratification
const M² = 1e-7 # [s⁻²] horizontal buoyancy gradient
const Lf = 100kilometers # width of the region of the front
const ϵv = 1e-2 * M² / f * Lz # velocity noise amplitude
const ϵb = 1e-2 * M² * Lf # buoyancy noise amplitude
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

@inline b_front(x, Lf) = Lf / 2 * tanh(2 * x / Lf)
@inline dbdx_front(x, Lf) = (sech(2 * x / Lf))^2
sgsh = ScalarBiharmonicDiffusivity(HorizontalFormulation(), ν=κ₄h, κ=κ₄h)
sgsz = VerticalScalarDiffusivity(ν=κ₂z, κ=κ₂z)
closure_sgs = (sgsh, sgsz)# SmagorinskyLilly()

model = HydrostaticFreeSurfaceModel(; grid,
                                    coriolis = FPlane(f=f),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = (:b, :bp),
                                    momentum_advection = WENO(order=9),
                                    tracer_advection = WENO(order=9))
                                    #closure = closure_sgs,
                                    #momentum_advection = Centered(order=6),
                                    #tracer_advection = Centered(order=6))

Random.seed!(45)
@inline bᵢ(x, y, z) = N² * (z + Lz) + M² * b_front(x, Lf) + ϵb * randn()
@inline vᵢ(x, y, z) = M² * (z + Lz) / f * dbdx_front(x, Lf) + ϵv * randn()
@inline uᵢ(x, y, z) = ϵv * randn()

set!(model, u=uᵢ, v=vᵢ, b=bᵢ, bp=bᵢ)
simulation = Simulation(model, Δt=5minutes, stop_time=10days)
conjure_time_step_wizard!(simulation, IterationInterval(2), cfl=0.7, max_Δt=20minutes)

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

const save_checkpoint_interval = 2days
const casename = "c11_M010_Q000_W000_D000_St0_Ri10"
const outdir = "/glade/derecho/scratch/zhihuaz/TracerInversion/Restart/Regular"

@info "Add checkpointer..."
ckpdir = outdir * "/" * casename
ispath(ckpdir) ? rm(ckpdir, recursive=true, force=true) : mkdir(ckpdir)
simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(save_checkpoint_interval),
                                                        dir=ckpdir, prefix="checkpoint")

@info "Running the simulation..."
run(`nvidia-smi`)
run!(simulation)
@info "Simulation completed in " * prettytime(simulation.run_wall_time)
