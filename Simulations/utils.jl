using NCDatasets: defVar
using Oceananigans: location
using Oceananigans.Fields: AbstractField
using Oceananigans.Grids: Center, Face
using Oceananigans.OutputWriters: fetch_and_convert_output, drop_output_dims,
                                  netcdf_spatial_dimensions, output_indices
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation, @at, ∂x, ∂y, ∂z
using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w#, div_Uc
using Oceananigans.BoundaryConditions: getbc
using Oceananigans.TurbulenceClosures: viscous_flux_ux, viscous_flux_uy, viscous_flux_uz,
                                       viscous_flux_vx, viscous_flux_vy, viscous_flux_vz,
                                       viscous_flux_wx, viscous_flux_wy, viscous_flux_wz,
                                       diffusive_flux_x, diffusive_flux_y, diffusive_flux_z,
                                       ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ
using Oceanostics: perturbation_fields
using Oceanostics.TKEBudgetTerms: ψf

using Oceananigans: prognostic_fields, HydrostaticFreeSurfaceModel
using Oceananigans.Biogeochemistry: biogeochemical_auxiliary_fields


@inline uˢ_transport(z) = pm.Dˢ * pm.Uˢ * (1 - exp(z / pm.Dˢ)) * cosd(pm.θ₀)
@inline vˢ_transport(z) = pm.Dˢ * pm.Uˢ * (1 - exp(z / pm.Dˢ)) * sind(pm.θ₀)
@inline uˢ(z) = pm.Uˢ * exp(z / pm.Dˢ) * cosd(pm.θ₀)
@inline vˢ(z) = pm.Uˢ * exp(z / pm.Dˢ) * sind(pm.θ₀)
@inline B̅(x)  = - pm.M² * x
@inline V̅(z)  = - pm.M² / pm.f * (z + pm.Lz/2)

@inline function tracer_bkg(x, z, p)
    p = merge(p, (; i=cmod(p.i, p.n_per_set)))
    return isodd(p.i) ? sin((p.i+1)/2 * π * (x+p.Lx/2)/p.Lx) : cos(p.i/2 * π * z/p.Lz)#cos(p.i/2 * π * (x+p.Lx/2)/p.Lx)
end

@inline function tracer_bkg_grad(x, z, p)
    p = merge(p, (; i=cmod(p.i, p.n_per_set)))
    return isodd(p.i) ? (p.i+1)/2*π/p.Lx * cos((p.i+1)/2 * π * (x+p.Lx/2)/p.Lx) : (-p.i/2*π/p.Lz) * sin(p.i/2 * π * z/p.Lz)#(-p.i/2*π/p.Lx) * sin(p.i/2 * π * (x+p.Lx/2)/p.Lx)
end

@inline function get_time_invariant_fields(grid::RectilinearGrid)
    Tus = Field{Nothing, Nothing, Face}(grid)
    Tvs = Field{Nothing, Nothing, Face}(grid)
    set!(Tus, uˢ_transport)
    set!(Tvs, vˢ_transport)
    usla = Field(-∂z(Tus))
    vsla = Field(-∂z(Tvs))
    compute!(usla)
    compute!(vsla)

    us = Field{Nothing, Nothing, Face}(grid)
    vs = Field{Nothing, Nothing, Face}(grid)
    set!(us, uˢ)
    set!(vs, vˢ)
    dusdzla = Field(∂z(us))
    dvsdzla = Field(∂z(vs))
    compute!(dusdzla)
    compute!(dvsdzla)

    Bbak = Field{Center, Nothing, Nothing}(grid)
    Vbak = Field{Nothing, Nothing, Center}(grid)
    set!(Bbak, B̅)
    set!(Vbak, V̅)
    compute!(Bbak)
    compute!(Vbak)

    basic_bak = Dict{Symbol, Any}(:us => usla, :vs => vsla, :dusdz => dusdzla, :dvsdz => dvsdzla,
                                  :Bbak => Bbak, :Vbak => Vbak)

    #Cbak  = Dict{Symbol, Any}(Symbol(:Cbak, i) => Field{Center, Nothing, Center}(grid) for i in 1:(pm.n_tracers - 2))
    #CbakG = Dict{Symbol, Any}(Symbol(:CbakG,i) => Field{Center, Nothing, Center}(grid) for i in 1:(pm.n_tracers - 2))
    #for i in 1:(pm.n_tracers - 2)
    #    tracer_params = (; pm.n_per_set, pm.Lx, pm.Lz, i)
    #    set!(Cbak[Symbol(:Cbak, i)], (x, z) -> tracer_bkg(x, z, tracer_params))
    #    compute!(Cbak[Symbol(:Cbak, i)])

    #    set!(CbakG[Symbol(:CbakG, i)], (x, z) -> tracer_bkg_grad(x, z, tracer_params))
    #    compute!(CbakG[Symbol(:CbakG, i)])
    #end
    #return merge(basic_bak, Cbak, CbakG)
    return basic_bak
end


@inline define_time_invariant_variable!(dataset, output::AbstractField, name, array_type, deflatelevel, output_attributes) =
    defVar(dataset, name, eltype(array_type), netcdf_spatial_dimensions(output),
           deflatelevel=deflatelevel, attrib=output_attributes)


@inline function save_output!(ds, output, model, ow, name)
    data = fetch_and_convert_output(output, model, ow)
    data = drop_output_dims(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[name][colons...] = data
    return nothing
end


@inline function write_time_invariant_fields!(model, ow::NetCDFOutputWriter, ti_fields; user_indices=(:, :, :), with_halos=false)
    ds = open(ow)
    @sync for (ti_field_name, ti_field) in ti_fields 
        indices = output_indices(ti_field, ti_field.grid, user_indices, with_halos)
        sliced_ti_field = Field(ti_field, indices=indices)

        define_time_invariant_variable!(ds, sliced_ti_field, ti_field_name, ow.array_type, 0, Dict())
        @async save_output!(ds, sliced_ti_field, model, ow, ti_field_name)
    end
    close(ds)
end


ccf_scratch = Field{Center, Center,  Face}(grid)
fff_scratch = Field{Face,   Face,    Face}(grid)
ccc_scratch = Field{Center, Center,  Center}(grid)
cnc_scratch = Field{Center, Nothing, Center}(grid)
cnf_scratch = Field{Center, Nothing, Face}(grid)
fnc_scratch = Field{Face,   Nothing, Center}(grid)
fnf_scratch = Field{Face,   Nothing, Face}(grid)
CellCenter  = (Center, Center, Center)

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

@inline Field_ccc(op) = Field((@at CellCenter op), data=ccc_scratch.data)


# ∂ⱼu₁ ⋅ F₁ⱼ
@inline Axᶜᶜᶜ_δuᶜᶜᶜ_F₁₁ᶜᶜᶜ(i, j, k, grid, closure, K_fields, clo, vels, fields, b) = -Axᶜᶜᶜ(i, j, k, grid) * δxᶜᵃᵃ(i, j, k, grid, vels.u) * viscous_flux_ux(i, j, k, grid, closure, K_fields, clo, fields, b)
@inline Ayᶠᶠᶜ_δuᶠᶠᶜ_F₁₂ᶠᶠᶜ(i, j, k, grid, closure, K_fields, clo, vels, fields, b) = -Ayᶠᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, vels.u) * viscous_flux_uy(i, j, k, grid, closure, K_fields, clo, fields, b)
@inline Azᶠᶜᶠ_δuᶠᶜᶠ_F₁₃ᶠᶜᶠ(i, j, k, grid, closure, K_fields, clo, vels, fields, b) = -Azᶠᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, vels.u) * viscous_flux_uz(i, j, k, grid, closure, K_fields, clo, fields, b)
 
# ∂ⱼu₂ ⋅ F₂ⱼ
@inline Axᶠᶠᶜ_δvᶠᶠᶜ_F₂₁ᶠᶠᶜ(i, j, k, grid, closure, K_fields, clo, vels, fields, b) = -Axᶠᶠᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, vels.v) * viscous_flux_vx(i, j, k, grid, closure, K_fields, clo, fields, b)
@inline Ayᶜᶜᶜ_δvᶜᶜᶜ_F₂₂ᶜᶜᶜ(i, j, k, grid, closure, K_fields, clo, vels, fields, b) = -Ayᶜᶜᶜ(i, j, k, grid) * δyᵃᶜᵃ(i, j, k, grid, vels.v) * viscous_flux_vy(i, j, k, grid, closure, K_fields, clo, fields, b)
@inline Azᶜᶠᶠ_δvᶜᶠᶠ_F₂₃ᶜᶠᶠ(i, j, k, grid, closure, K_fields, clo, vels, fields, b) = -Azᶜᶠᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, vels.v) * viscous_flux_vz(i, j, k, grid, closure, K_fields, clo, fields, b)
 
# ∂ⱼu₃ ⋅ F₃ⱼ
@inline Axᶠᶜᶠ_δwᶠᶜᶠ_F₃₁ᶠᶜᶠ(i, j, k, grid, closure, K_fields, clo, vels, fields, b) = -Axᶠᶜᶠ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, vels.w) * viscous_flux_wx(i, j, k, grid, closure, K_fields, clo, fields, b)
@inline Ayᶜᶠᶠ_δwᶜᶠᶠ_F₃₂ᶜᶠᶠ(i, j, k, grid, closure, K_fields, clo, vels, fields, b) = -Ayᶜᶠᶠ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, vels.w) * viscous_flux_wy(i, j, k, grid, closure, K_fields, clo, fields, b)
@inline Azᶜᶜᶜ_δwᶜᶜᶜ_F₃₃ᶜᶜᶜ(i, j, k, grid, closure, K_fields, clo, vels, fields, b) = -Azᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, vels.w) * viscous_flux_wz(i, j, k, grid, closure, K_fields, clo, fields, b)

@inline viscous_dissipation_ccc(i, j, k, grid, diffusivity_fields, vels, fields, p) =
    (Axᶜᶜᶜ_δuᶜᶜᶜ_F₁₁ᶜᶜᶜ(i, j, k, grid,         p.closure, diffusivity_fields, p.clock, vels, fields, p.buoyancy) + # C, C, C
     ℑxyᶜᶜᵃ(i, j, k, grid, Ayᶠᶠᶜ_δuᶠᶠᶜ_F₁₂ᶠᶠᶜ, p.closure, diffusivity_fields, p.clock, vels, fields, p.buoyancy) + # F, F, C  → C, C, C
     ℑxzᶜᵃᶜ(i, j, k, grid, Azᶠᶜᶠ_δuᶠᶜᶠ_F₁₃ᶠᶜᶠ, p.closure, diffusivity_fields, p.clock, vels, fields, p.buoyancy) + # F, C, F  → C, C, C

     ℑxyᶜᶜᵃ(i, j, k, grid, Axᶠᶠᶜ_δvᶠᶠᶜ_F₂₁ᶠᶠᶜ, p.closure, diffusivity_fields, p.clock, vels, fields, p.buoyancy) + # F, F, C  → C, C, C
     Ayᶜᶜᶜ_δvᶜᶜᶜ_F₂₂ᶜᶜᶜ(i, j, k, grid,         p.closure, diffusivity_fields, p.clock, vels, fields, p.buoyancy) + # C, C, C
     ℑyzᵃᶜᶜ(i, j, k, grid, Azᶜᶠᶠ_δvᶜᶠᶠ_F₂₃ᶜᶠᶠ, p.closure, diffusivity_fields, p.clock, vels, fields, p.buoyancy) + # C, F, F  → C, C, C

     ℑxzᶜᵃᶜ(i, j, k, grid, Axᶠᶜᶠ_δwᶠᶜᶠ_F₃₁ᶠᶜᶠ, p.closure, diffusivity_fields, p.clock, vels, fields, p.buoyancy) + # F, C, F  → C, C, C
     ℑyzᵃᶜᶜ(i, j, k, grid, Ayᶜᶠᶠ_δwᶜᶠᶠ_F₃₂ᶜᶠᶠ, p.closure, diffusivity_fields, p.clock, vels, fields, p.buoyancy) + # C, F, F  → C, C, C
     Azᶜᶜᶜ_δwᶜᶜᶜ_F₃₃ᶜᶜᶜ(i, j, k, grid,         p.closure, diffusivity_fields, p.clock, vels, fields, p.buoyancy)   # C, C, C
     ) / Vᶜᶜᶜ(i, j, k, grid) # This division by volume, coupled with the call to A*δuᵢ above, ensures a derivative operation

@inline function KineticEnergyDissipation(model::NonhydrostaticModel; energy_vel=model.velocities)
    parameters = (; model.closure, 
                  model.clock,
                  model.buoyancy)
    return KernelFunctionOperation{Center, Center, Center}(viscous_dissipation_ccc, model.grid, model.diffusivity_fields,
                                                           energy_vel, fields(model), parameters)
end


@inline function uᵢ_advectionᶜᶜᶜ(i, j, k, grid, energy_vel, velocities, advection)
    u∂ⱼuⱼu = ℑxᶜᵃᵃ(i, j, k, grid, ψf, energy_vel.u, div_𝐯u, advection, velocities, velocities.u)
    v∂ⱼuⱼv = ℑyᵃᶜᵃ(i, j, k, grid, ψf, energy_vel.v, div_𝐯v, advection, velocities, velocities.v)
    w∂ⱼuⱼw = ℑzᵃᵃᶜ(i, j, k, grid, ψf, energy_vel.w, div_𝐯w, advection, velocities, velocities.w)
    return u∂ⱼuⱼu + v∂ⱼuⱼv + w∂ⱼuⱼw
end

@inline function KineticEnergyAdvection(model::NonhydrostaticModel; velocities=model.velocities, energy_vel=model.velocities)
    return KernelFunctionOperation{Center, Center, Center}(uᵢ_advectionᶜᶜᶜ, model.grid, energy_vel, velocities, model.advection)
end


@inline function uᵢ_div_stressᶜᶜᶜ(i, j, k, grid, closure, diffusivity_fields,
                                  clock, model_fields, buoyancy, energy_vel)
    u∂ⱼ_τ₁ⱼ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, energy_vel.u, ∂ⱼ_τ₁ⱼ, closure, diffusivity_fields, clock, model_fields, buoyancy)
    v∂ⱼ_τ₂ⱼ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, energy_vel.v, ∂ⱼ_τ₂ⱼ, closure, diffusivity_fields, clock, model_fields, buoyancy)
    w∂ⱼ_τ₃ⱼ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, energy_vel.w, ∂ⱼ_τ₃ⱼ, closure, diffusivity_fields, clock, model_fields, buoyancy)
    return u∂ⱼ_τ₁ⱼ + v∂ⱼ_τ₂ⱼ + w∂ⱼ_τ₃ⱼ
end

@inline function KineticEnergyStress(model::NonhydrostaticModel; energy_vel=model.velocities)
    dependencies = (model.closure,
                    model.diffusivity_fields,
                    model.clock,
                    fields(model),
                    model.buoyancy,
                    energy_vel)
    return KernelFunctionOperation{Center, Center, Center}(uᵢ_div_stressᶜᶜᶜ, model.grid, dependencies...)
end


@inline function uᵢ_forcingᶜᶜᶜ(i, j, k, grid, forcings, clock, model_fields, energy_vel)
    uFᵘ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, energy_vel.u, forcings.u, clock, model_fields)
    vFᵛ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, energy_vel.v, forcings.v, clock, model_fields)
    wFʷ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, energy_vel.w, forcings.w, clock, model_fields)
    return uFᵘ + vFᵛ + wFʷ
end

@inline function KineticEnergyForcing(model::NonhydrostaticModel; energy_vel=model.velocities) 
    dependencies = (model.forcing,
                    model.clock,
                    fields(model),
                    energy_vel)
    return KernelFunctionOperation{Center, Center, Center}(uᵢ_forcingᶜᶜᶜ, model.grid, dependencies...)
end


@inline kernel_getbc(i, j, k, grid, boundary_condition, clock, fields) =
    getbc(boundary_condition, i, j, grid, clock, fields)

@inline function SurfaceMomentumFlux(model::NonhydrostaticModel, momentum_name)
    model_fields = fields(model)
    momentum = model.velocities[momentum_name]
    mom_bc = momentum.boundary_conditions.top
    LX = location(momentum, Int32(1))
    LY = location(momentum, Int32(2))
    return KernelFunctionOperation{LX, LY, Nothing}(kernel_getbc, model.grid, mom_bc, model.clock, model_fields)
end

@inline function SurfaceTracerFlux(model::NonhydrostaticModel, tracer_name)
    model_fields = fields(model)
    tracer = model.tracers[tracer_name]
    tra_bc = tracer.boundary_conditions.top
    LX = location(tracer, Int32(1))
    LY = location(tracer, Int32(2))
    return KernelFunctionOperation{LX, LY, Nothing}(kernel_getbc, model.grid, tra_bc, model.clock, model_fields)
end


@inline function XSubgridscaleNormalStress(model::NonhydrostaticModel)
    return KernelFunctionOperation{Center, Center, Center}(viscous_flux_ux, model.grid, model.closure,
                                                           model.diffusivity_fields, model.clock, fields(model), model.buoyancy)
end

@inline function YSubgridscaleNormalStress(model::NonhydrostaticModel)
    return KernelFunctionOperation{Center, Center, Center}(viscous_flux_vy, model.grid, model.closure,
                                                           model.diffusivity_fields, model.clock, fields(model), model.buoyancy)
end

@inline function ZSubgridscaleNormalStress(model::NonhydrostaticModel)
    return KernelFunctionOperation{Center, Center, Center}(viscous_flux_wz, model.grid, model.closure,
                                                           model.diffusivity_fields, model.clock, fields(model), model.buoyancy)
end

@inline function XSubgridscaleVerticalMomentumFlux(model::NonhydrostaticModel)
    return KernelFunctionOperation{Face, Center, Face}(viscous_flux_uz, model.grid, model.closure,
                                                       model.diffusivity_fields, model.clock, fields(model), model.buoyancy)
end

@inline function YSubgridscaleVerticalMomentumFlux(model::NonhydrostaticModel)
    return KernelFunctionOperation{Center, Face, Face}(viscous_flux_vz, model.grid, model.closure,
                                                       model.diffusivity_fields, model.clock, fields(model), model.buoyancy)
end

@inline function SubgridscaleTracerFlux(model::NonhydrostaticModel, tracer_name, direction; tracer=nothing)
    tracer_index = findfirst(n -> n === tracer_name, propertynames(model.tracers))
    tracer = tracer==nothing ? model.tracers[tracer_name] : tracer
    sgs_flux_func = direction==1 ? diffusive_flux_x : (direction==2 ? diffusive_flux_y : diffusive_flux_z)
    LX = direction==1 ? Face : Center
    LY = direction==2 ? Face : Center
    LZ = direction==3 ? Face : Center
    return KernelFunctionOperation{LX, LY, LZ}(sgs_flux_func, model.grid, model.closure,
                                               model.diffusivity_fields, Val(tracer_index), tracer,
                                               model.clock, fields(model), model.buoyancy)
end


@inline function Ertel_potential_vorticity_fff(i, j, k, grid, u, v, w, b, fx, fy, fz, dVgdz, dBdx)
    dWdy =  ℑxᶠᵃᵃ(i, j, k, grid, ∂yᶜᶠᶠ, w) # C, C, F  → C, F, F  → F, F, F
    dVdz =  ℑxᶠᵃᵃ(i, j, k, grid, ∂zᶜᶠᶠ, v) # C, F, C  → C, F, F  → F, F, F
    dbdx = ℑyzᵃᶠᶠ(i, j, k, grid, ∂xᶠᶜᶜ, b) # C, C, C  → F, C, C  → F, F, F
    pv_x = (fx + dWdy - dVdz - dVgdz) * (dbdx + dBdx) # F, F, F

    dUdz =  ℑyᵃᶠᵃ(i, j, k, grid, ∂zᶠᶜᶠ, u) # F, C, C  → F, C, F → F, F, F
    dWdx =  ℑyᵃᶠᵃ(i, j, k, grid, ∂xᶠᶜᶠ, w) # C, C, F  → F, C, F → F, F, F
    dbdy = ℑxzᶠᵃᶠ(i, j, k, grid, ∂yᶜᶠᶜ, b) # C, C, C  → C, F, C → F, F, F
    pv_y = (fy + dUdz - dWdx) * dbdy # F, F, F

    dVdx =  ℑzᵃᵃᶠ(i, j, k, grid, ∂xᶠᶠᶜ, v) # C, F, C  → F, F, C → F, F, F
    dUdy =  ℑzᵃᵃᶠ(i, j, k, grid, ∂yᶠᶠᶜ, u) # F, C, C  → F, F, C → F, F, F
    dbdz = ℑxyᶠᶠᵃ(i, j, k, grid, ∂zᶜᶜᶠ, b) # C, C, C  → C, C, F → F, F, F
    pv_z = (fz + dVdx - dUdy) * dbdz

    return pv_x + pv_y + pv_z
end

@inline function ErtelPotentialVorticityFrontalZone(model::NonhydrostaticModel, u, v, w, b, coriolis; M²=0)
    if coriolis isa FPlane
        fx = fy = 0
        fz = coriolis.f
    elseif coriolis isa ConstantCartesianCoriolis
        fx = coriolis.fx
        fy = coriolis.fy
        fz = coriolis.fz
    elseif coriolis == nothing
        fx = fy = fz = 0
    else
        throw(ArgumentError("ErtelPotentialVorticityFrontalZone is only implemented for FPlane and ConstantCartesianCoriolis"))
    end

    dBdx  = - M²
    dVgdz = dBdx / fz
    return KernelFunctionOperation{Face, Face, Face}(Ertel_potential_vorticity_fff, model.grid,
                                                     u, v, w, b, fx, fy, fz, dVgdz, dBdx)
end
