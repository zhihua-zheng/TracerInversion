#using Oceananigans.TurbulenceClosures: diffusivity, viscosity
#using Oceanostics.FlowDiagnostics: ErtelPotentialVorticity#, RichardsonNumber, RossbyNumber
#using Oceanostics.TKEBudgetTerms: PressureRedistributionTerm
#using Oceanostics.PotentialEnergyEquationTerms: PotentialEnergy

using Oceananigans.Grids: Center, Face
using Oceananigans.Operators
using Oceananigans.AbstractOperations: @at, ∂x, ∂z
ccc_scratch = Field{Center, Center,  Center}(grid)
cnc_scratch = Field{Center, Nothing, Center}(grid)
#cnf_scratch = Field{Center, Nothing, Face}(grid)
#fnc_scratch = Field{Face,   Nothing, Center}(grid)
#fnf_scratch = Field{Face,   Nothing, Face}(grid)
CellCenter  = (Center, Center, Center)

@inline Fintp_ccc(op) = Field((@at CellCenter op), data=ccc_scratch.data)
@inline Field_ccc(op) = Field(op,                  data=ccc_scratch.data)
@inline function y_average(F; field=true)
    avg = Average(F, dims=2)
    if field
        avg = Field(avg, data=cnc_scratch.data)
#        LX, LY, LZ = location(F)
#        if LX==Center && LZ==Center
#            avg = Field(avg, data=cnc_scratch.data)
#
#        elseif LX==Center && LZ==Face
#            avg = Field(avg, data=cnf_scratch.data)
#
#        elseif LX==Face && LZ==Center
#            avg = Field(avg, data=fnc_scratch.data)
#
#        elseif LX==Face && LZ==Face
#            avg = Field(avg, data=fnf_scratch.data)
#        end
    end
    return avg
end


@inline function get_outputs(model, save_mean)
    u  = @at CellCenter model.velocities.u
    v  = @at CellCenter model.velocities.v
    w  = @at CellCenter model.velocities.w
    b  = model.tracers.b

    #u′  = u - y_average(u)
    #w′  = w - y_average(w)
    #b′  = b - y_average(b)
    #cs′ = Dict{Symbol, Any}(Symbol(key, "′") => c - y_average(c) for (key,c) in pairs(cs))

    #Ri = Field(RichardsonNumber(model, uE, vE, w, model.tracers.b))
    #Ro = Field(RossbyNumber(model, uE, vE, w, model.coriolis))
    #q = Field(ErtelPotentialVorticityFrontalZone(model, uE, vE, w, b, model.coriolis, M²=p.M²), data=fff_scratch.data)

    # Diffusivity & viscosity
    #ufrc = -KernelFunctionOperation{Center, Center, Center}(∂ⱼ_τ₁ⱼ,   grid, model.closure, model.diffusivity_fields,
    #                                                        model.clock, fields(model), model.buoyancy)
    #vfrc = -KernelFunctionOperation{Center, Center, Center}(∂ⱼ_τ₂ⱼ,   grid, model.closure, model.diffusivity_fields,
    #                                                        model.clock, fields(model), model.buoyancy)
    #wfrc = -KernelFunctionOperation{Center, Center, Center}(∂ⱼ_τ₃ⱼ,   grid, model.closure, model.diffusivity_fields,
    #                                                        model.clock, fields(model), model.buoyancy)
    #bdia = -KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, grid, model.closure, model.diffusivity_fields, Val(1), b,
    #                                                        model.clock, fields(model), model.buoyancy)
    #νₑ = sum(viscosity(model.closure, model.diffusivity_fields))
    #κₑ = sum(diffusivity(model.closure, model.diffusivity_fields, Val(:b)))
    #νₑ = viscosity(model.closure, model.diffusivity_fields)
    #κₑ = diffusivity(model.closure, model.diffusivity_fields, Val(:b))
    #νₑ_ccf = @at (Center, Center, Face) νₑ_ccc
    #κₑ_ccf = @at (Center, Center, Face) κₑ_ccc

    #ub_sgs  = Field_ccc(SubgridscaleTracerFlux(model, :b,  1))
    #wb_sgs  = Field_ccc(SubgridscaleTracerFlux(model, :b,  3))
    #ubp_sgs = Field_ccc(SubgridscaleTracerFlux(model, :bp, 1))
    #wbp_sgs = Field_ccc(SubgridscaleTracerFlux(model, :bp, 3))
    #ucs_sgs = Dict{Symbol, Any}(Symbol(:u, key, :_sgs) => Field_ccc(SubgridscaleTracerFlux(model, key, 1)) for (key,_) in pairs(cs))
    #wcs_sgs = Dict{Symbol, Any}(Symbol(:w, key, :_sgs) => Field_ccc(SubgridscaleTracerFlux(model, key, 3)) for (key,_) in pairs(cs))

    # Correlations
    #u′b′  = Field_ccc(u′*b′)
    #w′b′  = Field_ccc(w′*b′)
    #u′cs′ = Dict{Symbol, Any}(Symbol(:u′, key) => Field_ccc(u′*c′) for (key,c′) in pairs(cs′))
    #w′cs′ = Dict{Symbol, Any}(Symbol(:w′, key) => Field_ccc(w′*c′) for (key,c′) in pairs(cs′))

    # Gradients
    #@compute κₑ_top = Field((@at (Center, Center, Face) κₑ), indices=(:, :, grid.Nz+1))
    #@compute bz_top = Field(- b.boundary_conditions.top.condition / κₑ_top)
    #bz_bcs = FieldBoundaryConditions(grid, (Center, Center, Face);
    #                                 top = OpenBoundaryCondition(bz_top),
    #                                 bottom = OpenBoundaryCondition(p.N₁²))
    #dbdzᶜᶜᶠ = Field(∂z(b), boundary_conditions=bz_bcs, data=ccf_scratch.data)
    #dbdz  = Field_ccc(dbdzᶜᶜᶠ)

    #dbdz  = Fintp_ccc(∂z(b))
    #dbdx  = Fintp_ccc(∂x(b))
    #dcsdz = Dict{Symbol, Any}(Symbol(:d, key, :dz) => Fintp_ccc(∂z(c)) for (key,c) in pairs(cs))
    #dcsdx = Dict{Symbol, Any}(Symbol(:d, key, :dx) => Fintp_ccc(∂x(c)) for (key,c) in pairs(cs))

    # Surface fluxes
    #Qu = Field(Average(SurfaceMomentumFlux(model, :u), dims=(1,2)))
    #Qv = Field(Average(SurfaceMomentumFlux(model, :v), dims=(1,2)))
    #Qb = Field(Average(SurfaceTracerFlux(model,   :b), dims=(1,2)))

    # Assemble outputs
    flow_state = Dict{Symbol, Any}(:u => u,
                                   :v => v,
                                   :w => w,
                                   :b => b)
    #flow_derived = Dict{Symbol, Any}(:dbdz => dbdz,
    #                                 :dbdx => dbdx,
    #                                 :u′b′ => u′b′,
    #                                 :w′b′ => w′b′)
    #surf_fluxes = Dict{Symbol, Any}(:Qu => Qu,
    #                                :Qv => Qv,
    #                                :Qb => Qb)

    #outputs_derived = merge(outputs_state, flow_derived, u′cs′, w′cs′, dcsdx, dcsdz)#, ucs_sgs, wcs_sgs)
    #outputs_mean    = Dict{Symbol, Any}(Symbol(key, :_ym) => y_average(val) for (key,val) in pairs(outputs_derived))
    #outputs_mean  = merge(outputs_ymean, surf_fluxes)

    if save_mean
        Bym = Field(Average(b, dims=2))
        Wym = Field(Average(w, dims=2))
        Vym = Field(Average(v, dims=2))
        Uym = Field(Average(u, dims=2))
        uprime = u - Uym
        vprime = v - Vym
        wprime = w - Wym
        bprime = b - Bym
        up2 = Field(Average(Field(Integral(uprime^2, dims=3)), dims=2))
        vp2 = Field(Average(Field(Integral(vprime^2, dims=3)), dims=2))
        wp2 = Field(Average(Field(Integral(wprime^2, dims=3)), dims=2))
        wbt = Field(Average(Field(Integral(wprime*bprime, dims=3)), dims=2))
        flow_mean = (; up2=up2, vp2=vp2, wp2=wp2, wbt=wbt)
        return flow_state, flow_mean
    else
        cs = Dict{Symbol, Any}(key => c for (key,c) in pairs(model.tracers) if key != :b)
        outputs_state = merge(flow_state, cs)
        return outputs_state
    end
end
