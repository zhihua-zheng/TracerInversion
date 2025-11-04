using Parameters

@with_kw struct SimParams
    commons = (;
               Lz = 150meters,     # depth
               Lf = 0.5kilometers, # initial frontal width
               Hm = 60meters,      # initial mixed layer depth
               f  = 1e-4,   # [s⁻¹] Coriolis frequency
               ρₐ = 1.225,  # [kg m⁻³] average density of air at the surface
               ρ₀ = 1026.0, # [kg m⁻³] average density of seawater at the surface
               cₚ = 3991.0, # [J K⁻¹ kg⁻¹] typical heat capacity for seawater
               αᵀ = 2e-4,   # [K⁻¹], thermal expansion coefficient
               ν₀ = 1.0e-6, # [m² s⁻¹] molecular viscosity
               κ₀ = 1.5e-7, # [m² s⁻¹] molecular diffusivity

               RiB₀  = 1,    # mixed layer balanced Richardson number
               noise = 1e-2, # initial noise amplitude
               save_ckp_interval = 1days,    # how often to save checkpoints 
               tracer_reset_interval = 1, # [Tf] how often to reset tracer distribution
               tracer_restoring_rate = 1/(10*24*3600), # [s⁻¹] the tracer restoring rate
               n_tracers = 7, # number of conserved passive tracers

               z_refinement = 1.38,# controls spacing near surface (higher means finer spaced)
               z_stretching = 26,  # controls rate of stretching at bottom
               damping_rate = 1/60, # [s⁻¹] relax fields on a time-scale comparable to N₁, following Taylor & Ferrari 2010
 
               use_stretched_z = true,
               use_fluxed_c    = true,
               restore_tracer  = false,
              )

    DoubleFront = (; commons...,
               Lx = 8kilometers, # east-west extent
               Ly = 4kilometers, # north-south extent
               Nx_full = 2000, # number of points in the x direction for full simulation
               Ny_full = 1000, # number of points in the y direction for full simulation
               Nz_full = 64,   # number of points in the z direction for full simulation
               cfl     = 0.9,  # Courant-Friedrichs–Lewy number
               max_Δt  = 5minutes, # max time step
               ckp_group = "double-front",
               )
end


function decode_casename(casename)
    fres, b_gradient, Ri_lower, ensemble, heat_flux, wind_stress, wind_direction, Stokes_flag = split(casename, "_")
    case_type = fres[1]
    coarsen_h = parse(Int64, fres[2])
    coarsen_z = parse(Int64, fres[3])
    M² = parse(Float64, lstrip(b_gradient, 'M')) / 1e8 # [s⁻²] horizontal buoyancy gradient
    Q₀ = parse(Float64, lstrip(heat_flux,  'Q'))
    τ₀ = parse(Float64, lstrip(wind_stress,'W')) / 1e3
    if startswith(wind_direction, 'D')
        θ₀ = parse(Float64, lstrip(wind_direction, 'D'))
        wind_oscillation = false
    elseif startswith(wind_direction, 'O')
        θ₀ = parse(Float64, lstrip(wind_direction, 'O'))
        wind_oscillation = true
    else
        throw(DomainError(wind_direction, "wind direction identifier not supported"))
    end
    use_Stokes = parse(Bool, Stokes_flag[end])
    RiB₁ = parse(Float64, lstrip(Ri_lower, ['R', 'i']))
    em   = parse(Int64,   lstrip(ensemble, ['e', 'm']))
    return case_type, coarsen_h, coarsen_z, M², RiB₁, em, Q₀, τ₀, θ₀, wind_oscillation, use_Stokes
end


function enrich_parameters(params, casename, spinup, init_tracer)
    if spinup
        spinup_casename = SubString(casename, 1:19) * "_Q000_W000_D000_St0"
        output_prefix = SubString(casename, 1:19) * "_spinup"
        case_type, coarsen_h, coarsen_z, M², RiB₁, em, Q₀, τ₀, θ₀, wind_oscillation, use_Stokes = decode_casename(spinup_casename)
        stop_time = 10days
        save_out_interval = 30minutes
        ckp_prefix = "spinup"
        save_mean  = true
        save_ckp   = true
    else
        case_type, coarsen_h, coarsen_z, M², RiB₁, em, Q₀, τ₀, θ₀, wind_oscillation, use_Stokes = decode_casename(casename)
        save_out_interval = 20minutes
        ckp_prefix = SubString(casename, 21:24) * ifelse(params.use_fluxed_c, "_with-tracer-fluxed", "_with-tracer")
        if init_tracer
            pickup_time = ifelse(M²==3e-8, 6days, 4days)
            pickup_idx  = Int64(pickup_time / params.save_ckp_interval) + 1 # accounting for day 0
            stop_time   = pickup_time + 2days
            save_mean   = false
            save_ckp    = true
            pickup_prefix = "spinup"
            output_prefix = casename * ifelse(params.use_fluxed_c, "_init-tracer-fluxed", "_init-tracer")
        else
            pickup_time = 7days
            pickup_idx  = Int64((pickup_time - 5days) / params.save_ckp_interval)
            stop_time   = 9days
            save_mean   = false
            save_ckp    = false
            pickup_prefix = SubString(casename, 21:24) * ifelse(params.use_fluxed_c, "_with-tracer-fluxed", "_with-tracer")
            output_prefix = casename * ifelse(params.use_fluxed_c, "_with-tracer-fluxed", "_with-tracer")
        end
    end

    sponge_σ   = round(√((params.Lz/5)^2 / 6 / 2), sigdigits=3) # [m] sponge layer Gaussian mask width (thickness: Lz/5, tapering to e⁻⁶)
    σ_wind     = ifelse(wind_oscillation, params.f, 0)
    wind_waves = ifelse((τ₀==0) & use_Stokes, false, true)

    Lp  = params.Lx / 4 # initial frontal center location
    N₀² = params.RiB₀ * (M² / params.f)^2 # [s⁻²] mixed layer buoyancy frequency squared
    N₁² =        RiB₁ * (M² / params.f)^2 # [s⁻²] thermocline buoyancy frequency squared
    ϵv  = params.noise * M² / params.f * params.Hm # velocity noise amplitude
    ϵb  = params.noise * M² * params.Lf            # buoyancy noise amplitude
    #Tf = 2π/params.f # [s] inertial period
    #t₀ = start_Tf*Tf # [s] when to introduce wind stress
    #tᵣ = √2*Tf       # [s] length of the linear ramp of wind forcing

    Nx = params.Nx_full ÷ coarsen_h
    Ny = params.Ny_full ÷ coarsen_h
    Nz = params.Nz_full ÷ coarsen_z

    τ₀ˣ   = τ₀ * cosd(θ₀)
    τ₀ʸ   = τ₀ * sind(θ₀)
    Cd    = 1.2e-3 # neutral drag coefficient, LP81, #Fig. 9 from Edson et al. 2013
    U₁₀   = ifelse(wind_waves, √(τ₀ / params.ρₐ / Cd), 5) # [m s⁻¹] surface wind speed used to specify surface waves
    Uˢ    = ifelse(use_Stokes, 0.068, 0)#0.0155 * U₁₀, 0) # [m s⁻¹] surface Stokes drift velocity
    #Dˢ    = ifelse(use_Stokes, 0.14 * U₁₀^2 / g_Earth, 1) # [m] e-folding scale of the Stokes drift (avoid blowing up)
    Dˢ    = ifelse(use_Stokes, 4.77, 1)
    B₀    = g_Earth * params.αᵀ * Q₀ / (params.ρ₀ * params.cₚ) # [m² s⁻³] surface buoyancy flux
    ustar = √(τ₀ / params.ρ₀) # [m s⁻¹] friction velocity
    wstar = ∛(B₀ * params.Hm) # [m s⁻¹] convective velocity
    Vg    = M² / params.f * params.Hm # [m s⁻¹] geostrophic velocity scale

    n_per_set = (params.n_tracers - 1)/2 # number of tracers in each set that has the same restoring time scale

    extra_params = Base.@locals()
    delete!(extra_params, :params)
    delete!(extra_params, :casename)
    return merge(params, NamedTuple(extra_params))
end
