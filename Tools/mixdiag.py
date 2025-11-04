#!/usr/bin/env python3

import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.ndimage import uniform_filter


def interp_to_zi(a, zi):
    return xr.apply_ufunc(np.interp, zi, a.zC, a,
                          input_core_dims=[[], ['zC'], ['zC']],
                          output_core_dims=[[]],
                          output_dtypes=[float],
                          dask='parallelized',
                          vectorize=True)


def front_boundary_ufunc(M, x, Mc=0.2, in_km=0.1, out_km=0.1):
    dx = np.abs(x[1] - x[0])
    mask_fz = np.zeros_like(x)
    idx_front = np.argwhere(M >= Mc)
    idx_first, idx_last = idx_front[[0, -1]]
    idx_fisrt_expanded = int(idx_first[0] - in_km//dx)
    idx_last_expanded  = int(idx_last[0] + 1 + out_km//dx)
    mask_fz[idx_fisrt_expanded:idx_last_expanded] = 1
    mask_fz[(idx_last_expanded+1):] = 2
    return mask_fz


def double_front_boundary(M, Mc=0.2, in_km=0.1, out_km=0.1):
    Nhalf = M.xC.size//2
    # moving average in x direction
    M_smooth = xr.apply_ufunc(uniform_filter, M,
                              input_core_dims=[['yC', 'xC']],
                              output_core_dims=[['yC', 'xC']],
                              kwargs=dict(size=25, mode='wrap'),
                              output_dtypes=[float],
                              dask='parallelized')
    Ml = M_smooth.isel(xC=slice(None, Nhalf)).isel(xC=slice(None, None, -1))
    Mr = M_smooth.isel(xC=slice(Nhalf, None))
    mask_fzl = xr.apply_ufunc(front_boundary_ufunc, Ml, Ml.xC,
                              input_core_dims=[['xC'], ['xC']],
                              output_core_dims=[['xC']],
                              kwargs=dict(Mc=Mc, in_km=in_km, out_km=out_km),
                              vectorize=True,
                              output_dtypes=[float],
                              dask='parallelized')
    mask_fzr = xr.apply_ufunc(front_boundary_ufunc, Mr, Mr.xC,
                              input_core_dims=[['xC'], ['xC']],
                              output_core_dims=[['xC']],
                              kwargs=dict(Mc=Mc, in_km=in_km, out_km=out_km),
                              vectorize=True,
                              output_dtypes=[float],
                              dask='parallelized')
    mask_fz = xr.concat([mask_fzl.isel(xC=slice(None, None, -1)), mask_fzr], 'xC')
    return mask_fz


def arg_local_max_last(y, threshold):
    """
    https://gist.github.com/ben741/d8c70b608d96d9f7ed231086b237ba6b
    """
    idx_peaks = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]) * (y[1:-1] > threshold))[0] + 1
    return idx_peaks[-1]


def linear_interp_zero_crossing(x0, x1, y0, y1):
    """
    Find the value of y when x = 0
    """
    return y0 - (y1 - y0) / (x1 - x0) * x0


def PE_Kernel_linear(R_layer, dRdz, Z_U, Z_L):
    PE = R_layer / 2 * (Z_U**2 - Z_L**2) +\
         dRdz / 3 * (Z_U**3 - Z_L**3) -\
         dRdz / 2 * (Z_U**2 - Z_L**2) * (Z_U + Z_L) / 2
    return PE


def PE_Kernel_constant(R_layer, Z_U, Z_L):
    PE = R_layer / 2 * (Z_U**2 - Z_L**2)
    return PE


def MixLayers(Tracer, dZ, keepdims=True):
    DZ_Mixed = np.sum(dZ, axis=-1, keepdims=keepdims)
    T_Mixed  = np.sum((Tracer*dZ), axis=-1, keepdims=keepdims) / DZ_Mixed
    return T_Mixed, DZ_Mixed


def get_mld_PE_anomaly_Newton_ufunc(Zc, dZ, b_layer, dbdz=None, energy=10, rho0=1026, CNVG_T=1e-2, grav=9.81):
    """
    Source:
    https://github.com/breichl/oceanmixedlayers/blob/3aba2fcb05e2b65e343de9b0c49bfafa28345117/oceanmixedlayers/energy_Newton.py
    """
    Rho0_layer = rho0 - b_layer * rho0 / grav
    energy = energy / grav

    # The syntax below is written assuming an nd structure of Rho0, Zc, and dZ, where n >= 2.
    # If a single column is passed in we convert to a 2d array.
    if len(np.shape(Rho0_layer)) == 1:
        Rho0_layer = np.atleast_2d(Rho0_layer)
    if np.shape(Rho0_layer) != np.shape(Zc):
        Zc = np.broadcast_to(Zc, np.shape(Rho0_layer))
        dZ = np.broadcast_to(dZ, np.shape(Rho0_layer))
    
    ND = Rho0_layer.shape[:-1]
    NZ = Rho0_layer.shape[-1]

    Z_U = Zc + dZ/2
    Z_L = Zc - dZ/2
    
    Rho0_Mixed = np.copy(Rho0_layer)
    
    ACTIVE   = np.ones(ND, dtype='bool')
    FINISHED = np.zeros(ND,dtype='bool')
    
    # ACTIVE[np.isnan(Rho0_layer[..., -1])]   = False
    # FINISHED[np.isnan(Rho0_layer[..., -1])] = True
    
    MLD               = np.full(ND, np.nan)
    MLD[ACTIVE]       = 0
    PE_after          = np.full(ND, np.nan)
    PE_after[ACTIVE]  = 0
    PE_before         = np.full(ND, np.nan)
    PE_before[ACTIVE] = 0
    IT_total          = np.full(ND, np.nan)
    IT_total[ACTIVE]  = 0
    
    z = NZ
    while(z > 0 and np.sum(ACTIVE) > 0):
        z -= 1
        
        CNVG  = np.zeros(ND, dtype='bool')
        FINAL = np.zeros(ND, dtype='bool')
        # ACTIVE[np.isnan(Rho0_layer[..., z])]   = False
        # FINISHED[np.isnan(Rho0_layer[..., z])] = True
        
        PE_before[ACTIVE] = np.sum(PE_Kernel_constant(Rho0_layer[ACTIVE, z:],
                                                             Z_U[ACTIVE, z:],
                                                             Z_L[ACTIVE, z:]), axis=-1)
        
        Rho0_Mixed[ACTIVE,z:],_ = MixLayers(Rho0_layer[ACTIVE, z:],
                                                    dZ[ACTIVE, z:])
        
        PE_after[ACTIVE] = np.sum(PE_Kernel_constant(Rho0_Mixed[ACTIVE, z:],
                                                            Z_U[ACTIVE, z:],
                                                            Z_L[ACTIVE, z:]), axis=-1)
        
        FINAL[ACTIVE]  = (PE_after[ACTIVE] - PE_before[ACTIVE]) >= energy
        ACTIVE[ACTIVE] = (PE_after[ACTIVE] - PE_before[ACTIVE]) <  energy
        
        MLD[ACTIVE] += dZ[ACTIVE, z]

        # First guess for an iteration using Newton's method
        X = dZ[FINAL, z] * 0.5

        IT = 0
        IT_Lim = 10
        while (np.sum(FINAL) >= 1 and IT <= IT_Lim):
            IT += 1
            cnvgd_at_this_IT = np.zeros(ND, dtype='bool')

            #Needed for the Newton iteration
            #Within the iteration so that the size updates with each iteration.
            #In principle we might move this outside in favor of more logical indexing
            # but since the iteration converges in 2 steps it is probably OK.
            R1,D1 = MixLayers(Rho0_layer[FINAL, (z+1):], dZ[FINAL, (z+1):], keepdims=False)
            R2,D2 = Rho0_layer[FINAL, z], dZ[FINAL, z]
            
            Ca  = -R2
            Cb  = -(R1 * D1 + R2 * (2 * D1))
            D   = D1**2
            Cc  = -(R1 * D1 * (2 * D1) + (R2 * D))
            Cd  = -R1 * (D1 * D)
            Ca2 = R2
            Cb2 = R2 * (2. * D1)
            C   = D2**2 + D1**2 + 2 * (D1 * D2)
            Cc2 = R2 * (D - C)

            # We are trying to solve the function:
            # F(x) = G(x)/H(x)+I(x)
            # for where F(x) = PE+PE_threshold, or equivalently for where
            # F(x) = G(x)/H(x)+I(x) - (PE+PE_threshold) = 0
            # We also need the derivative of this function for the Newton's method iteration
            # F'(x) = (G'(x)H(x)-G(x)H'(x))/H(x)^2 + I'(x)
            # G and its derivative
            Gx  = 0.5 * (Ca * (X*X*X) + Cb * X**2 + Cc * X + Cd)
            Gpx = 0.5 * (3 * (Ca * X**2) + 2 * (Cb * X) + Cc)
            # H, its inverse, and its derivative
            Hx  = D1 + X
            iHx = 1 / Hx
            Hpx = 1
            # I and its derivative
            Ix  = 0.5 * (Ca2 * X**2 + Cb2 * X + Cc2)
            Ipx = 0.5 * (2 * Ca2 * X + Cb2)
            
            # The Function and its derivative:
            PE_Mixed = Gx * iHx + Ix
            Fgx = PE_Mixed - (PE_before[FINAL] + energy)
            Fpx = (Gpx * Hx - Hpx * Gx) * iHx**2 + Ipx
                
            # Check if our solution is within the threshold bounds, if not update
            # using Newton's method.  This appears to converge almost always in
            # one step because the function is very close to linear in most applications.
            CNVG_ = abs(Fgx) < (energy * CNVG_T)
            CNVG[FINAL] = CNVG_
            #Disable any that have converged and add to output
            cnvgd_at_this_IT[FINAL] = CNVG_
            MLD[cnvgd_at_this_IT] += X[CNVG_]
            FINAL[CNVG] = False

            #Update those that haven't converged
            nCNVG_ = ~CNVG_
            X2 = X[nCNVG_] - Fgx[nCNVG_] / Fpx[nCNVG_]
            X  = X2
            
            IT_FAILED = (X2 < 0) | (X2 > D2[nCNVG_])
            # The iteration seems to be robust, but we need to do something *if*
            # things go wrong... How should we treat failed iteration?
            # Present solution: Fail the entire algorithm.
            if np.sum(IT_FAILED) > 0:
                print(IT, 'Iteration failed in energy_newiteration')
            
            if (IT == IT_Lim and np.sum(FINAL) > 0):
                print(IT, np.sum(FINAL), ' #not converged')
    return MLD


def get_mld_PE_anomaly_bisection_ufunc(Zc, dZ, b_layer, dbdz, energy=10, rho0=1026, CNVG_T=1e-2, grav=9.81):
    """
    Source:
    https://github.com/breichl/oceanmixedlayers/blob/3aba2fcb05e2b65e343de9b0c49bfafa28345117/oceanmixedlayers/energy.py
    """
    Rho0_layer = rho0 - b_layer * rho0 / grav
    dRho0dz_layer = - dbdz * rho0 / grav
    energy = energy / grav

    # The syntax below is written assuming an nd structure of Rho0, Zc, and dZ, where n >= 2.
    # If a single column is passed in we convert to a 2d array.
    if len(np.shape(Rho0_layer)) == 1:
        Rho0_layer    = np.atleast_2d(Rho0_layer)
        dRho0dz_layer = np.atleast_2d(dRho0dz_layer)
    if np.shape(Rho0_layer) != np.shape(Zc):
        Zc = np.broadcast_to(Zc, np.shape(Rho0_layer))
        dZ = np.broadcast_to(dZ, np.shape(Rho0_layer))

    ND = Rho0_layer.shape[:-1]
    NZ = Rho0_layer.shape[-1]
    Rho0_Mixed = np.copy(Rho0_layer)

    Z_U = Zc + dZ/2
    Z_L = Zc - dZ/2

    ACTIVE   = np.ones(ND,  dtype='bool')
    FINISHED = np.zeros(ND, dtype='bool')

    MLD                     = np.full(ND, np.nan)
    MLD[ACTIVE]             = 0
    PE_after                = np.full(ND, np.nan)
    PE_after[ACTIVE]        = 0
    PE_before               = np.full(ND, np.nan)
    PE_before[ACTIVE]       = 0
    PE_before_above         = np.full(ND, np.nan)
    PE_before_above[ACTIVE] = 0
    dz_Mixed                = np.full(ND+(1,), np.nan)
    dz_Mixed[ACTIVE, :]     = 0

    z = NZ
    while(z > 0 and np.sum(ACTIVE) > 0):
        z -= 1

        CNVG  = np.zeros(ND, dtype='bool')
        FINAL = np.zeros(ND, dtype='bool')

        PE_before_above[ACTIVE] = np.sum(PE_Kernel_linear(Rho0_layer[ACTIVE, (z+1):],
                                                       dRho0dz_layer[ACTIVE, (z+1):],
                                                                 Z_U[ACTIVE, (z+1):],
                                                                 Z_L[ACTIVE, (z+1):]), axis=-1)

        PE_before[ACTIVE] = np.sum(PE_Kernel_linear(Rho0_layer[ACTIVE, z:],
                                                 dRho0dz_layer[ACTIVE, z:],
                                                           Z_U[ACTIVE, z:],
                                                           Z_L[ACTIVE, z:]), axis=-1)

        Rho0_Mixed[ACTIVE, z:], dz_Mixed[ACTIVE, :] = MixLayers(Rho0_layer[ACTIVE, z:],
                                                                        dZ[ACTIVE, z:])

        PE_after[ACTIVE] = np.sum(PE_Kernel_constant(Rho0_Mixed[ACTIVE, z:],
                                                            Z_U[ACTIVE, z:],
                                                            Z_L[ACTIVE, z:]), axis=-1)

        FINAL[ACTIVE]  = (PE_after[ACTIVE] - PE_before[ACTIVE]) >= energy
        ACTIVE[ACTIVE] = (PE_after[ACTIVE] - PE_before[ACTIVE]) <  energy

        IT   = -1
        DZup = dZ[FINAL, z:(z+1)]
        DZlo = np.zeros_like(DZup)
        while np.sum(FINAL) >= 1:
            IT += 1
            DZ = np.concatenate(((DZlo + DZup) / 2, dZ[FINAL, (z+1):]), axis=-1)
            # print(f'Iteration {IT}: dz={DZ[:, 0]}')

            Rho0_layers_linear = Rho0_layer[FINAL, z:]
            Rho0_layers_linear[:, 0] = Rho0_layer[FINAL, z] + dRho0dz_layer[FINAL, z] * dZ[FINAL, z] / 2\
                                       - dRho0dz_layer[FINAL, z] * DZ[:, 0] / 2 # adjustment from value at top to value at center

            PE_before[FINAL] = PE_before_above[FINAL] + PE_Kernel_linear(Rho0_layers_linear[:, 0],
                                                                          dRho0dz_layer[FINAL, z],
                                                                                    Z_U[FINAL, z],
                                                                                    Z_U[FINAL, z] - DZ[:, 0])

            Rho0_Mixed[FINAL, z:], dz_Mixed[FINAL, :] = MixLayers(Rho0_layers_linear, DZ)

            PE_after[FINAL] = np.sum(PE_Kernel_constant(Rho0_Mixed[FINAL, z:],
                                                               Z_U[FINAL, z:],
                                                               Z_U[FINAL, z:] - DZ), axis=-1)
            if IT < 50:
                cf = abs(PE_after[FINAL] - PE_before[FINAL] - energy) <= (energy*CNVG_T)
                CNVG[FINAL] = cf
                # print(f'PE before: {PE_before[FINAL]}')
                # print(f'PE after: {PE_after[FINAL]}')
                # print(f'PE change: {PE_after[FINAL] - PE_before[FINAL]}\n')
                
                MLD[CNVG]      = dz_Mixed[CNVG, 0]
                FINISHED[CNVG] = True
                FINAL[CNVG]    = False

                if np.sum(~cf) > 0:
                    DZup = DZup[~cf, :]
                    DZlo = DZlo[~cf, :]
                    TOO_HIGH = (PE_after[FINAL] - PE_before[FINAL]) > energy
                    TOO_LOW  = (PE_after[FINAL] - PE_before[FINAL]) < energy
                    DZup[TOO_HIGH, 0] = DZ[~cf, 0][TOO_HIGH]
                    DZlo[TOO_LOW,  0] = DZ[~cf, 0][TOO_LOW]
            else:
                print(f'iteration not converged, remaining profiles: {np.sum(FINAL)}')
                # print(np.mean(np.abs(PE_after[FINAL]-PE_before[FINAL])))
                # print(DZup, DZlo, DZ[:, 0])
                # print(Z_U[FINAL, :])
                MLD[FINAL]   = dz_Mixed[FINAL, 0]
                FINAL[FINAL] = False
    return MLD


def get_mld_PE_anomaly_interp_ufunc(Zc, dZ, b_layer, dbdz, energy=10, rho0=1026, CNVG_T=1e-2, grav=9.81):
    """
    Use interpolation to estimate MLD where the PE anomaly equals to the prescribed energy.
    This is much faster and stabler than the iterative approach (Reichl et al. 2022), but may be inaccurate if the vertical grid is coarse.
    """
    Rho0_layer = rho0 - b_layer * rho0 / grav
    dRho0dz_layer = - dbdz * rho0 / grav
    energy = energy / grav

    # The syntax below is written assuming an nd structure of Rho0, Zc, and dZ, where n >= 2.
    # If a single column is passed in we convert to a 2d array.
    if len(np.shape(Rho0_layer)) == 1:
        Rho0_layer    = np.atleast_2d(Rho0_layer)
        dRho0dz_layer = np.atleast_2d(dRho0dz_layer)
    if np.shape(Rho0_layer) != np.shape(Zc):
        Zc = np.broadcast_to(Zc, np.shape(Rho0_layer))
        dZ = np.broadcast_to(dZ, np.shape(Rho0_layer))

    ND = Rho0_layer.shape[:-1]
    NZ = Rho0_layer.shape[-1]
    Rho0_Mixed = np.copy(Rho0_layer)

    Z_U = Zc + dZ/2
    Z_L = Zc - dZ/2

    ACTIVE   = np.ones(ND,  dtype='bool')
    FINISHED = np.zeros(ND, dtype='bool')

    MLD                     = np.full(ND, np.nan)
    MLD[ACTIVE]             = 0
    PE_after                = np.full(ND, np.nan)
    PE_after[ACTIVE]        = 0
    PE_before               = np.full(ND, np.nan)
    PE_before[ACTIVE]       = 0
    PE_change               = np.full(ND, np.nan)
    PE_change[ACTIVE]       = 0
    PE_change_above         = np.full(ND, np.nan)
    PE_change_above[ACTIVE] = 0

    z = NZ
    while(z > 0 and np.sum(ACTIVE) > 0):
        z -= 1

        FINAL = np.zeros(ND, dtype='bool')

        PE_before[ACTIVE] = np.sum(PE_Kernel_linear(Rho0_layer[ACTIVE, z:],
                                                 dRho0dz_layer[ACTIVE, z:],
                                                           Z_U[ACTIVE, z:],
                                                           Z_L[ACTIVE, z:]), axis=-1)

        Rho0_Mixed[ACTIVE, z:],_ = MixLayers(Rho0_layer[ACTIVE, z:],
                                                     dZ[ACTIVE, z:])

        PE_after[ACTIVE] = np.sum(PE_Kernel_constant(Rho0_Mixed[ACTIVE, z:],
                                                            Z_U[ACTIVE, z:],
                                                            Z_L[ACTIVE, z:]), axis=-1)

        PE_change[ACTIVE] = PE_after[ACTIVE]  -  PE_before[ACTIVE]
        FINAL[ACTIVE]     = PE_change[ACTIVE] >= energy

        if np.sum(FINAL) >= 1:
            PE_change_too_high = PE_change[FINAL]       - energy # positive
            PE_change_too_low  = PE_change_above[FINAL] - energy # negative

            MLD[FINAL] = -linear_interp_zero_crossing(PE_change_too_low, PE_change_too_high,
                                                      Zc[FINAL, z+1],    Zc[FINAL, z])
            FINISHED[FINAL] = True

        PE_change_above[ACTIVE] = PE_change[ACTIVE]
        ACTIVE[ACTIVE] = PE_change[ACTIVE] < energy
    return MLD


def get_mld_PE_anomaly(dzF, b, attrs, dbdz=None, energy=10, CNVG_T=1e-2):
    if dbdz is None: dbdz = b*0
    rho0 = attrs['ρ₀'].item()
    return xr.apply_ufunc(get_mld_PE_anomaly_interp_ufunc, b.zC, dzF, b, dbdz,
                          input_core_dims=[['zC'], ['zC'], ['zC'], ['zC']],
                          output_core_dims=[[]],
                          output_dtypes=[float],
                          kwargs=dict(energy=energy, rho0=rho0, CNVG_T=CNVG_T),
                          dask='parallelized',
                          vectorize=False)


def get_mld_delb_ufunc(z, delb):
    if np.all(delb >= 0) or (delb[-1] <= 0):
        return np.nan
    last_idx = np.max(np.where(delb <= 0))
    b0, b1 = delb[last_idx], delb[last_idx+1]
    z0, z1 = z[last_idx], z[last_idx+1]
    zmld = linear_interp_zero_crossing(b0, b1, z0, z1)
    return -zmld


def get_mld_delb(b, db_threshold=5.4e-6):
    delb = b - (b.isel(zC=-1) - db_threshold)
    return xr.apply_ufunc(get_mld_delb_ufunc, b.zC, delb,
                          input_core_dims=[['zC'], ['zC']],
                          output_core_dims=[[]],
                          output_dtypes=[float],
                          dask='parallelized',
                          vectorize=True)


def get_kpp_phi(zeta):
    phis = np.full_like(zeta, np.nan, dtype=float)
    idx_stable          = zeta >= 0
    idx_unstable_weak   = (zeta > -1) & (zeta < 0)
    idx_unstable_strong = zeta <= -1
    phis[idx_stable]          = 1 + 5*zeta[idx_stable]
    phis[idx_unstable_weak]   = (1 - 16*zeta[idx_unstable_weak])**(-1/2)
    phis[idx_unstable_strong] = (-28.86 - 98.96*zeta[idx_unstable_strong])**(-1/3)
    return phis#, phim


def get_kpp_w_scale(sigma, bld, ustar, B0, vonK=0.4, rsl=0.1):
    if ustar > 0:
        if B0 <= 0: # stablizing
            sigma_loc = sigma
        else: # destablizing
            sigma_loc = np.minimum(rsl, sigma)
        LObk = -ustar**3 / vonK / B0
        zeta = sigma_loc * bld / LObk
        phis = get_kpp_phi(zeta)
        ws   = vonK * ustar / phis
    elif ustar == 0:
        if B0 <= 0: # stablizing
            ws = np.full_like(sigma, 0)
        else:  # destablizing
            cs = 98.96
            wstar = (bld * B0)**(1/3)
            ws = vonK * (cs*vonK * np.minimum(rsl, sigma))**(1/3) * wstar
    return ws#, wm


def get_vt2(d, ws, NNmax, Ribc=0.3, rsl=0.1, vonK=0.4, beta_T=-0.2, opt='LMD94'):
    Ne = np.sqrt(np.maximum(0, NNmax))
    Cv = 1.7 if Ne > 2e-3 else 2.1 - 200*Ne
    cs = 98.96
    if opt == 'LMD94': # Large et al. 1994
        vt2 = Cv*d*Ne*ws*np.sqrt(-beta_T/cs/rsl) / (vonK**2) / Ribc
    else:
        print(f'option {opt} not supported.')
    #else: # Li & Fox-Kemper 2017
    #    rL  = 1 #(1+0.49*La_sl**(-2))
    #    vt2 = Cv*d*Ne*np.sqrt((0.15*wstar3 + 0.17*ustar**3*rL) / ws) / Ribc
    return np.maximum(1e-10, vt2)


def get_zref_idx_for_Rib(zC, zF, rsl=0.1):
    zref_idx = np.ones_like(zC, dtype=int)
    slt_of_d = rsl * np.abs(zC)
    for k in reversed(range(len(zC))):
        for kk in reversed(range(k, len(zC))):
            if (zF[-1] - zF[kk]) >= slt_of_d[k]:
                zref_idx[k] = kk
                break
    return zref_idx


def reversed_cumsum(arr):
    return np.cumsum(arr[::-1])[::-1]


def get_bld_Rib_1d(zC, zF, dzF, idxR, b, u, v, NN, LS, ustar=6e-3, B0=1e-9, Ribc=0.3, rsl=0.1):
    Rib  = np.full_like(b, np.nan)
    bref = b[-1]
    uref = u[-1]
    vref = v[-1]
    nz   = zC.size

    bdzF_cumsum = reversed_cumsum(b*dzF)
    udzF_cumsum = reversed_cumsum(u*dzF)
    vdzF_cumsum = reversed_cumsum(v*dzF)
    LS_abs_mean = reversed_cumsum(np.sqrt(LS)*dzF) / reversed_cumsum(np.ones_like(LS)*dzF)

    # bulk Richardson number (column sampling approach, see Griffies et al. 2015)
    for k in reversed(range(nz)):
        d = -zC[k]
        kref = idxR[k]
        slt_of_d = rsl * d
        # update reference value
        if kref < (nz - 1):
            partial_cell_thickness = slt_of_d + zF[kref+1]
            bref = (b[kref] * partial_cell_thickness + bdzF_cumsum[kref+1]) / slt_of_d
            uref = (u[kref] * partial_cell_thickness + udzF_cumsum[kref+1]) / slt_of_d
            vref = (v[kref] * partial_cell_thickness + vdzF_cumsum[kref+1]) / slt_of_d

        duv2   = (uref - u[k])**2 + (vref - v[k])**2
        ws     = get_kpp_w_scale(1, d, ustar, B0)
        vt2    = get_vt2(d, ws, NN[k], Ribc=Ribc, rsl=rsl)
        Duv2   = duv2 + vt2 + LS_abs_mean[k]**2 * d**2
        Rib[k] = (d - slt_of_d/2) * (bref - b[k]) / Duv2

    if np.nanmin(Rib) > Ribc:
        bld = -zC[-1]
    elif np.nanmax(Rib) < Ribc:
        bld = np.nan
    else:
        first_idx = np.max(np.where((Rib - Ribc) >= 0))
        bld = -linear_interp_zero_crossing(Rib[first_idx]   - Ribc,
                                           Rib[first_idx+1] - Ribc, zC[first_idx], zC[first_idx+1])
    return bld, Rib


def get_bld_Rib_ufunc(zC, zF, dzF, idxR, b, u, v, NN, LS, ustar=6e-3, B0=1e-9, Ribc=0.3, rsl=0.1, save_Rib=False):
    leading = b.shape[:-1]
    nz = b.shape[-1]
    ncols = int(np.prod(leading)) if leading else 1
    b2d  = b.reshape(ncols, nz)
    u2d  = u.reshape(ncols, nz)
    v2d  = v.reshape(ncols, nz)
    NN2d = NN.reshape(ncols, nz)
    LS2d = LS.reshape(ncols, nz)
    bld  = np.full((ncols,), np.nan)
    Rib  = np.full((ncols, nz), np.nan)
    for i in range(ncols):
        bld[i], Rib[i,:] = get_bld_Rib_1d(zC, zF, dzF, idxR, b2d[i], u2d[i], v2d[i], NN2d[i], LS2d[i],
                                          ustar=ustar, B0=B0, Ribc=Ribc, rsl=rsl)
    if save_Rib:
        return bld.reshape(leading), Rib.reshape(leading+(nz,))
    else:
        return bld.reshape(leading)


def get_bld_Rib(zF, dzF, b, u, v, NN, attrs, LS=None, Ribc=0.3, save_Rib=False):
    ustar = attrs['ustar'].item()
    B0    = attrs['B₀'].item()
    # dzC   = xr.DataArray(np.concatenate(([np.nan], b.zC.diff('zC').data, [np.nan])), dims=['zF']).assign_coords(zF=zF)
    idxR  = xr.DataArray(get_zref_idx_for_Rib(b.zC, zF, rsl=0.1), dims=['zC']).assign_coords(zC=b.zC)
    LS    = xr.zeros_like(u) if LS is None else LS # lateral shear
    Odims = [[], ['zC']] if save_Rib else [[]]
    Odtypes = [float, float] if save_Rib else [float]
    return xr.apply_ufunc(get_bld_Rib_ufunc, b.zC, zF, dzF, idxR, b, u, v, NN, LS,
                          input_core_dims=[['zC'], ['zF'], ['zC'], ['zC'], ['zC'], ['zC'], ['zC'], ['zC'], ['zC']],
                          output_core_dims=Odims,
                          output_dtypes=Odtypes,
                          kwargs=dict(Ribc=Ribc, ustar=ustar, B0=B0, save_Rib=save_Rib),
                          dask='parallelized',
                          vectorize=False)


def get_mld_ufunc(b, z, criteria=5.4e-6):
    if isinstance(criteria, str) and criteria.startswith('max_over'):
        threshold = float(criteria.split('_')[-1])
        idx_mld = arg_local_max_last(b, threshold=threshold)#np.argmax(b)
        mld = -z[idx_mld]
    else:
        if criteria == '5-percent':
            rtau = 0.05
            delb = b - rtau
        elif criteria == 'critical':
            delb = b - 1e-8
        elif criteria == 0:
            delb = b - 0
        else:
            delb = b - (b[-1] - criteria)

        if np.all(delb >= 0) or (delb[-1] <= 0) or np.all(np.isnan(b)):
            mld = np.nan
        else:
            last_idx = np.max(np.where(delb <= 0))
            crossing = range(last_idx, (last_idx + 2))
            f = interpolate.interp1d(delb[crossing], z[crossing], assume_sorted=True)
            mld = -f(0)
    return mld


def get_Ri_ufunc(zC, zF, dzC, dzF, idxR, b, u, v, ustar, B0, Ribc=0.3, rsl=0.1):
    Rib = np.full_like(zC, np.nan)
    Rig = np.full_like(zF, np.nan)
    NN  = np.full_like(zF, np.nan)
    SS  = np.full_like(zF, np.nan)

    # dzC      = zC[1:] - zC[:-1]
    NN[1:-1] = (b[1:] - b[:-1]) / dzC[1:-1]
    NN[0]    = NN[1]
    NN[-1]   = 0
    SS[1:-1] = ((u[1:] - u[:-1]) / dzC[1:-1])**2 + ((v[1:] - v[:-1]) / dzC[1:-1])**2
    SS[0]    = SS[1]
    SS[-1]   = SS[-2]
    NNmax    = np.maximum(NN[:-1], NN[1:])

    bref  = b[-1]
    uref  = u[-1]
    vref  = v[-1]
    nz    = zC.size

    bdzF_cumsum = reversed_cumsum(b*dzF)
    udzF_cumsum = reversed_cumsum(u*dzF)
    vdzF_cumsum = reversed_cumsum(v*dzF)

    # bulk Richardson number (column sampling approach, see Griffies et al. 2015)
    for k in reversed(range(nz)):
        d = -zC[k]
        kref = idxR[k]
        slt_of_d = rsl * d
        # update reference value
        if kref < (nz - 1):
            partial_cell_thickness = slt_of_d + zF[kref+1]
            bref = (b[kref] * partial_cell_thickness + bdzF_cumsum[kref+1]) / slt_of_d
            uref = (u[kref] * partial_cell_thickness + udzF_cumsum[kref+1]) / slt_of_d
            vref = (v[kref] * partial_cell_thickness + vdzF_cumsum[kref+1]) / slt_of_d

        duv2   = (uref - u[k])**2 + (vref - v[k])**2
        ws     = get_kpp_w_scale(1, d, ustar, B0)
        vt2    = get_vt2(d, ws, NNmax[k], Ribc=Ribc, rsl=rsl)
        Duv2   = duv2 + vt2
        Rib[k] = (d - slt_of_d/2) * (bref - b[k]) / Duv2

    # gradient Richardson number
    tmp       = NN / (SS + 1e-14)
    kernel    = np.array([1, 2, 1]) / 4
    Rig[1:-1] = np.convolve(tmp, kernel, mode='valid')
    Rig[0]    = Rig[1]
    Rig[-1]   = 0
    return Rib, Rig


def get_bld_ufunc(Rib, z, Ribc=0.3):
    d = np.abs(z)
    if np.nanmin(Rib) > Ribc:
        bld = d[-1]
    elif np.nanmax(Rib) < Ribc:
        bld = np.nan
    else:
        first_idx = np.max(np.where((Rib - Ribc) >= 0)) # z-coord is from bottom to surface
        # stencil = range(first_idx, first_idx + 3) # first cell where Rib > Ribc, and the two cells above that cell
        # f = interpolate.interp1d(Rib[stencil], d[stencil], kind='quadratic', assume_sorted=False) # can lead to absurd bld if the curvature is large
        # stencil = range(first_idx, first_idx + 2) # first cell where Rib > Ribc, and the one above that cell
        # f = interpolate.interp1d(Rib[stencil], d[stencil], kind='linear', assume_sorted=False)
        # bld = f(Ribc)
        bld = linear_interp_zero_crossing(Rib[first_idx] - Ribc, Rib[first_idx+1] - Ribc, d[first_idx], d[first_idx+1])
    return bld


def get_bld(Rib, Ribc=0.3):
    return xr.apply_ufunc(get_bld_ufunc, Rib, Rib.zC,
                          input_core_dims=[['zC'], ['zC']],
                          output_core_dims=[[]],
                          output_dtypes=[float],
                          kwargs=dict(Ribc=Ribc),
                          dask='parallelized',
                          vectorize=True)


def kpp_interior_shear_mixing(Rig, nu0=5e-3, Rig0=0.7, p1=3):
    nu = np.full_like(Rig, np.nan)
    idx_static_unstable =  Rig <  0
    idx_shear_unstable  = (Rig >= 0) & (Rig <= Rig0)
    idx_stable          =  Rig >  Rig0
    nu[idx_static_unstable] = nu0
    nu[idx_shear_unstable]  = nu0 * (1 - (Rig[idx_shear_unstable]/Rig0)**2)**p1
    nu[idx_stable]          = 0
    return nu


def kpp_boundary_layer_mixing(sigma, h, ustar, B0, Kint_h, match_interior_K=True, a1=1, dGds_1=0):
    if match_interior_K:
        ws_1 = get_kpp_w_scale(1, h, ustar, B0)
        G_1  = Kint_h / h / ws_1
    else:
        G_1 = 0
    a2 = -2*a1 + 3*G_1 - dGds_1
    a3 =    a1 - 2*G_1 + dGds_1
    G  = a1*sigma + a2*sigma**2 + a3*sigma**3
    ws = get_kpp_w_scale(sigma, h, ustar, B0)
    Kbl = h*ws*G
    return Kbl


def get_kpp_K_ufunc(zC, zF, Rig, h, ustar, B0, match_interior_K=True, enhanced_K=True, vonK=0.4, rsl=0.1):
    K  = np.full_like(zF, np.nan)
    cs = 98.96
    wstar = (h*B0)**(1/3)
    sigma = np.abs(zF) / h
    sigma[sigma > 1] = 0
    a1     = 1
    dGds_1 = 0

    # interior diffusivity only accounts for shear mixing
    Kint   = kpp_interior_shear_mixing(Rig)
    Kint_h = np.interp(-h, zF, Kint) # linear interpolation
    Kbl    = kpp_boundary_layer_mixing(sigma, h, ustar, B0, Kint_h, match_interior_K=match_interior_K, a1=a1, dGds_1=dGds_1)

    # index of cell face right above the boundary layer depth
    kFup = np.min(np.where((zF + h) > 0))
    # index of cell center right above the boundary layer depth
    kCup = np.min(np.where((zC + h) > 0))
    if enhanced_K: # see Appendix D of Large et al. 1994
        sigma_kCup = np.abs(zC[kCup]) / h
        Kbl_kCup = kpp_boundary_layer_mixing(sigma_kCup, h, ustar, B0, Kint_h, match_interior_K=match_interior_K, a1=a1, dGds_1=dGds_1)
        delta  = (zC[kCup] + h) / (zC[kCup] - zC[kCup-1])
        if kFup > kCup: # boundary layer depth in lower half of the cell
            kFup  = kFup -1
            Kstar = (1 - delta)**2 * Kbl_kCup + delta**2 * Kint[kFup]
        else: # boundary layer depth in upper half of the cell
            Kstar = (1 - delta)**2 * Kbl_kCup + delta**2 * Kbl[kFup]
        Gamma = (1 - delta) * Kint[kFup] + delta * Kstar
        K[:kFup]   = Kint[:kFup]
        K[kFup]    = Gamma
        K[kFup+1:] = Kbl[kFup+1:]
    else:
        K[:kFup] = Kint[:kFup]
        K[kFup:] = Kbl[kFup:]
    Kc = np.interp(zC, zF, K)
    return Kc


def get_kpp_K(ds, domain_ave=False):
    if domain_ave:
        vlist = ['⟨bᵝ⟩ₐ', '⟨uᵢ⟩ₐ', 'zF']
        dss = ds[vlist].mean(['xC'], keep_attrs=True)
        dss['Rib'], dss['Rig'] = get_Ri(dss)
        dss['bld'] = get_bld(dss)
    else:
        dss = ds
    return xr.apply_ufunc(get_kpp_K_ufunc, dss.zC, dss.zF, dss.Rig, dss.bld, dss.ustar, dss.attrs['B₀'],
                          input_core_dims=[['zC'], ['zF'], ['zF'], [], [], []],
                          output_core_dims=[['zC']],
                          output_dtypes=[float],
                          vectorize=True,
                          dask='parallelized')
