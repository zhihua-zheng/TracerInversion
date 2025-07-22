#!/usr/bin/env python3

import numpy as np
import xarray as xr
from scipy import interpolate


def arg_local_max_last(y, threshold):
    """
    https://gist.github.com/ben741/d8c70b608d96d9f7ed231086b237ba6b
    """
    idx_peaks = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]) * (y[1:-1] > threshold))[0] + 1
    return idx_peaks[-1]


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


def get_mld(ds, cvar='⟨bᵝ⟩', dims=['zC']):
    a = ds[cvar]
    if cvar == '⟨bᵝ⟩' or cvar == '⟨bᵝ⟩ₐ':
        a = a.isel(β=0)
        criteria = 5.4e-6 #ds.attrs['N₀²'] * ds.attrs['hᵢ']
    elif cvar == 'dbdz':
        N2_base = ds.attrs['N₁²']
        criteria = f'max_over_{N2_base}'
    elif cvar == 'wuvn':
        criteria = '5-percent'
    elif cvar == 'TKE_eps' or cvar == 'eps':
        criteria = 'critical'
    elif cvar == 'wb':
        criteria = 0
    return xr.apply_ufunc(get_mld_ufunc, a, ds.zC,
                          input_core_dims=[dims, dims],
                          output_core_dims=[[]],
                          output_dtypes=[float],
                          kwargs=dict(criteria=criteria),
                          dask='parallelized',
                          vectorize=True)


def get_kpp_phi(zeta):
    phis = np.full_like(zeta, np.nan)
    idx_stable          = zeta >= 0
    idx_unstable_weak   = (zeta > -1) & (zeta < 0)
    idx_unstable_strong = zeta <= -1
    phis[idx_stable]          = 1 + 5*zeta[idx_stable]
    phis[idx_unstable_weak]   = (1 - 16*zeta[idx_unstable_weak])**(-1/2)
    phis[idx_unstable_strong] = (-28.86 - 98.96*zeta[idx_unstable_strong])**(-1/3)
    return phis#, phim


def get_kpp_w_scale(sigma, bld, ustar, B0, vonk=0.4, rsl=0.1):
    if ustar > 0:
        if B0 <= 0: # stablizing
            sigma_loc = sigma
        else: # destablizing
            sigma_loc = np.minimum(rsl, sigma)
        LObk = -ustar**3 / vonk / B0
        zeta = sigma_loc * bld / LObk
        phis = get_kpp_phi(zeta)
        ws   = vonk * ustar / phis
    elif ustar == 0:
        if B0 <= 0: # stablizing
            ws = np.full_like(sigma, 0)
        else:  # destablizing
            cs = 98.96
            wstar = (bld * B0)**(1/3)
            ws = vonk * (cs*vonk * np.minimum(rsl, sigma))**(1/3) * wstar
    return ws#, wm


def get_vt2(d, ws, NNmax, Ribc=0.3, rsl=0.1, vonk=0.4, beta_T=-0.2, opt='LMD94'):
    Ne = np.sqrt(np.maximum(0, NNmax))
    Cv = 1.7 if Ne > 2e-3 else 2.1 - 200*Ne
    cs = 98.96
    if opt == 'LMD94': # Large et al. 1994
        vt2 = Cv*d*Ne*ws*np.sqrt(-beta_T/cs/rsl) / (vonk**2) / Ribc
    else:
        print(f'option {opt} not supported.')
    #else: # Li & Fox-Kemper 2017
    #    rL  = 1 #(1+0.49*La_sl**(-2))
    #    vt2 = Cv*d*Ne*np.sqrt((0.15*wstar3 + 0.17*ustar**3*rL) / ws) / Ribc
    return np.maximum(1e-10, vt2)


def get_Ri_ufunc(zC, zF, b, u, v, ustar, B0, Ribc=0.3, rsl=0.1):
    Rib = np.full_like(zC, np.nan)
    Rig = np.full_like(zF, np.nan)
    NN  = np.full_like(zF, np.nan)
    SS  = np.full_like(zF, np.nan)

    dzC      = zC[1:] - zC[:-1]
    NN[1:-1] = (b[1:] - b[:-1]) / dzC
    NN[0]    = NN[1]
    NN[-1]   = 0
    SS[1:-1] = ((u[1:] - u[:-1]) / dzC)**2 + ((v[1:] - v[:-1]) / dzC)**2
    SS[0]    = SS[1]
    SS[-1]   = SS[-2]

    bref  = b[-1]
    uref  = u[-1]
    vref  = v[-1]
    depth = np.abs(zC)
    dz = np.diff(zF)
    nz = depth.size

    # bulk Richardson number (column sampling approach, see Griffies et al. 2015)
    for k in reversed(range(nz)):
        d = depth[k]
        slt_of_d = rsl * d
        for kk in reversed(range(k,nz)):
            if (zF[-1] - zF[kk]) >= slt_of_d:
                kref = kk
                break
        # update reference value
        if kref < (nz - 1):
            bref = b[kref] * (slt_of_d + zF[kref+1])
            uref = u[kref] * (slt_of_d + zF[kref+1])
            vref = v[kref] * (slt_of_d + zF[kref+1])
            for kk in reversed(range(kref+1,nz)):
                bref = bref + b[kk]*dz[kk]
                uref = uref + u[kk]*dz[kk]
                vref = vref + v[kk]*dz[kk]
            bref = bref / slt_of_d
            uref = uref / slt_of_d
            vref = vref / slt_of_d

        NNmax  = np.maximum(NN[k], NN[k+1])
        duv2   = (uref - u[k])**2 + (vref - v[k])**2
        ws     = get_kpp_w_scale(1, d, ustar, B0)
        vt2    = get_vt2(d, ws, NNmax, Ribc=Ribc, rsl=rsl)
        Duv2   = duv2 + vt2
        Rib[k] = (d - slt_of_d/2) * (bref - b[k]) / Duv2

    # gradient Richardson number
    tmp       = NN / (SS + 1e-14)
    kernel    = np.array([1, 2, 1]) / 4
    Rig[1:-1] = np.convolve(tmp, kernel, mode='valid')
    Rig[0]    = Rig[1]
    Rig[-1]   = 0
    return Rib, Rig


def get_Ri(ds, Ribc=0.3):
    ustar = ds.attrs['ustar']
    B0    = ds.attrs['B₀']
    dsb   = ds['⟨bᵝ⟩ₐ'].isel(β=0)
    ubar  = ds['⟨uᵢ⟩ₐ'].sel(i=1)
    vbar  = ds['⟨uᵢ⟩ₐ'].sel(i=2)
    return xr.apply_ufunc(get_Ri_ufunc, ds.zC, ds.zF, dsb, ubar, vbar, ustar, B0,
                          input_core_dims=[['zC'], ['zF'], ['zC'], ['zC'], ['zC'], [], []],
                          output_core_dims=[['zC'], ['zF']],
                          output_dtypes=[float, float],
                          kwargs=dict(Ribc=Ribc),
                          dask='parallelized',
                          vectorize=True)


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
        stencil = range(first_idx, first_idx + 2) # first cell where Rib > Ribc, and the one above that cell
        f = interpolate.interp1d(Rib[stencil], d[stencil], kind='linear', assume_sorted=False)
        bld = f(Ribc)
    return bld


def get_bld(ds, Ribc=0.3):
    return xr.apply_ufunc(get_bld_ufunc, ds.Rib, ds.zC,
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


def get_kpp_K_ufunc(zC, zF, Rig, h, ustar, B0, match_interior_K=True, enhanced_K=True, vonk=0.4, rsl=0.1):
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
