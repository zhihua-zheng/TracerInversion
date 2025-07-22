#!/usr/bin/env python3

import os
import re
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from xgcm import Grid
from mixdiag import get_Ri, get_mld, get_bld
from numpy.linalg import pinv
#from scipy.linalg import pinv
from dask.diagnostics import ProgressBar
#from dask_jobqueue import PBSCluster
#from dask.distributed import Client


def startswith_char_num(s: str, char: str) -> bool:
    return bool(re.match(rf'^{re.escape(char)}\d', s))


def separate_tracers(ds, used_basename='c', inversion_tracer_index=np.arange(1, 5),
                     unused_basename='c', unused_indexname='m', unused_index_sup='ᵐ'):
    """
    Separate tracers into ones used in the inversion process and ones unused in the inversion process
    Default to use the same basename
    """
    # Test if `inversion_tracer_index` makes sense
    inversion_tracer_index = inversion_tracer_index.sortby(inversion_tracer_index)
    inversion_tracer_index_in_α = inversion_tracer_index.isin(ds.α)
    if all(inversion_tracer_index_in_α):
        if (len(ds.α) == len(inversion_tracer_index)):
            if all(ds.α == inversion_tracer_index): # If we use all tracers for the reconstruction there's nothing to be done
                return ds
            else:
                raise(ValueError(f"Something's wrong. Maybe repeated values for `inversion_tracer_index`?"))
    else:
        raise(ValueError(f'`inversion_tracer_index` = {inversion_tracer_index} is not contained in α = {ds.α.data}.'))

    # Do what we're here for
    α_used_for_inversion = ds.α.isin(inversion_tracer_index)
    for var in list(ds.keys()):
        if 'α' in ds[var].coords:# and ds[var].ndim > 2:
            newvar = var.replace(used_basename, unused_basename).replace('ᵅ', unused_index_sup)
            ds[newvar] = ds[var].where(np.logical_not(α_used_for_inversion), drop=True).rename(α=unused_indexname)
    return ds.sel(α=inversion_tracer_index)


def concat_tracers_back(ds, used_basename='c', unused_basename='c', unused_indexname='m', unused_index_sup='ᵐ'):
    """
    Merge back tracers that were separated into ones used in the inversion process and ones unused in the inversion process
    """
    if unused_indexname not in ds: # If there's no m that means we used all tracers for the reconstruction and there's nothing to be done
        return ds

    α_original = sorted(np.concatenate([ds.α, ds[unused_indexname]]))
    ds_aux = ds.reindex(α=α_original)
    for var in list(ds.keys()):
        if unused_indexname in ds[var].coords:
            original_var = var.replace(unused_basename, used_basename).replace(unused_index_sup, 'ᵅ')
            ds_aux[original_var] = xr.concat([ds[original_var], ds[var].rename({unused_indexname: 'α'})], dim='α').sortby('α')
            ds_aux = ds_aux.drop_vars(var)
    return ds_aux.drop_vars(unused_indexname)


def cmod(a, b):
    result = a % b
    return b if result == 0 else result


def redefine_flux_gradient(ds, time_average, ensemble_average, time_average_window=6):
    """
    Apply time or ensemble average to fields and adjust the eddy flux accordingly
    time_average_window has unit of hours
    """
    if time_average:
        redefine_turb = True
        avetime = np.timedelta64(time_average_window, 'h')
        dtime = np.timedelta64(int(ds.save_out_interval), 's')
        n_per_window = int(avetime / dtime)
        t_fine, t_coarse = [arr.flatten() for arr in np.meshgrid(range(n_per_window), ds.time.coarsen(time=n_per_window).mean().data)]
        ds = ds.assign_coords(t_fine=('time', t_fine), t_coarse=('time', t_coarse))
        ds = ds.set_index(time=('t_fine', 't_coarse')).unstack('time')
        if ensemble_average:
            print('Averaging over time and ensemble members, and redefining fluxes and gradients')
            rdims = ['t_fine', 'em']
        else:
            print('Averaging over time, and redefining fluxes and gradients')
            rdims = ['t_fine']
    else:
        redefine_turb = False
        if ensemble_average:
            print('Averaging over ensemble members, and redefining submesoscale fluxes and gradients')
            redefine_subm = True
            rdims = ['em']
        else:
            redefine_subm = False

    if redefine_subm:
        subm_var  = ['⟨u⟩ₐ', '⟨w⟩ₐ', '⟨cᵅ⟩ₐ', '⟨bᵝ⟩ₐ', '⟨uᵢˢcᵅˢ⟩ₐ', '⟨uᵢˢbᵝˢ⟩ₐ', '∇ⱼ⟨cᵅ⟩ₐ', '∇ⱼ⟨bᵝ⟩ₐ']
        subm_dsm  = ds[subm_var].mean(rdims)
        subm_mvar = subm_var[:4]
        dsp = ds[subm_mvar] - subm_dsm[subm_mvar]
        dsp = condense(dsp, vlist=['⟨u⟩ₐ', '⟨w⟩ₐ'], varname='⟨uᵢ⟩ₐ', dimname='i', indices=[1, 3])
        subm_dsm['⟨uᵢˢcᵅˢ⟩ₐ'] = subm_dsm['⟨uᵢˢcᵅˢ⟩ₐ'] + (dsp['⟨uᵢ⟩ₐ'] * dsp['⟨cᵅ⟩ₐ']).mean(rdims)
        subm_dsm['⟨uᵢˢbᵝˢ⟩ₐ'] = subm_dsm['⟨uᵢˢbᵝˢ⟩ₐ'] + (dsp['⟨uᵢ⟩ₐ'] * dsp['⟨bᵝ⟩ₐ']).mean(rdims)

        if redefine_turb:
            turb_dsm  = ds.drop_vars(subm_var).mean('t_fine', keep_attrs=True)
            turb_mvar = ['⟨u⟩', '⟨v⟩', '⟨w⟩', '⟨cᵅ⟩', '⟨bᵝ⟩']
            dsp = ds[turb_mvar] - turb_dsm[turb_mvar]
            turb_dsm['⟨wᵗbᵝᵗ⟩'] = turb_dsm['⟨wᵗbᵝᵗ⟩'] + (dsp['⟨w⟩'] * dsp['⟨bᵝ⟩']).mean('t_fine')
            turb_dsm['⟨wᵗcᵅᵗ⟩'] = turb_dsm['⟨wᵗcᵅᵗ⟩'] + (dsp['⟨w⟩'] * dsp['⟨cᵅ⟩']).mean('t_fine')
            turb_dsm['⟨uᵗbᵝᵗ⟩'] = turb_dsm['⟨uᵗbᵝᵗ⟩'] + (dsp['⟨u⟩'] * dsp['⟨bᵝ⟩']).mean('t_fine')
            turb_dsm['⟨uᵗcᵅᵗ⟩'] = turb_dsm['⟨uᵗcᵅᵗ⟩'] + (dsp['⟨u⟩'] * dsp['⟨cᵅ⟩']).mean('t_fine')
            turb_dsm['⟨vᵗbᵝᵗ⟩'] = turb_dsm['⟨vᵗbᵝᵗ⟩'] + (dsp['⟨v⟩'] * dsp['⟨bᵝ⟩']).mean('t_fine')
            turb_dsm['⟨vᵗcᵅᵗ⟩'] = turb_dsm['⟨vᵗcᵅᵗ⟩'] + (dsp['⟨v⟩'] * dsp['⟨cᵅ⟩']).mean('t_fine')
            turb_dsm['⟨uᵗuᵗ⟩'] = turb_dsm['⟨uᵗuᵗ⟩'] + (dsp['⟨u⟩'] * dsp['⟨u⟩']).mean('t_fine')
            turb_dsm['⟨vᵗvᵗ⟩'] = turb_dsm['⟨vᵗvᵗ⟩'] + (dsp['⟨v⟩'] * dsp['⟨v⟩']).mean('t_fine')
            turb_dsm['⟨wᵗwᵗ⟩'] = turb_dsm['⟨wᵗwᵗ⟩'] + (dsp['⟨w⟩'] * dsp['⟨w⟩']).mean('t_fine')
            turb_dsm['⟨wᵗuᵗ⟩'] = turb_dsm['⟨wᵗuᵗ⟩'] + (dsp['⟨w⟩'] * dsp['⟨u⟩']).mean('t_fine')
            turb_dsm['⟨wᵗvᵗ⟩'] = turb_dsm['⟨wᵗvᵗ⟩'] + (dsp['⟨w⟩'] * dsp['⟨v⟩']).mean('t_fine')
            ds = xr.merge([turb_dsm, subm_dsm]).rename({'t_coarse': 'time'})
        else:
            ds = xr.merge([ds.drop_vars(subm_var), subm_dsm])
    return ds


def main():
    # process input arguments
    parser = argparse.ArgumentParser(description="""Calculate transport tensor.""")
    parser.add_argument('-c', '--case', action='store', dest='cname', metavar='CASENAME', help='simulation case name')
    parser.add_argument('--version', action='version', version='%(prog)s: 1.0')
    args = parser.parse_args()

    # check input
    if not args.cname:
        print('Simulation case name is required. Stop.\n')
        parser.print_help()
        sys.exit(1)

    t0 = time.time()

    # specify file path
    if sys.platform == 'linux' or sys.platform == 'linux2':
        data_dir = '/glade/derecho/scratch/zhihuaz/TracerInversion/Output/'
    elif sys.platform == 'darwin':
        data_dir = '/Users/zhihua/Documents/Work/Research/Projects/TRACE-SEAS/TracerInversion/Data/'
    else:
        print('OS not supported.')

    n_ensemble        = 1
    subset_tracers    = 6
    add_sgs_fluxes    = False
    normalize_tracers = True
    time_average      = False
    ensemble_average  = False

    # read data
    ds = []
    for n in range(n_ensemble):
        cname = args.cname[:14] + f'_em0{n+1}' + args.cname[14:]
        tmpf = xr.open_dataset(data_dir+cname+'_finescale_fluxes.nc').chunk(time='auto')
        tmpf.close()
        ds.append(tmpf)
    ds = xr.concat(ds, dim=xr.DataArray(range(1,6)[:n_ensemble], dims='em'))
    ds.coords['em'].attrs = {'long_name': 'Ensemble member index'}

    # ds = redefine_flux_gradient(ds, time_average, ensemble_average)
    ds['⟨uᵢᵗbᵝᵗ⟩ₐ'] = ds['⟨uᵢᵗbᵝᵗ⟩'].mean('yC')
    ds['⟨uᵢᵗcᵅᵗ⟩ₐ'] = ds['⟨uᵢᵗcᵅᵗ⟩'].mean('yC')
    ds['∇ⱼ⟨bᵝ⟩ₐ']   = ds['∇ⱼ⟨bᵝ⟩'].mean('yC')
    ds['∇ⱼ⟨cᵅ⟩ₐ']   = ds['∇ⱼ⟨cᵅ⟩'].mean('yC')
    ds['⟨uᵢ⟩ₐ']     = ds['⟨uᵢ⟩'].mean('yC')
    ds['⟨bᵝ⟩ₐ']     = ds['⟨bᵝ⟩'].mean('yC')
    
    ds['Rib'], ds['Rig'] = get_Ri(ds)
    ds['mld'] = get_mld(ds, cvar='⟨bᵝ⟩ₐ')
    ds['bld'] = get_bld(ds)

    wct  = ds['⟨uᵢᵗcᵅᵗ⟩ₐ'].sel(i=3).expand_dims(dim=dict(μi=[1]))
    dcdz = ds['∇ⱼ⟨cᵅ⟩ₐ'].sel(j=3).expand_dims(dim=dict(μj=[1]))
    dbdz = ds['∇ⱼ⟨bᵝ⟩ₐ'].sel(j=3).expand_dims(dim=dict(μj=[1]))

    # only use a subset of tracers for inversion
    if (type(subset_tracers) is float) or (type(subset_tracers) is int):
        if subset_tracers > 0:
            inversion_tracer_index = ds.α[:subset_tracers]
        elif subset_tracers < 0:
            inversion_tracer_index = ds.α[subset_tracers:]
        else:
            inversion_tracer_index = ds.α
    else:
        inversion_tracer_index = ds.α[subset_tracers]

    ds = separate_tracers(ds, inversion_tracer_index=inversion_tracer_index,
                          used_basename='c', unused_indexname='m', unused_index_sup='ᵐ')

    if add_sgs_fluxes:
        print('Adding SGS fluxes')
        ds['⟨uᵢ′cᵅ′⟩'] += ds['⟨Fᵢcᵅ⟩']
        ds['⟨uᵢ′bᵝ′⟩'] += ds['⟨Fᵢbᵝ⟩']

    # reshaping matrices
    # following Bachman et al. (2015) the shapes are
    # ⟨uᵢ′cᵅ′⟩   n × t (number of dimensions × number of tracers)
    # ∇ⱼ⟨cᵅ⟩     n × t (number of dimensions × number of tracers)
    # Rᵢⱼ        n × n (number of dimensions × number of dimensions)

    if normalize_tracers:
        #norm_mag = 1
        #mask_lcf = np.sqrt((ds['⟨uᵢˢcᵅˢ⟩ₐ']**2).sum('i')) > 0.1*np.sqrt((ds['⟨uᵢˢcᵅˢ⟩ₐ']**2).sum('i')).max(['xC', 'zC'])
        #norm_mag = np.sqrt((ds['∇ⱼ⟨cᵅ⟩ₐ']**2).sum('j').where(mask_lcf).median(['xC', 'zC']))
        #norm_mag = np.sqrt((ds['∇ⱼ⟨cᵅ⟩ₐ']**2).sum('j')).median(['xC', 'zC']) # using mean gradient gives bad results
        #norm_mag = np.sqrt((ds['⟨uᵢˢcᵅˢ⟩ₐ']**2).sum('i')).mean(['xC', 'zC'])
        norm_mag = np.sqrt(ds['∇ⱼ⟨cᵅ⟩ₐ'].sel(j=3)**2).mean(['xC', 'zC'])
        wctn  = wct / norm_mag
        dcdzn = dcdz / norm_mag

    # invert gradient matrix using a Pseudo-Inverse algorithm
    ds = ds.chunk(α=-1, β=-1, m=-1, i=-1, j=-1) if 'm' in ds.dims else ds.chunk(α=-1, β=-1, i=-1, j=-1)
    dcdzn_pinv = xr.apply_ufunc(pinv, dcdzn,
                                input_core_dims=[['μj', 'α']],
                                output_core_dims=[['α', 'μj']],
                                dask='parallelized').transpose(..., 'α', 'μj')

    # get the transport tensor Rᵢⱼ
    # for the matrix multiplication, the shapes are:
    # (n × t) ⋅ (t × n) = (n × n)
    # (number of dimensions × number of tracers) ⋅ (number of tracers × number of dimensions)
    ds['K'] = -xr.apply_ufunc(np.matmul, wctn, dcdzn_pinv,
                              input_core_dims=[['μi', 'α'], ['α', 'μj']],
                              output_core_dims=[['μi', 'μj']],
                              dask='parallelized')

    # concatenate used and unused tracers
    ds = concat_tracers_back(ds)
    ds = ds.chunk(α=-1, β=-1, i=-1, j=-1)

    # reconstruct tracer fluxes
    ds['⟨wᵗcᵅᵗ⟩ₐᵣ'] = -xr.apply_ufunc(np.matmul, ds['K'], dcdz,
                                     input_core_dims=[['μi', 'μj'], ['μj', 'α']],
                                     output_core_dims=[['μi', 'α']],
                                     dask='parallelized')

    # reconstruct buoyancy fluxes
    ds['⟨wᵗbᵝᵗ⟩ₐᵣ'] = -xr.apply_ufunc(np.matmul, ds['K'], dbdz,
                                     input_core_dims=[['μi', 'μj'], ['μj', 'β']],
                                     output_core_dims=[['μi', 'β']],
                                     dask='parallelized')
    ds = ds.squeeze(['μi', 'μj']).drop_vars(['μi', 'μj'])

    # compute flux divergence
    # ds = ds.chunk(xC=-1, zC=-1, time='auto')
    # ds['∂x⟨u′b′⟩']  = ds['⟨uᵢ′b′⟩'].sel(i=1).differentiate('xC')
    # ds['∂z⟨w′b′⟩']  = ds['⟨uᵢ′b′⟩'].sel(i=3).differentiate('zC')
    # ds['∇ᵢ⟨uᵢ′b′⟩'] = ds['∂x⟨u′b′⟩'] + ds['∂z⟨w′b′⟩']

    # ds['∂x⟨u′b′⟩ᵣ']  = ds['⟨uᵢ′b′⟩ᵣ'].sel(i=1).differentiate('xC')
    # ds['∂z⟨w′b′⟩ᵣ']  = ds['⟨uᵢ′b′⟩ᵣ'].sel(i=3).differentiate('zC')
    # ds['∇ᵢ⟨uᵢ′b′⟩ᵣ'] = ds['∂x⟨u′b′⟩ᵣ'] + ds['∂z⟨w′b′⟩ᵣ']

    # create mask based on region of interest and flux magnitude
    roi = ds.zC < -40 #np.logical_and(ds.xC >= -20e3, ds.xC <= 20e3)
    turb_mask_c = np.logical_and(np.abs(ds['⟨uᵢᵗcᵅᵗ⟩ₐ'].sel(i=3)) > 0.1*np.abs(ds['⟨uᵢᵗcᵅᵗ⟩ₐ'].sel(i=3).where(roi)).max(['xC', 'zC']), roi)
    turb_mask_b = np.logical_and(np.abs(ds['⟨uᵢᵗbᵝᵗ⟩ₐ'].sel(i=3)) > 0.1*np.abs(ds['⟨uᵢᵗbᵝᵗ⟩ₐ'].sel(i=3).where(roi)).max(['xC', 'zC']), roi)

    # compute conditional volume average (only works for evenly spaced grid for now!)
    ds['⟨wᵗcᵅᵗ⟩ₐ_err'] = np.abs((ds['⟨uᵢᵗcᵅᵗ⟩ₐ'].sel(i=3) - ds['⟨wᵗcᵅᵗ⟩ₐᵣ'])/ds['⟨uᵢᵗcᵅᵗ⟩ₐ'].sel(i=3).where(turb_mask_c)).median(['xC', 'zC'])
    ds['⟨wᵗbᵝᵗ⟩ₐ_err'] = np.abs((ds['⟨uᵢᵗbᵝᵗ⟩ₐ'].sel(i=3) - ds['⟨wᵗbᵝᵗ⟩ₐᵣ'])/ds['⟨uᵢᵗbᵝᵗ⟩ₐ'].sel(i=3).where(turb_mask_b)).median(['xC', 'zC'])

    # alternative error metric
    # turb_wc_rms = np.sqrt( (ds['⟨uᵢᵗcᵅᵗ⟩'].sel(ti=3)**2).where(turb_mask_c).mean(['xC', 'yC', 'zC']) )
    #turb_wc_rms = np.sqrt( (ds['⟨wᵗcᵅᵗ⟩']**2).where(turb_mask_c).mean(['xC', 'yC', 'zC']) )
    #turb_wb_rms = np.sqrt( (ds['⟨wᵗbᵝᵗ⟩']**2).where(turb_mask_b).mean(['xC', 'yC', 'zC']) )
    #ds['⟨wᵗcᵅᵗ⟩_er'] = np.sqrt( ((ds['⟨uᵢᵗcᵅᵗ⟩'].sel(ti=3) - ds['⟨wᵗcᵅᵗ⟩ᵣ'])**2).where(turb_mask_c).mean(['xC', 'yC', 'zC']) ) / turb_wc_rms
    #ds['⟨wᵗcᵅᵗ⟩_er'] = np.sqrt( ((ds['⟨wᵗcᵅᵗ⟩'] - ds['⟨wᵗcᵅᵗ⟩ᵣ'])**2).where(turb_mask_c).mean(['xC', 'yC', 'zC']) ) / turb_wc_rms
    #ds['⟨wᵗbᵝᵗ⟩_er'] = np.sqrt( ((ds['⟨wᵗbᵝᵗ⟩'] - ds['⟨wᵗbᵝᵗ⟩ᵣ'])**2).where(turb_mask_b).mean(['xC', 'yC', 'zC']) ) / turb_wb_rms

    ds['tracer_used'] = ds.α.isin(inversion_tracer_index)
    ds.coords['α'].attrs = {'long_name': 'Used tracer index'}
    ds.coords['β'].attrs = {'long_name': 'Buoyancy index'}
    ds.coords['j'].attrs = {'long_name': 'Gradient direction'}
    ds.coords['i'].attrs = {'long_name': 'Flux direction'}

    # save to disk
    fpath = data_dir + args.cname + '_finescale_transport_tensor.nc'
    delayed_nc = ds.to_netcdf(fpath, compute=False, mode='w')
    with ProgressBar():
        #results = delayed_nc.compute()
        results = delayed_nc.compute(num_workers=16, memory_limit='5GB')
    
#    with LocalCluster() as cluster, Client(cluster) as client:
#        results = delayed_nc.compute()
    t1 = time.time()
    print(f'Done saving to {fpath}')
    print(f'Computation finished in {((t1-t0)/60):.1f} minutes')


if __name__ == "__main__":
    main()
