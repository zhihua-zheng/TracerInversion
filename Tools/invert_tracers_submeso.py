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


def construct_fluxes(udata, cdata):
    udim = list(set(udata.dims).difference(set(cdata.dims)))[0]
    cdim = list(set(cdata.dims).difference(set(udata.dims)))[0]
    uvec = udata.expand_dims(dim=dict(d=[1])).transpose(..., udim, 'd')
    cvec = cdata.expand_dims(dim=dict(d=[1])).transpose(..., 'd', cdim)
    return xr.apply_ufunc(np.matmul, uvec, cvec,
                          input_core_dims=[[udim, 'd'], ['d', cdim]],
                          output_core_dims=[[udim, cdim]],
                          dask='parallelized')


def condense(ds, vlist, varname, dimname='α', indices=None):
    """
    Condense variables in `vlist` into one variable named `varname`.
    In the process, individual variables in `vlist` are removed from `ds`.
    """
    if indices == None:
        indices = range(1, len(vlist)+1)

    tmp = ds[vlist].to_array(dim=dimname).assign_coords({dimname: indices})
    tmp.attrs = {}
    ds[varname] = tmp
    ds = ds.drop_vars(vlist)
    return ds


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


def decomposeSA(M):
    """
    Decomposes a matrix into symmetric and antisymmetric parts
    Transpose is only conducted for the last 2 axes
    """
    M_transposed = np.moveaxis(M, -2, -1)
    S = 0.5 * (M + M_transposed)
    A = 0.5 * (M - M_transposed)
    return np.stack((S, A), axis=-1)


def cmod(a, b):
    result = a % b
    return b if result == 0 else result


def redefine_flux_gradient(ds, time_average, ensemble_average, time_average_window=6):
    """
    Apply time or ensemble average to fields and adjust the eddy flux accordingly
    time_average_window has unit of hours
    """
    if time_average:
        redefine_subm = True
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
        dsp = condense(dsp, vlist=['⟨u⟩ₐ', '⟨w⟩ₐ'], varname='⟨uᵢ⟩ₐ', dimname='i', indices=[1, 3]).chunk(i=-1)
        subm_dsm['⟨uᵢˢcᵅˢ⟩ₐ'] = subm_dsm['⟨uᵢˢcᵅˢ⟩ₐ'] + construct_fluxes(dsp['⟨uᵢ⟩ₐ'], dsp['⟨cᵅ⟩ₐ']).mean(rdims)
        subm_dsm['⟨uᵢˢbᵝˢ⟩ₐ'] = subm_dsm['⟨uᵢˢbᵝˢ⟩ₐ'] + construct_fluxes(dsp['⟨uᵢ⟩ₐ'], dsp['⟨bᵝ⟩ₐ']).mean(rdims)
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

    n_ensemble        = 5
    subset_tracers    = 6
    add_sgs_fluxes    = False
    normalize_tracers = True
    time_average      = False
    ensemble_average  = True

    # read data
    ds = []
    for n in range(n_ensemble):
        cname = args.cname[:14] + f'_em0{n+1}' + args.cname[14:]
        tmps = xr.open_dataset(data_dir+cname+'_submeso_fluxes.nc').chunk(time='auto')
        tmps.close()
        ds.append(tmps)
    ds = xr.concat(ds, dim=xr.DataArray(range(1,6)[:n_ensemble], dims='em'))
    ds.coords['em'].attrs = {'long_name': 'Ensemble member index'}

    ds = redefine_flux_gradient(ds, time_average, ensemble_average)

    #ds['Rib'], ds['Rig'] = get_Ri(ds)
    ds['mld'] = get_mld(ds, cvar='⟨bᵝ⟩ₐ')
    #ds['bld'] = get_bld(ds)

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
    ds['⟨uᵢˢcᵅˢ⟩ₐ'] = ds['⟨uᵢˢcᵅˢ⟩ₐ'].transpose(..., 'i', 'α')
    ds['∇ⱼ⟨cᵅ⟩ₐ']   = ds['∇ⱼ⟨cᵅ⟩ₐ'].transpose(..., 'j', 'α')

    if normalize_tracers:
        #norm_mag = 1
        #mask_lcf = np.sqrt((ds['⟨uᵢˢcᵅˢ⟩ₐ']**2).sum('i')) > 0.1*np.sqrt((ds['⟨uᵢˢcᵅˢ⟩ₐ']**2).sum('i')).max(['xC', 'zC'])
        #norm_mag = np.sqrt((ds['∇ⱼ⟨cᵅ⟩ₐ']**2).sum('j').where(mask_lcf).median(['xC', 'zC']))
        #norm_mag = np.sqrt((ds['∇ⱼ⟨cᵅ⟩ₐ']**2).sum('j')).median(['xC', 'zC']) # using mean gradient gives bad results
        norm_mag = np.sqrt((ds['⟨uᵢˢcᵅˢ⟩ₐ']**2).sum('i')).mean(['xC', 'zC'])
        ds['⟨uᵢˢcᵅˢ⟩ₐ|ⁿ'] = ds['⟨uᵢˢcᵅˢ⟩ₐ'] / norm_mag
        ds['∇ⱼ⟨cᵅ⟩ₐ|ⁿ']   = ds['∇ⱼ⟨cᵅ⟩ₐ'] / norm_mag

    # invert gradient matrix using a Pseudo-Inverse algorithm
    ds = ds.chunk(α=-1, β=-1, m=-1, i=-1, j=-1) if 'm' in ds.dims else ds.chunk(α=-1, β=-1, i=-1, j=-1)
    ds['∇ⱼ⟨cᵅ⟩ₐ|ⁿ ⁻¹'] = xr.apply_ufunc(pinv, ds['∇ⱼ⟨cᵅ⟩ₐ|ⁿ'],
                                        input_core_dims=[['j', 'α']],
                                        output_core_dims=[['α', 'j']],
                                        #kwargs=dict(rcond=1e-9),
                                        dask='parallelized').transpose(..., 'α', 'j')

    # get the transport tensor Rᵢⱼ
    # for the matrix multiplication, the shapes are:
    # (n × t) ⋅ (t × n) = (n × n)
    # (number of dimensions × number of tracers) ⋅ (number of tracers × number of dimensions)
    ds['Rᵢⱼ'] = -xr.apply_ufunc(np.matmul, ds['⟨uᵢˢcᵅˢ⟩ₐ|ⁿ'], ds['∇ⱼ⟨cᵅ⟩ₐ|ⁿ ⁻¹'],
                                input_core_dims=[['i', 'α'], ['α', 'j']],
                                output_core_dims=[['i', 'j']],
                                dask='parallelized')

    # get the symmetric and antisymmetric part
    SA = xr.apply_ufunc(decomposeSA, ds['Rᵢⱼ'],
                        input_core_dims=[['i', 'j']],
                        output_core_dims=[['i', 'j', 'p']],
                        output_dtypes=float,
                        dask_gufunc_kwargs=dict(output_sizes={'p': 2}),
                        dask='parallelized')
    ds['Sᵢⱼ'] = SA.isel(p=0)
    ds['Aᵢⱼ'] = SA.isel(p=1)

    # get eigenvalues of the symmetric tensor
    ds['Kappa'], ds['Kappa_vec'] = xr.apply_ufunc(np.linalg.eigh, ds['Sᵢⱼ'],
                                                  input_core_dims=[['i', 'j']],
                                                  output_core_dims=[['k'], ['kv', 'k']],
                                                  dask_gufunc_kwargs=dict(output_sizes={'k': len(ds.i), 'kv': len(ds.i)}),
                                                  dask='parallelized')
    ds = ds.isel(k=slice(None, None, -1)) # np.linalg.eigh returns eigenvalues in ascending order
    ds = ds.assign_coords(k=('k', np.arange(1, len(ds.i)+1)))

    # concatenate used and unused tracers
    ds = concat_tracers_back(ds)
    ds = ds.chunk(α=-1, β=-1, i=-1, j=-1)

    # reconstruct tracer fluxes
    ds['⟨uᵢˢcᵅˢ⟩ₐᵣ'] = -xr.apply_ufunc(np.matmul, ds['Rᵢⱼ'], ds['∇ⱼ⟨cᵅ⟩ₐ'],
                                       input_core_dims=[['i', 'j'], ['j', 'α']],
                                       output_core_dims=[['i', 'α']],
                                       dask='parallelized')

    # reconstruct buoyancy fluxes
    ds['⟨uᵢˢbᵝˢ⟩ₐ']  = ds['⟨uᵢˢbᵝˢ⟩ₐ'].transpose(..., 'i', 'β')
    ds['∇ⱼ⟨bᵝ⟩ₐ']    = ds['∇ⱼ⟨bᵝ⟩ₐ'].transpose(..., 'j', 'β')
    ds['⟨uᵢˢbᵝˢ⟩ₐᵣ'] = -xr.apply_ufunc(np.matmul, ds['Rᵢⱼ'], ds['∇ⱼ⟨bᵝ⟩ₐ'],
                                       input_core_dims=[['i', 'j'], ['j', 'β']],
                                       output_core_dims=[['i', 'β']],
                                       dask='parallelized')

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
    mask_b = np.logical_and(np.abs(ds['⟨uᵢˢbᵝˢ⟩ₐ']) > 0.1*np.abs(ds['⟨uᵢˢbᵝˢ⟩ₐ'].where(roi)).max(['xC', 'zC']), roi)
    mask_c = np.logical_and(np.abs(ds['⟨uᵢˢcᵅˢ⟩ₐ']) > 0.1*np.abs(ds['⟨uᵢˢcᵅˢ⟩ₐ'].where(roi)).max(['xC', 'zC']), roi)

    # compute conditional volume average (only works for evenly spaced grid for now!)
    #ds['⟨uᵢ′b′⟩ᶜˣᶻ']    = ds['⟨uᵢ′b′⟩'].where(mask_b).mean(['xC', 'zC'])
    #ds['⟨uᵢ′b′⟩ᵣᶜˣᶻ']   = ds['⟨uᵢ′b′⟩ᵣ'].where(mask_b).mean(['xC', 'zC'])
    #ds['∇ᵢ⟨uᵢ′b′⟩ᶜˣᶻ']  = ds['∇ᵢ⟨uᵢ′b′⟩'].where(mask_b).mean(['xC', 'zC'])
    #ds['∇ᵢ⟨uᵢ′b′⟩ᵣᶜˣᶻ'] = ds['∇ᵢ⟨uᵢ′b′⟩ᵣ'].where(mask_b).mean(['xC', 'zC'])
    ds['⟨uᵢˢbᵝˢ⟩ₐ_err'] = np.abs((ds['⟨uᵢˢbᵝˢ⟩ₐ'] - ds['⟨uᵢˢbᵝˢ⟩ₐᵣ'])/ds['⟨uᵢˢbᵝˢ⟩ₐ'].where(mask_b)).median(['xC', 'zC'])
    ds['⟨uᵢˢcᵅˢ⟩ₐ_err'] = np.abs((ds['⟨uᵢˢcᵅˢ⟩ₐ'] - ds['⟨uᵢˢcᵅˢ⟩ₐᵣ'])/ds['⟨uᵢˢcᵅˢ⟩ₐ'].where(mask_c)).median(['xC', 'zC'])

    # alternative error metric
    #uwb_rms = np.sqrt( (ds['⟨uᵢˢbᵝˢ⟩ₐ']**2).where(mask_b).mean(['xC', 'zC']) )
    #uwc_rms = np.sqrt( (ds['⟨uᵢˢcᵅˢ⟩ₐ']**2).where(mask_c).mean(['xC', 'zC']) )
    #ds['⟨uᵢˢbᵝˢ⟩ₐ_er'] = np.sqrt( ((ds['⟨uᵢˢbᵝˢ⟩ₐ'] - ds['⟨uᵢˢbᵝˢ⟩ₐᵣ'])**2).where(mask_b).mean(['xC', 'zC']) ) / uwb_rms
    #ds['⟨uᵢˢcᵅˢ⟩ₐ_er'] = np.sqrt( ((ds['⟨uᵢˢcᵅˢ⟩ₐ'] - ds['⟨uᵢˢcᵅˢ⟩ₐᵣ'])**2).where(mask_c).mean(['xC', 'zC']) ) / uwc_rms

    ds['tracer_used'] = ds.α.isin(inversion_tracer_index)
    ds.coords['α'].attrs = {'long_name': 'Used tracer index'}
    ds.coords['β'].attrs = {'long_name': 'Buoyancy index'}
    ds.coords['j'].attrs = {'long_name': 'Gradient direction'}
    ds.coords['i'].attrs = {'long_name': 'Flux direction'}
    ds.coords['k'].attrs = {'long_name': 'Diffusivity index'}

    # save to disk
    fpath = data_dir + args.cname + '_submeso_transport_tensor.nc'
    delayed_nc = ds.to_netcdf(fpath, compute=False, mode='w')
    with ProgressBar():
        #results = delayed_nc.compute()
        results = delayed_nc.compute(num_workers=2, memory_limit='5GB', processes=False)
    
#    with LocalCluster() as cluster, Client(cluster) as client:
#        results = delayed_nc.compute()
    t1 = time.time()
    print(f'Done saving to {fpath}')
    print(f'Computation finished in {((t1-t0)/60):.1f} minutes')


if __name__ == "__main__":
    main()
