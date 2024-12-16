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
from numpy.linalg import pinv
#from scipy.linalg import pinv
from dask.diagnostics import ProgressBar
#from dask_jobqueue import PBSCluster
#from dask.distributed import Client


def startswith_char_num(s: str, char: str) -> bool:
    return bool(re.match(rf'^{re.escape(char)}\d', s))


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
    #α_used_for_inversion = ds.α.isin(inversion_tracer_index)
    for var in list(ds.keys()):
        if 'α' in ds[var].coords:
            newvar = var.replace(used_basename, unused_basename).replace('ᵅ', unused_index_sup)
            ds[newvar] = ds[var].where(np.logical_not(ds.tracer_used), drop=True).rename(α=unused_indexname)
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


def sort_tracer_gradients(G, F, invert_tracers=6):
    G_magnitude = np.sqrt((G**2).sum(axis=-2, keepdims=True))
    ind = np.argsort(-G_magnitude, axis=-1) # descending
    G_sorted = np.take_along_axis(G, ind, axis=-1)
    F_sorted = np.take_along_axis(F, ind, axis=-1)
    return G_sorted[..., :invert_tracers], F_sorted[..., :invert_tracers]


def main():
    # process input arguments
    parser = argparse.ArgumentParser(description="""
            Calculate transport tensor.""")
    parser.add_argument('-c', '--case', action='store', dest='cname',
            metavar='CASENAME', help='simulation case name')
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

    # read data
    dsa = xr.open_dataset(data_dir+args.cname+'_averages.nc').chunk(time='auto').isel(time=slice(1,None))
    dsa.close()

    if args.cname.startswith('c'):
        add_background_tracer = False

    ## construct coordinates
    #periodic_coords = {dim : dict(left=f'{dim}F', center=f'{dim}C') for dim in 'xy'}
    #bounded_coords = {dim : dict(outer=f'{dim}F', center=f'{dim}C') for dim in 'z'}
    #coords = {dim : periodic_coords[dim] if tpl=='P' else bounded_coords[dim] for dim, tpl in zip('xyz', 'PPN')}
    #grid = Grid(dsa, coords=coords)

    subset_tracers = 8
    invert_tracers = 6
    add_sgs_fluxes = False
    normalize_tracers = True
    dsa_vlist = list(dsa.keys())

    # add background tracer field
    if add_background_tracer:
        vars_dcdx = [f'dc{i}dx_ym' for i in range(1, dsa.n_tracers)]
        vars_dcdz = [f'dc{i}dz_ym' for i in range(1, dsa.n_tracers)]
        for i,v in enumerate(vars_dcdx):
            ic = i+1
            imod = cmod(ic, dsa.n_per_set)
            if (ic % 2) != 0:
                if ic != (dsa.n_tracers - 1):
                    var_cbkg_grad = f'CbakG{ic}'
                    dsa[v] = dsa[v] + dsa[var_cbkg_grad]
                else:
                    dsa[v] = dsa[v] - dsa.attrs['M²']

        for i,v in enumerate(vars_dcdz):
            ic = i+1
            imod = cmod(ic, dsa.n_per_set)
            var_c = f'c{ic}_ym'
            if ic != (dsa.n_tracers - 1):
                var_cbkg = f'Cbak{ic}'
                dsa[var_c] = dsa[var_c] + dsa[var_cbkg]
            else:
                dsa[var_c] = dsa[var_c] + dsa.Bbak

            if (ic % 2) == 0:
                var_cbkg_grad = f'CbakG{ic}'
                dsa[v] = dsa[v] + dsa[var_cbkg_grad]

    # merge tracer statistics into one with counter α
    #get_tracer_index = lambda x: int(x.split('′')[1][1:])
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (startswith_char_num(v, 'c') and v.endswith('ym'))],
                                     key=lambda x: int(x.split('_')[0][1:])),
                   varname='⟨cᵅ⟩', dimname='α')
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (startswith_char_num(v, 'dc') and v.endswith('dx_ym'))],
                                     key=lambda x: int(x.split('d')[1][1:])),
                   varname='d⟨cᵅ⟩dx', dimname='α')
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (startswith_char_num(v, 'dc') and v.endswith('dz_ym'))],
                                     key=lambda x: int(x.split('d')[1][1:])),
                   varname='d⟨cᵅ⟩dz', dimname='α')
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (startswith_char_num(v, 'u′c') and v.endswith('ym'))],
                                     key=lambda x: int(x.split('′')[1][1:])),
                   varname='⟨u′cᵅ′⟩', dimname='α')
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (startswith_char_num(v, 'w′c') and v.endswith('ym'))],
                                     key=lambda x: int(x.split('′')[1][1:])),
                   varname='⟨w′cᵅ′⟩', dimname='α')
    #dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (startswith_char_num(v, 'uc') and v.endswith('sgs_ym'))],
    #                                 key=lambda x: int(x.split('_')[0][2:])),
    #               varname='⟨Fcᵅ_x⟩', dimname='α')
    #dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (startswith_char_num(v, 'wc') and v.endswith('sgs_ym'))],
    #                                 key=lambda x: int(x.split('_')[0][2:])),
    #               varname='⟨Fcᵅ_z⟩', dimname='α')

    # merge all directions into one
    dir_indices = [1, 3]
    dsa = condense(dsa, vlist=['d⟨cᵅ⟩dx', 'd⟨cᵅ⟩dz'], varname='∇ⱼ⟨cᵅ⟩',   dimname='j', indices=dir_indices)
    dsa = condense(dsa, vlist=['⟨u′cᵅ′⟩', '⟨w′cᵅ′⟩'], varname='⟨uᵢ′cᵅ′⟩', dimname='i', indices=dir_indices)
    #dsa = condense(dsa, vlist=['⟨Fcᵅ_x⟩', '⟨Fcᵅ_z⟩'], varname='⟨Fᵢcᵅ⟩',   dimname='i', indices=dir_indices)

    # do the same for bouyancy and buoyancy-like tracer
    b_indices = ['active', 'passive']
    dsa = dsa.rename_vars(dict(u_ym='⟨u⟩', v_ym='⟨v⟩', w_ym='⟨w⟩'))#, q_ym='⟨q⟩'))#, κₑ_ym='⟨κₑ⟩', νₑ_ym='⟨νₑ⟩'))
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (v.startswith('b') and v.endswith('ym'))]),
                   varname='⟨bᵝ⟩', dimname='β', indices=b_indices)
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (v.startswith('db') and v.endswith('dx_ym'))]),
                   varname='d⟨bᵝ⟩dx', dimname='β', indices=b_indices)
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (v.startswith('db') and v.endswith('dz_ym'))]),
                   varname='d⟨bᵝ⟩dz', dimname='β', indices=b_indices)
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (v.startswith('u′b') and v.endswith('ym'))]),
                   varname='⟨u′bᵝ′⟩', dimname='β', indices=b_indices)
    dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (v.startswith('w′b') and v.endswith('ym'))]),
                   varname='⟨w′bᵝ′⟩', dimname='β', indices=b_indices)
    #dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (v.startswith('ub') and v.endswith('sgs_ym'))]),
    #               varname='⟨Fbᵝ_x⟩', dimname='β', indices=b_indices)
    #dsa = condense(dsa, vlist=sorted([v for v in dsa_vlist if (v.startswith('wb') and v.endswith('sgs_ym'))]),
    #               varname='⟨Fbᵝ_z⟩', dimname='β', indices=b_indices)

    dsa = condense(dsa, vlist=['d⟨bᵝ⟩dx', 'd⟨bᵝ⟩dz'], varname='∇ⱼ⟨bᵝ⟩',   dimname='j', indices=dir_indices)
    dsa = condense(dsa, vlist=['⟨u′bᵝ′⟩', '⟨w′bᵝ′⟩'], varname='⟨uᵢ′bᵝ′⟩', dimname='i', indices=dir_indices)
    #dsa = condense(dsa, vlist=['⟨Fbᵝ_x⟩', '⟨Fbᵝ_z⟩'], varname='⟨Fᵢbᵝ⟩',   dimname='i', indices=dir_indices)
    #dsa = condense(dsa, vlist=['dbdx_ym',   'dbdz_ym'],  varname='∇ⱼ⟨b⟩',    dimname='j', indices=dir_indices)
    #dsa = condense(dsa, vlist=['u′b′_ym',   'w′b′_ym'],  varname='⟨uᵢ′b′⟩',  dimname='i', indices=dir_indices)
    #dsa = condense(dsa, vlist=['ub_sgs_ym',   'wb_sgs_ym'], varname='⟨Fᵢb⟩',  dimname='i', indices=dir_indices)

    # only use a subset of tracers for inversion
    if (type(subset_tracers) is float) or (type(subset_tracers) is int):
        if subset_tracers > 0:
            inversion_tracer_index = dsa.α[:subset_tracers]
        elif subset_tracers < 0:
            inversion_tracer_index = dsa.α[subset_tracers:]
        else:
            inversion_tracer_index = dsa.α

    dsa['tracer_used'] = dsa.α.isin(inversion_tracer_index) 
    dsa = separate_tracers(dsa, inversion_tracer_index=inversion_tracer_index,
                           used_basename='c', unused_indexname='m', unused_index_sup='ᵐ')

    if add_sgs_fluxes:
        print('Adding SGS fluxes')
        dsa['⟨uᵢ′cᵅ′⟩'] += dsa['⟨Fᵢcᵅ⟩']
        dsa['⟨uᵢ′bᵝ′⟩'] += dsa['⟨Fᵢbᵝ⟩']

    # reshaping matrices
    # following Bachman et al. (2015) the shapes are
    # ⟨uᵢ′cᵅ′⟩   n × t (number of dimensions × number of tracers)
    # ∇ⱼ⟨cᵅ⟩     n × t (number of dimensions × number of tracers)
    # Rᵢⱼ        n × n (number of dimensions × number of dimensions)
    dsa['⟨uᵢ′cᵅ′⟩'] = dsa['⟨uᵢ′cᵅ′⟩'].transpose(..., 'i', 'α')
    dsa['∇ⱼ⟨cᵅ⟩']   = dsa['∇ⱼ⟨cᵅ⟩'].transpose(..., 'j', 'α')

    #if local_tracer_selection:
    dsa = dsa.chunk(α=-1, β=-1, m=-1, i=-1, j=-1) if 'm' in dsa.dims else dsa.chunk(α=-1, β=-1, i=-1, j=-1)
    dsa['∇ⱼ⟨cᵅ⟩ₛ'], dsa['⟨uᵢ′cᵅ′⟩ₛ'] = xr.apply_ufunc(sort_tracer_gradients, dsa['∇ⱼ⟨cᵅ⟩'], dsa['⟨uᵢ′cᵅ′⟩'],
                                                      input_core_dims=[['j', 'α'], ['i', 'α']],
                                                      output_core_dims=[['j', 'αs'], ['i', 'αs']],
                                                      output_dtypes=[float, float],
                                                      kwargs=dict(invert_tracers=invert_tracers),
                                                      dask_gufunc_kwargs=dict(output_sizes={'αs': len(dsa.α[:invert_tracers])}),
                                                      dask='parallelized')

    if normalize_tracers:
        mask_lcf = np.sqrt((dsa['⟨uᵢ′cᵅ′⟩ₛ']**2).sum('i')) > 0.1*np.sqrt((dsa['⟨uᵢ′cᵅ′⟩ₛ']**2).sum('i')).max(['xC', 'zC'])
        rms_mag = np.sqrt((dsa['⟨uᵢ′cᵅ′⟩ₛ']**2).sum('i').where(mask_lcf).median(['xC', 'zC']))
        #grd_mag = np.sqrt((dsa['∇ⱼ⟨cᵅ⟩']**2).sum('j'))#.median(['xC', 'zC'])
        #grd_max = grd_mag.max('α')
        #rms_mag = grd_mag/grd_max
        #mask_lcf = np.sqrt((dsa['⟨uᵢ′cᵅ′⟩']**2).sum('i')) > 0.1*np.sqrt((dsa['⟨uᵢ′cᵅ′⟩']**2).sum('i')).max(['xC', 'zC'])
        #rms_mag = np.sqrt((dsa['∇ⱼ⟨cᵅ⟩']**2).sum('j').where(mask_lcf).median(['xC', 'zC']))
        dsa['⟨uᵢ′cᵅ′⟩|ⁿ'] = dsa['⟨uᵢ′cᵅ′⟩ₛ'] / rms_mag
        dsa['∇ⱼ⟨cᵅ⟩|ⁿ']   = dsa['∇ⱼ⟨cᵅ⟩ₛ']   / rms_mag

    # invert gradient matrix using a Pseudo-Inverse algorithm
    dsa = dsa.chunk(αs=-1)
    dsa['∇ⱼ⟨cᵅ⟩|ⁿ ⁻¹'] = xr.apply_ufunc(pinv, dsa['∇ⱼ⟨cᵅ⟩|ⁿ'],
                                        input_core_dims=[['j', 'αs']],
                                        output_core_dims=[['αs', 'j']],
                                        #kwargs=dict(rcond=1e-9),
                                        dask='parallelized').transpose(..., 'αs', 'j')

    # get the transport tensor Rᵢⱼ
    # for the matrix multiplication, the shapes are:
    # (n × t) ⋅ (t × n) = (n × n)
    # (number of dimensions × number of tracers) ⋅ (number of tracers × number of dimensions)
    dsa['Rᵢⱼ'] = -xr.apply_ufunc(np.matmul, dsa['⟨uᵢ′cᵅ′⟩|ⁿ'], dsa['∇ⱼ⟨cᵅ⟩|ⁿ ⁻¹'],
                                 input_core_dims=[['i', 'αs'], ['αs', 'j']],
                                 output_core_dims=[['i', 'j']],
                                 dask='parallelized')

    # get the symmetric and antisymmetric part
    SA = xr.apply_ufunc(decomposeSA, dsa['Rᵢⱼ'],
                        input_core_dims=[['i', 'j']],
                        output_core_dims=[['i', 'j', 'p']],
                        output_dtypes=float,
                        dask_gufunc_kwargs=dict(output_sizes={'p': 2}),
                        dask='parallelized')
    dsa['Sᵢⱼ'] = SA.isel(p=0)
    dsa['Aᵢⱼ'] = SA.isel(p=1)

    # get eigenvalues of the symmetric tensor
    dsa['Kappa'] = xr.apply_ufunc(np.linalg.eigvals, dsa['Sᵢⱼ'],
                                  input_core_dims=[['i', 'j']],
                                  output_core_dims=[['k']],
                                  dask_gufunc_kwargs=dict(output_sizes={'k': len(dsa.i)}),
                                  dask='parallelized')
    dsa = dsa.assign_coords(k=('k', np.arange(1, len(dsa.i)+1)))

    # concatenate used and used tracers
    #if subset_tracers:
    #    dsa = concat_tracers_back(dsa)

    # reconstruct tracer fluxes
    if subset_tracers: # from unused tracer gradients 
        dsa.coords['m'].attrs = {'long_name': 'Retained tracer index'}
        dsa['⟨uᵢ′cᵐ′⟩']  = dsa['⟨uᵢ′cᵐ′⟩'].transpose(..., 'i', 'm')
        dsa['∇ⱼ⟨cᵐ⟩']    = dsa['∇ⱼ⟨cᵐ⟩'].transpose(..., 'j', 'm')
        dsa['⟨uᵢ′cᵐ′⟩ᵣ'] = -xr.apply_ufunc(np.matmul, dsa['Rᵢⱼ'], dsa['∇ⱼ⟨cᵐ⟩'],
                                           input_core_dims=[['i', 'j'], ['j', 'm']],
                                           output_core_dims=[['i', 'm']],
                                           dask='parallelized')
    # from used tracer gradients
    dsa['⟨uᵢ′cᵅ′⟩ᵣ'] = -xr.apply_ufunc(np.matmul, dsa['Rᵢⱼ'], dsa['∇ⱼ⟨cᵅ⟩'],
                                       input_core_dims=[['i', 'j'], ['j', 'α']],
                                       output_core_dims=[['i', 'α']],
                                       dask='parallelized')

    # reconstruct buoyancy fluxes
    #dsa['∇ⱼ⟨b⟩'] = dsa['∇ⱼ⟨b⟩'].expand_dims(dim=dict(β=[1])).transpose(..., 'j', 'β')
    dsa['⟨uᵢ′bᵝ′⟩'] = dsa['⟨uᵢ′bᵝ′⟩'].transpose(..., 'i', 'β')
    dsa['∇ⱼ⟨bᵝ⟩']   = dsa['∇ⱼ⟨bᵝ⟩'].transpose(..., 'j', 'β')
    dsa['⟨uᵢ′bᵝ′⟩ᵣ'] = -xr.apply_ufunc(np.matmul, dsa['Rᵢⱼ'], dsa['∇ⱼ⟨bᵝ⟩'],
                                      input_core_dims=[['i', 'j'], ['j', 'β']],
                                      output_core_dims=[['i', 'β']],
                                      dask='parallelized')
    #dsa = dsa.squeeze('β').drop_vars('β')

    # compute flux divergence
    # dsa = dsa.chunk(xC=-1, zC=-1, time='auto')
    # dsa['∂x⟨u′b′⟩']  = dsa['⟨uᵢ′b′⟩'].sel(i=1).differentiate('xC')
    # dsa['∂z⟨w′b′⟩']  = dsa['⟨uᵢ′b′⟩'].sel(i=3).differentiate('zC')
    # dsa['∇ᵢ⟨uᵢ′b′⟩'] = dsa['∂x⟨u′b′⟩'] + dsa['∂z⟨w′b′⟩']

    # dsa['∂x⟨u′b′⟩ᵣ']  = dsa['⟨uᵢ′b′⟩ᵣ'].sel(i=1).differentiate('xC')
    # dsa['∂z⟨w′b′⟩ᵣ']  = dsa['⟨uᵢ′b′⟩ᵣ'].sel(i=3).differentiate('zC')
    # dsa['∇ᵢ⟨uᵢ′b′⟩ᵣ'] = dsa['∂x⟨u′b′⟩ᵣ'] + dsa['∂z⟨w′b′⟩ᵣ']

    # create mask based on ⟨w'b'⟩
    #wb_max  = dsa['⟨uᵢ′b′⟩'].sel(i=3).max(['xC', 'zC'])
    #mask_wb = dsa['⟨uᵢ′b′⟩'].sel(i=3) > 0.1*wb_max
    mask_b = np.abs(dsa['⟨uᵢ′bᵝ′⟩']) > 0.1*np.abs(dsa['⟨uᵢ′bᵝ′⟩']).max(['xC', 'zC'])
    mask_m = np.abs(dsa['⟨uᵢ′cᵐ′⟩']) > 0.1*np.abs(dsa['⟨uᵢ′cᵐ′⟩']).max(['xC', 'zC'])
    mask_α = np.abs(dsa['⟨uᵢ′cᵅ′⟩']) > 0.1*np.abs(dsa['⟨uᵢ′cᵅ′⟩']).max(['xC', 'zC'])

    # compute conditional volume average (only works for evenly spaced grid for now!)
    #dsa['⟨uᵢ′b′⟩ᶜˣᶻ']    = dsa['⟨uᵢ′b′⟩'].where(mask_b).mean(['xC', 'zC'])
    #dsa['⟨uᵢ′b′⟩ᵣᶜˣᶻ']   = dsa['⟨uᵢ′b′⟩ᵣ'].where(mask_b).mean(['xC', 'zC'])
    #dsa['∇ᵢ⟨uᵢ′b′⟩ᶜˣᶻ']  = dsa['∇ᵢ⟨uᵢ′b′⟩'].where(mask_b).mean(['xC', 'zC'])
    #dsa['∇ᵢ⟨uᵢ′b′⟩ᵣᶜˣᶻ'] = dsa['∇ᵢ⟨uᵢ′b′⟩ᵣ'].where(mask_b).mean(['xC', 'zC'])
    dsa['⟨uᵢ′bᵝ′⟩_err'] = np.abs((dsa['⟨uᵢ′bᵝ′⟩'] - dsa['⟨uᵢ′bᵝ′⟩ᵣ'])/dsa['⟨uᵢ′bᵝ′⟩']).where(mask_b).mean(['xC', 'zC'])
    dsa['⟨uᵢ′cᵐ′⟩_err'] = np.abs((dsa['⟨uᵢ′cᵐ′⟩'] - dsa['⟨uᵢ′cᵐ′⟩ᵣ'])/dsa['⟨uᵢ′cᵐ′⟩']).where(mask_m).mean(['xC', 'zC'])
    dsa['⟨uᵢ′cᵅ′⟩_err'] = np.abs((dsa['⟨uᵢ′cᵅ′⟩'] - dsa['⟨uᵢ′cᵅ′⟩ᵣ'])/dsa['⟨uᵢ′cᵅ′⟩']).where(mask_α).mean(['xC', 'zC'])

    dsa.coords['α'].attrs = {'long_name': 'Used tracer index'}
    dsa.coords['β'].attrs = {'long_name': 'Buoyancy index'}
    dsa.coords['j'].attrs = {'long_name': 'Gradient direction'}
    dsa.coords['i'].attrs = {'long_name': 'Flux direction'}
    dsa.coords['k'].attrs = {'long_name': 'Diffusivity index'}

    # save to disk
    fpath = data_dir+args.cname+'_transport_tensor.nc'
    delayed_nc = dsa.to_netcdf(fpath, compute=False, mode='w')
    with ProgressBar():
        #results = delayed_nc.compute()
        results = delayed_nc.compute(num_workers=40, memory_limit='4GB', processes=False)
    
#    with LocalCluster() as cluster, Client(cluster) as client:
#        results = delayed_nc.compute()
    t1 = time.time()
    print(f'Done saving to {fpath}')
    print(f'Computation finished in {((t1-t0)/60):.1f} minutes')


if __name__ == "__main__":
    main()
