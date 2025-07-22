#!/usr/bin/env python3

import os
import sys
import argparse
import warnings
import numpy as np
import xarray as xr
from xgcm import Grid
from xgcm.autogenerate import generate_grid_ds


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


def alongfront_mean_ufunc(u, w, b, c):
    # print(f'u: {u.shape} | w: {w.shape} | b: {b.shape} | c: {c.shape}')
    def _decompose(a):
        afm = a.mean(axis=-1, keepdims=True)
        prime = a - afm
        return afm.squeeze(axis=-1), prime

    u_afm, u_prime = _decompose(u)
    #v_afm, v_prime = _decompose(v)
    w_afm, w_prime = _decompose(w)
    b_afm, b_prime = _decompose(b)
    c_afm, c_prime = _decompose(c)
    wbs = (w_prime*b_prime).mean(axis=-1)
    wcs = (w_prime*c_prime).mean(axis=-1)
    ubs = (u_prime*b_prime).mean(axis=-1)
    ucs = (u_prime*c_prime).mean(axis=-1)
    return u_afm, w_afm, b_afm, c_afm, wbs, wcs, ubs, ucs


def alongfront_mean(ds, dims):
    bdims = ['β'] + dims
    cdims = ['α'] + dims
    return xr.apply_ufunc(alongfront_mean_ufunc, ds['⟨uᵢ⟩'].sel(i=1), ds['⟨uᵢ⟩'].sel(i=3), ds['⟨bᵝ⟩'], ds['⟨cᵅ⟩'],
                          input_core_dims=[dims, dims, bdims, cdims],
                          output_core_dims=[[], [], ['β'], ['α'], ['β'], ['α'], ['β'], ['α']],
                          output_dtypes=[float, float, float, float, float, float, float, float],
                          vectorize=True,
                          dask='parallelized')


def main():
    # process input arguments
    parser = argparse.ArgumentParser(description="""
            Calculate gradients corresponding to submesoscale and finescale fluxes.""")
    parser.add_argument('-c', '--case', action='store', dest='cname',
            metavar='CASENAME', help='Simulation case name')
    # parser.add_argument('-o', '--output', action='store', dest='fname_out',
    #         metavar='FIGNAME', help='Output figure name')
    parser.add_argument('--version', action='version', version='%(prog)s: 1.0')
    # parsing arguments and save to args
    args = parser.parse_args()

    # check input
    if not args.cname:
        print('Oceananigans simulation case name are required. Stop.\n')
        parser.print_help()
        sys.exit(1)

    # specify file path
    if sys.platform == 'linux' or sys.platform == 'linux2':
        data_dir = '/glade/derecho/scratch/zhihuaz/TracerInversion/Output/'
    elif sys.platform == 'darwin':
        data_dir = '/Users/zhihua/Documents/Work/Research/Projects/TRACE-SEAS/TracerInversion/Data/'
    else:
        print('OS not supported.')

    # read data
    dsc = xr.open_dataset(data_dir+args.cname+'_filtered_fields.nc', decode_timedelta=True).chunk(time=1)
    dsc.close()

    ua, wa, ba, ca, wbs, wcs, ubs, ucs = alongfront_mean(dsc, ['yC'])
    ua.name = '⟨u⟩ₐ'
    wa.name = '⟨w⟩ₐ'
    ba.name = '⟨bᵝ⟩ₐ'
    ca.name = '⟨cᵅ⟩ₐ'
    wbs.name = '⟨wˢbᵝˢ⟩ₐ'
    wcs.name = '⟨wˢcᵅˢ⟩ₐ'
    ubs.name = '⟨uˢbᵝˢ⟩ₐ'
    ucs.name = '⟨uˢcᵅˢ⟩ₐ'
    dss = xr.merge([ua, wa, ba, ca, wbs, wcs, ubs, ucs]).assign_coords(dsc.coords).drop_vars(['i', 'yC', 'yF'])
    dss = dss.assign_attrs(dsc.attrs)

    periodic_coords = {dim : dict(left=f'{dim}F', center=f'{dim}C') for dim in 'xy'}
    bounded_coords = {dim : dict(outer=f'{dim}F', center=f'{dim}C') for dim in 'z'}
    coords = {dim : periodic_coords[dim] if tpl=='P' else bounded_coords[dim] for dim, tpl in zip('xyz', 'PPN')}
    grid = Grid(dsc, coords=coords)
    dxF = dsc.xF.diff('xF').data[0]
    dyF = dsc.yF.diff('yF').data[0]
    dzF = dsc.zF.diff('zF').data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b_zf = grid.interp(dsc['⟨bᵝ⟩'], axis='z', boundary='extend').transpose(..., 'zF')
        b_xf = grid.interp(dsc['⟨bᵝ⟩'], axis='x').transpose(..., 'xF')
        b_yf = grid.interp(dsc['⟨bᵝ⟩'], axis='y').transpose(..., 'yF')
        c_zf = grid.interp(dsc['⟨cᵅ⟩'], axis='z', boundary='extend').transpose(..., 'zF')
        c_xf = grid.interp(dsc['⟨cᵅ⟩'], axis='x').transpose(..., 'xF')
        c_yf = grid.interp(dsc['⟨cᵅ⟩'], axis='y').transpose(..., 'yF')

        dsc['d⟨bᵝ⟩dz'] = grid.diff(b_zf, axis='z') / dzF
        dsc['d⟨bᵝ⟩dx'] = grid.diff(b_xf, axis='x') / dxF
        dsc['d⟨bᵝ⟩dy'] = grid.diff(b_yf, axis='y') / dyF
        dsc['d⟨cᵅ⟩dz'] = grid.diff(c_zf, axis='z') / dzF
        dsc['d⟨cᵅ⟩dx'] = grid.diff(c_xf, axis='x') / dxF
        dsc['d⟨cᵅ⟩dy'] = grid.diff(c_yf, axis='y') / dyF

    # merge all directions into one
    dir_indices = [1, 2, 3]
    dsc = condense(dsc, vlist=['d⟨cᵅ⟩dx', 'd⟨cᵅ⟩dy', 'd⟨cᵅ⟩dz'], varname='∇ⱼ⟨cᵅ⟩', dimname='j', indices=dir_indices)
    dsc = condense(dsc, vlist=['d⟨bᵝ⟩dx', 'd⟨bᵝ⟩dy', 'd⟨bᵝ⟩dz'], varname='∇ⱼ⟨bᵝ⟩', dimname='j', indices=dir_indices)
    dsc = dsc.isel(zC=slice(1, -1), zF=slice(1, -1))

    fpath = data_dir + args.cname + '_finescale_fluxes.nc'
    delayed_c = dsc.to_netcdf(fpath, compute=False)
    delayed_c.compute()


    periodic_coords = {dim : dict(left=f'{dim}F', center=f'{dim}C') for dim in 'x'}
    coords = {dim : periodic_coords[dim] if tpl=='P' else bounded_coords[dim] for dim, tpl in zip('xz', 'PN')}
    grid = Grid(dss, coords=coords)
    dxF = dss.xF.diff('xF').data[0]
    dzF = dss.zF.diff('zF').data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ba_xf = grid.interp(dss['⟨bᵝ⟩ₐ'], axis='x').transpose(..., 'xF')
        ba_zf = grid.interp(dss['⟨bᵝ⟩ₐ'], axis='z', boundary='extend').transpose(..., 'zF')
        ca_xf = grid.interp(dss['⟨cᵅ⟩ₐ'], axis='x').transpose(..., 'xF')
        ca_zf = grid.interp(dss['⟨cᵅ⟩ₐ'], axis='z', boundary='extend').transpose(..., 'zF')

        dss['d⟨bᵝ⟩ₐdx'] = grid.diff(ba_xf, axis='x') / dxF
        dss['d⟨bᵝ⟩ₐdz'] = grid.diff(ba_zf, axis='z') / dzF
        dss['d⟨cᵅ⟩ₐdx'] = grid.diff(ca_xf, axis='x') / dxF
        dss['d⟨cᵅ⟩ₐdz'] = grid.diff(ca_zf, axis='z') / dzF

    # merge all directions into one
    dir_indices = [1, 3]
    dss = condense(dss, vlist=['d⟨cᵅ⟩ₐdx', 'd⟨cᵅ⟩ₐdz'], varname='∇ⱼ⟨cᵅ⟩ₐ',   dimname='j', indices=dir_indices)
    dss = condense(dss, vlist=['d⟨bᵝ⟩ₐdx', 'd⟨bᵝ⟩ₐdz'], varname='∇ⱼ⟨bᵝ⟩ₐ',   dimname='j', indices=dir_indices)
    dss = condense(dss, vlist=['⟨uˢcᵅˢ⟩ₐ', '⟨wˢcᵅˢ⟩ₐ'], varname='⟨uᵢˢcᵅˢ⟩ₐ', dimname='i', indices=dir_indices)
    dss = condense(dss, vlist=['⟨uˢbᵝˢ⟩ₐ', '⟨wˢbᵝˢ⟩ₐ'], varname='⟨uᵢˢbᵝˢ⟩ₐ', dimname='i', indices=dir_indices)
    dss = dss.isel(zC=slice(1, -1), zF=slice(1, -1))

    fpath = data_dir + args.cname + '_submeso_fluxes.nc'
    delayed_s = dss.to_netcdf(fpath, compute=False)
    delayed_s.compute()


if __name__ == "__main__":
    main()
