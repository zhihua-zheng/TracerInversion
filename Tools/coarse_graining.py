#!/usr/bin/env python3

import os
import gc
import sys
import argparse
import gcm_filters
import cupy_xarray
import cupy as cp
import numpy as np
import xarray as xr


def construct_fluxes(udata, cdata):
    udim = list(set(udata.dims).difference(set(cdata.dims)))[0]
    cdim = list(set(cdata.dims).difference(set(udata.dims)))[0]
    uvec = udata.expand_dims(dim=dict(d=[1])).transpose(..., udim, 'd')
    cvec = cdata.expand_dims(dim=dict(d=[1])).transpose(..., 'd', cdim)
    return xr.apply_ufunc(np.matmul, uvec, cvec,
                          input_core_dims=[[udim, 'd'], ['d', cdim]],
                          output_core_dims=[[udim, cdim]],
                          dask='parallelized')


def convert_cupy_to_numpy(ds_gpu):
    tmp = {}
    for var in list(ds_gpu.keys()):
        da_gpu   = ds_gpu[var]
        tmp[var] = da_gpu.copy(data=cp.asnumpy(da_gpu.data))
    ds_cpu = xr.Dataset(tmp, coords=ds_gpu.coords)
    ds_cpu.attrs = ds_gpu.attrs
    return ds_cpu


#def alongfront_mean_ufunc(u, w, b, c):
#    # print(f'u: {u.shape} | w: {w.shape} | b: {b.shape} | c: {c.shape}')
#    def _decompose(a):
#        afm = a.mean(axis=-1, keepdims=True)
#        prime = a - afm 
#        return afm.squeeze(axis=-1), prime 
#
#    u_afm, u_prime = _decompose(u)
#    #v_afm, v_prime = _decompose(v)
#    w_afm, w_prime = _decompose(w)
#    b_afm, b_prime = _decompose(b)
#    c_afm, c_prime = _decompose(c)
#    wbs = (w_prime*b_prime).mean(axis=-1)
#    wcs = (w_prime*c_prime).mean(axis=-1)
#    ubs = (u_prime*b_prime).mean(axis=-1)
#    ucs = (u_prime*c_prime).mean(axis=-1)
#    return u_afm, w_afm, b_afm, c_afm, wbs, wcs, ubs, ucs
#
#
#def alongfront_mean(ds, dims):
#    bdims = ['β'] + dims
#    cdims = ['α'] + dims
#    return xr.apply_ufunc(alongfront_mean_ufunc, ds['⟨uᵢ⟩'].sel(i=1), ds['⟨uᵢ⟩'].sel(i=3), ds['⟨bᵝ⟩'], ds['⟨cᵅ⟩'],
#                          input_core_dims=[dims, dims, bdims, cdims],
#                          output_core_dims=[[], [], ['β'], ['α'], ['β'], ['α'], ['β'], ['α']],
#                          output_dtypes=[float, float, float, float, float, float, float, float],
#                          vectorize=True,
#                          dask='parallelized')


def main():
    # process input arguments
    parser = argparse.ArgumentParser(description="""
            Apply low-pass horizontal filter to 3D Oceananigans fields and compute fluxes.""")
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
    #slice('7 days 00:20:00', '9 days 00:20:00') for em01-em05 with Q000
    #slice('7 days', '8 days 23:00:00')
    dsf = xr.open_dataset(data_dir+args.cname+'_state.nc', decode_timedelta=True)\
            .sel(time=slice('7 days 00:20:00', '9 days 00:20:00'))\
            .isel(time=slice(None, None, 36), zC=slice(-54, None), zF=slice(-55, None)).chunk(time=4, zC=6).as_cupy()
    dsf.close()

    cutoff_scale = 200 # [m]
    dxy = (dsf.xC[1] - dsf.xC[0]).data

    vlist = ['u', 'v', 'w']
    tmp = dsf[vlist].to_array(dim='i').assign_coords({'i': range(1, 4)})
    tmp.attrs = {}
    dsf['uᵢ'] = tmp
    dsf = dsf.drop_vars(vlist).chunk(i=-1)

    vlist = [f'c{i+1}' for i in range(dsf.n_tracers)]
    tmp = dsf[vlist].to_array(dim='α').assign_coords({'α': range(1, dsf.n_tracers+1)})
    tmp.attrs = {}
    dsf['cᵅ'] = tmp
    dsf = dsf.drop_vars(vlist).chunk(α=-1)

    vlist = ['b']
    tmp = dsf[vlist].to_array(dim='β').assign_coords({'β': ['active']}) 
    tmp.attrs = {}
    dsf['bᵝ'] = tmp
    dsf = dsf.drop_vars(vlist).chunk(β=-1)

    # construct total fluxes
    dsf['uᵢcᵅ'] = construct_fluxes(dsf['uᵢ'], dsf['cᵅ']) 
    dsf['uᵢbᵝ'] = construct_fluxes(dsf['uᵢ'], dsf['bᵝ'])

    filter_lowpass = gcm_filters.Filter(filter_scale=cutoff_scale,
                                        dx_min=dxy,
                                        #filter_shape=gcm_filters.FilterShape.GAUSSIAN,
                                        filter_shape=gcm_filters.FilterShape.TAPER,
                                        #transition_width=np.pi,
                                        grid_type=gcm_filters.GridType.REGULAR)

    dsl = filter_lowpass.apply(dsf, dims=['yC', 'xC'])
    original_vars = list(dsf.keys())
    filtered_vars = [f'⟨{v}⟩' for v in original_vars]
    finescale_vars = [f'{v}ᵗ' for v in original_vars[:3]]
    dsh = dsf[original_vars[:3]] - dsl[original_vars[:3]]
    dsh = dsh.rename_vars(dict(zip(original_vars[:3], finescale_vars))).assign_coords(dsf.coords)
    dsh = dsh.assign_attrs(dsf.attrs)
    dsh_cpu = dsh.map_blocks(convert_cupy_to_numpy)

    fpath = data_dir + args.cname + '_finescale_fields.nc'
    delayed_h = dsh_cpu.to_netcdf(fpath, compute=False)
    delayed_h.compute(num_workers=4, threads_per_worker=2, memory_limit='10GB', processes=False)
    del dsh
    gc.collect()
    cp._default_memory_pool.free_all_blocks() # free unused GPU memory

    dsl = dsl.rename_vars(dict(zip(original_vars, filtered_vars)))
    ulcl = construct_fluxes(dsl['⟨uᵢ⟩'], dsl['⟨cᵅ⟩'])
    ulbl = construct_fluxes(dsl['⟨uᵢ⟩'], dsl['⟨bᵝ⟩'])
    #dsl['⟨uᵢᵗcᵅᵗ⟩'] = dsl['⟨uᵢcᵅ⟩'] - ulcl
    #dsl['⟨uᵢᵗbᵝᵗ⟩'] = dsl['⟨uᵢbᵝ⟩'] - ulbl
    #dsl = dsl.drop_vars(filtered_vars[-2:])
    dsl['⟨uᵢcᵅ⟩'] = dsl['⟨uᵢcᵅ⟩'] - ulcl
    dsl['⟨uᵢbᵝ⟩'] = dsl['⟨uᵢbᵝ⟩'] - ulbl
    dsl = dsl.rename_vars({'⟨uᵢcᵅ⟩': '⟨uᵢᵗcᵅᵗ⟩', '⟨uᵢbᵝ⟩': '⟨uᵢᵗbᵝᵗ⟩'})
    dsl_cpu = dsl.map_blocks(convert_cupy_to_numpy)

#    ua, wa, ba, ca, wbs, wcs, ubs, ucs = alongfront_mean(dsl, ['yC'])
#    ua.name = '⟨u⟩ₐ'
#    wa.name = '⟨w⟩ₐ'
#    ba.name = '⟨bᵝ⟩ₐ'
#    ca.name = '⟨cᵅ⟩ₐ'
#    wbs.name = '⟨wˢbᵝˢ⟩ₐ'
#    wcs.name = '⟨wˢcᵅˢ⟩ₐ'
#    ubs.name = '⟨uˢbᵝˢ⟩ₐ'
#    ucs.name = '⟨uˢcᵅˢ⟩ₐ'
#    dss = xr.merge([ua, wa, ba, ca, wbs, wcs, ubs, ucs]).assign_coords(dsl.coords).drop_vars(['i', 'yC', 'yF'])
#    dss = dss.assign_attrs(dsl.attrs)

    fpath = data_dir + args.cname + '_filtered_fields.nc'
    delayed_l = dsl_cpu.to_netcdf(fpath, compute=False)
    delayed_l.compute(num_workers=4, threads_per_worker=2, memory_limit='10GB', processes=False)

#    fpath = data_dir + args.cname + '_submeso_fluxes.nc'
#    delayed_s = dss.as_numpy().to_netcdf(fpath, compute=False)
#    delayed_s.compute()


if __name__ == "__main__":
    main()
