#!/usr/bin/env python3

import os
import sys
import argparse
import warnings
import gcm_filters
import numpy as np
import xarray as xr
from dask.distributed import LocalCluster, Client
from xgcm import Grid
from mixdiag import double_front_boundary, get_bld_Rib, get_mld_PE_anomaly


def Ifske_budget_without_VSPuv(ds, filter_taper, dxF, dyF, dzF, grid):
    uu = ds.u * ds.u
    uv = ds.u * ds.v
    uw = ds.u * ds.w
    vv = ds.v * ds.v
    vw = ds.v * ds.w
    ww = ds.w * ds.w
    wb = ds.w * ds.b
    vb = ds.v * ds.b
    ub = ds.u * ds.b
    wc = ds.w * ds.c
    vc = ds.v * ds.c
    uc = ds.u * ds.c

    uu.name = 'uu'
    uv.name = 'uv'
    uw.name = 'uw'
    vv.name = 'vv'
    vw.name = 'vw'
    ww.name = 'ww'
    wb.name = 'wb'
    vb.name = 'vb'
    ub.name = 'ub'
    wc.name = 'wc'
    vc.name = 'vc'
    uc.name = 'uc'
    dsf = xr.merge([uu, uv, uw, vv, vw, ww, wb, vb, ub, wc, vc, uc])

    dsl = filter_taper.apply(ds,  dims=['yC', 'xC'])
    dsf = filter_taper.apply(dsf, dims=['yC', 'xC'])
    dsf['uu'] = dsf.uu - (dsl.u * dsl.u)
    dsf['uv'] = dsf.uv - (dsl.u * dsl.v)
    dsf['uw'] = dsf.uw - (dsl.u * dsl.w)
    dsf['vv'] = dsf.vv - (dsl.v * dsl.v)
    dsf['vw'] = dsf.vw - (dsl.v * dsl.w)
    dsf['ww'] = dsf.ww - (dsl.w * dsl.w)
    dsf['wb'] = dsf.wb - (dsl.w * dsl.b)
    dsf['vb'] = dsf.vb - (dsl.v * dsl.b)
    dsf['ub'] = dsf.ub - (dsl.u * dsl.b)
    dsf['wc'] = dsf.wc - (dsl.w * dsl.c)
    dsf['vc'] = dsf.vc - (dsl.v * dsl.c)
    dsf['uc'] = dsf.uc - (dsl.u * dsl.c)
    dsf['fske'] = (dsf.uu + dsf.vv + dsf.ww) / 2

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # b_xf = grid.interp(dsl.b, axis='x').transpose(..., 'xF')
        # b_yf = grid.interp(dsl.b, axis='y').transpose(..., 'yF')
        u_xf = grid.interp(dsl.u, axis='x').transpose(..., 'xF')
        u_yf = grid.interp(dsl.u, axis='y').transpose(..., 'yF')
        v_xf = grid.interp(dsl.v, axis='x').transpose(..., 'xF')
        v_yf = grid.interp(dsl.v, axis='y').transpose(..., 'yF')
        w_xf = grid.interp(dsl.w, axis='x').transpose(..., 'xF')
        w_yf = grid.interp(dsl.w, axis='y').transpose(..., 'yF')

        # dbdx = grid.diff(b_xf, axis='x') / dxF
        # dbdy = grid.diff(b_yf, axis='y') / dyF

        dudx = grid.diff(u_xf, axis='x') / dxF
        dudy = grid.diff(u_yf, axis='y') / dyF

        dvdx = grid.diff(v_xf, axis='x') / dxF
        dvdy = grid.diff(v_yf, axis='y') / dyF

        dwdx = grid.diff(w_xf, axis='x') / dxF
        dwdy = grid.diff(w_yf, axis='y') / dyF

    dsl['δ']  = dudx + dvdy
    dsl['σₙ'] = dudx - dvdy
    dsl['σₛ'] = dvdx + dudy
    # dsl['M²'] = np.sqrt(dbdx**2 + dbdy**2)

    dsf['HSP_δ'] = - (dsf.uu + dsf.vv) * dsl['δ']  / 2
    dsf['HSP_σ'] = - (dsf.uu - dsf.vv) * dsl['σₙ'] / 2 - dsf.uv * dsl['σₛ']
    dsf['VSP_w'] = -  dsf.uw * dwdx - dsf.vw * dwdy + dsf.ww * dsl['δ']

    # M2l = dsl['M²'].where(dsl.zC >= -dsl.Hm).mean('zC')/dsl.attrs['M²']
    # dsl['mask_fz'] = double_front_boundary(M2l, Mc=0.2, in_km=0, out_km=0)

    dsf['Ifske'] = (dsf.fske.transpose(..., 'zC') * dzF).sum('zC')
    dsf['IVBP']  = (dsf.wb.transpose(..., 'zC')   * dzF).sum('zC')
    dsf['IHSP_δ'] = (dsf['HSP_δ'].transpose(..., 'zC') * dzF).sum('zC')
    dsf['IHSP_σ'] = (dsf['HSP_σ'].transpose(..., 'zC') * dzF).sum('zC')
    dsf['IVSP_w'] = (dsf['VSP_w'].transpose(..., 'zC') * dzF).sum('zC')

    var_dsl_to_save   = ['u', 'v', 'w', 'b', 'c', 'xF', 'yF', 'zF'] #'mask_fz',
    var3d_dsf_to_save = ['uu', 'vv', 'ww', 'uv', 'uw', 'vw', 'wb', 'vb', 'ub', 'wc', 'vc', 'uc']
    var2d_dsf_to_save = ['Ifske', 'IVBP', 'IHSP_δ', 'IHSP_σ', 'IVSP_w']

    dsl_save = xr.merge([dsl[var_dsl_to_save], dsf[var3d_dsf_to_save]])
    dsf_save = dsf[var2d_dsf_to_save]
    return dsl_save, dsf_save


def main():
    parser = argparse.ArgumentParser(description="""
             Compute (vertically integrated) fine-scale kinetic energy budget""")
    parser.add_argument('-c', '--case', action='store', dest='cname',
                        help='Simulation case name')
    parser.add_argument('-hn', '--hour_number', action='store', dest='hour', type=int,
                        help='Index of the begining hour number to process')
    args = parser.parse_args()

    # specify file path
    if sys.platform == 'linux' or sys.platform == 'linux2':
        data_dir = '/glade/derecho/scratch/zhihuaz/TracerInversion/Output/'
    elif sys.platform == 'darwin':
        data_dir = '/Users/zhihua/Documents/Work/Research/Projects/TRACE-SEAS/TracerInversion/Data/'
    else:
        print('OS not supported.')

    isubset_z_time = dict(zC=slice(19,None), zF=slice(19,None), time=slice(2,146,3))
    ihr = int(args.hour - 1)
    ds = xr.open_dataset(data_dir + args.cname + '_state.nc', decode_timedelta=True).isel(isubset_z_time).isel(time=slice(ihr, ihr+2)).chunk(time=1, zC=5)
    ds.close()
    ds = ds.drop_vars(['c7'])
    ds = ds.rename_vars({'c8': 'c'})

    dxF = (ds.xF[1] - ds.xF[0]).data
    dyF = (ds.yF[1] - ds.yF[0]).data
    dzF = ds.zF.diff('zF').data
    periodic_coords = {dim : dict(left=f'{dim}F', center=f'{dim}C') for dim in 'xy'}
    bounded_coords = {dim : dict(outer=f'{dim}F', center=f'{dim}C') for dim in 'z'}
    coords = {dim : periodic_coords[dim] if tpl=='P' else bounded_coords[dim] for dim, tpl in zip('xyz', 'PPN')}
    grid = Grid(ds, coords=coords)

    filter_taper = gcm_filters.Filter(filter_scale=100,
                                      dx_min=dxF,
                                      filter_shape=gcm_filters.FilterShape.TAPER,
                                      transition_width=np.pi*4,
                                      grid_type=gcm_filters.GridType.REGULAR,
                                     )
    TMPDIR = os.getenv('TMPDIR')
    cluster_kw = dict(n_workers=128, threads_per_worker=1, memory_limit='1.8GB',
                      local_directory=TMPDIR, dashboard_address=':8787')

    with LocalCluster(**cluster_kw) as cluster:
        with Client(cluster) as client:
            dsl, dsf = Ifske_budget_without_VSPuv(ds, filter_taper, dxF, dyF, dzF, grid)
            dsf = dsf.persist()

            delayed_nc_dsl = dsl.to_netcdf(data_dir + args.cname + f'/hr{args.hour:02d}_filtered.nc', compute=False)
            delayed_nc_dsl.compute()

            dsl = xr.open_dataset(data_dir + args.cname + f'/hr{args.hour:02d}_filtered.nc', decode_timedelta=True).chunk(xC=200, yC=200, time=1)
            dsl.close()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                b_zf = grid.interp(dsl.b, axis='z', boundary='extend').transpose(..., 'zF')
                u_zf = grid.interp(dsl.u, axis='z', boundary='extend').transpose(..., 'zF')
                v_zf = grid.interp(dsl.v, axis='z', boundary='extend').transpose(..., 'zF')

                dbdz = grid.diff(b_zf, axis='z') / dzF
                du   = grid.diff(u_zf, axis='z')
                dv   = grid.diff(v_zf, axis='z')

            dsf['IVSP_uv'] = - (dsl.uw.transpose(..., 'zC') * du + dsl.vw.transpose(..., 'zC') * dv).sum('zC')
            dsf['bld'] = get_bld_Rib(dsl.zF, dzF, dsl.b, dsl.u, dsl.v, dbdz, dsl.attrs, Ribc=0.3)

            uniform_zF  = np.arange(np.ceil(dsl.zF[0]), 1)
            uniform_zC  = (uniform_zF[:-1] + uniform_zF[1:]) / 2
            uniform_dzF = np.diff(uniform_zF)
            pe_anomaly  = 0.115*(np.maximum(dsl.attrs['Q₀'], 1) / 1)**(3/2) # for M006 set only
            dsf['mld']  = get_mld_PE_anomaly(uniform_dzF,
                                            dsl.b.interp(zC=uniform_zC, kwargs={'fill_value': 'extrapolate'}),
                                            dsl.attrs, energy=pe_anomaly)

            delayed_nc_dsf = dsf.to_netcdf(data_dir + args.cname + f'/hr{args.hour:02d}_Ifske_budget.nc', compute=False)
            delayed_nc_dsf.compute()


if __name__ == "__main__":
    main()