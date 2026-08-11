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
from mixdiag import interp_to_zi, integrate_to_zi, get_mld_PE_anomaly


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

    ihr = int(args.hour - 1)
    vars_to_drop = ['u', 'v', 'w', 'c7', 'c8', 'eps']
    isubset_z_time = dict(zC=slice(5,None), zF=slice(5,None), time=slice(0,119,2))
    ds = xr.open_dataset(data_dir + args.cname + '_state.nc', decode_timedelta=True).drop_vars(vars_to_drop)\
           .isel(isubset_z_time).isel(time=slice(ihr, ihr+2)).chunk(zC=5, time=1)
    ds.close()
    dxF = (ds.xF[1] - ds.xF[0]).data
    filter_gauss = gcm_filters.Filter(filter_scale=150,
                                      dx_min=dxF,
                                      filter_shape=gcm_filters.FilterShape.GAUSSIAN,
                                      grid_type=gcm_filters.GridType.REGULAR,
                                     )
    uniform_zF  = np.arange(np.ceil(ds.zF[0]), 1)
    uniform_zC  = (uniform_zF[:-1] + uniform_zF[1:]) / 2
    uniform_dzF = np.diff(uniform_zF)
    pe_anomaly  = 0.15*(np.maximum(ds.attrs['Q₀'], 1) / 1)**(3/2) # for M006 set only

    chunks = dict(xC=440, yC=220)
    dsf = xr.open_dataset(data_dir + args.cname + f'/hr{args.hour:02d}_GIfske_budget.nc', decode_timedelta=True).chunk(chunks)
    dsf.close()
    mld = dsf.mld

    dsl = xr.open_dataset(data_dir + args.cname + f'/hr{args.hour:02d}_Gfiltered.nc', decode_timedelta=True).chunk(chunks)
    dsl.close()
    dzF = dsl.zF.diff('zF').data

    TMPDIR = os.getenv('TMPDIR')
    cluster_kw = dict(n_workers=128, threads_per_worker=1, memory_limit='1.8GB',
                      local_directory=TMPDIR, dashboard_address=':8787')

    with LocalCluster(**cluster_kw) as cluster:
        with Client(cluster) as client:
            bl = filter_gauss.apply(ds.b, dims=['yC', 'xC']).chunk(xC=440, yC=220, zC=-1)
            mld_old = get_mld_PE_anomaly(uniform_dzF,
                                         bl.interp(zC=uniform_zC, kwargs={'fill_value': 'extrapolate'}),
                                         ds.attrs, energy=pe_anomaly).rename({'time': 'time_old'})

            dsl['u_c'] = dsl.u * dsl.c
            dsl['v_c'] = dsl.v * dsl.c
            dsl['w_c'] = dsl.w * dsl.c

            c_mld   = interp_to_zi(dsl.c,   -mld)
            uc_mld  = interp_to_zi(dsl.uc,  -mld)
            vc_mld  = interp_to_zi(dsl.vc,  -mld)
            wc_mld  = interp_to_zi(dsl.wc,  -mld)
            u_c_mld = interp_to_zi(dsl.u_c, -mld)
            v_c_mld = interp_to_zi(dsl.v_c, -mld)
            w_c_mld = interp_to_zi(dsl.w_c, -mld)

            cint = integrate_to_zi(dsl.c, dsl.zC, c_mld, -mld)
            # cint = integrate_to_zi(dsl.c, dsl.zF, dzF -mld)

            mld.name     = 'mld'
            cint.name    = 'cint'
            c_mld.name   = 'c_mld'
            uc_mld.name  = 'uc_mld'
            vc_mld.name  = 'vc_mld'
            wc_mld.name  = 'wc_mld'
            u_c_mld.name = 'u_c_mld'
            v_c_mld.name = 'v_c_mld'
            w_c_mld.name = 'w_c_mld'
            mld_old.name = 'mld_old'

            dsi = xr.merge([mld, mld_old, cint, c_mld, uc_mld, vc_mld, wc_mld, u_c_mld, v_c_mld, w_c_mld])
            dsi['xF'] = dsl.xF
            dsi['yF'] = dsl.yF
            dsi = dsi.assign_attrs(dsl.attrs)

            delayed_nc = dsi.to_netcdf(data_dir + args.cname + f'/hr{args.hour:02d}_c_fluxes_at_mld.nc', compute=False)
            delayed_nc.compute()


if __name__ == "__main__":
    main()