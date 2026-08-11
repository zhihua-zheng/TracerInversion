#!/usr/bin/env python3

import sys
import warnings
import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from viztool import FormatScalarFormatter

def main():
    # process input arguments
    parser = argparse.ArgumentParser(description="""
            Animate top slice and Cw1ss front transect from Oceananigans simualtion. Accept multiple variables.""")
    parser.add_argument('-f', '--file', action='store', dest='fname',
            metavar='FILENAME', help='Input simulation data')
    #parser.add_argument('-f2', '--file2', action='store', dest='fname2',
    #        metavar='FILENAME', help='Input GOTM data 2')
    #parser.add_argument('-v', '--variable', action='store', dest='vname',
    #        metavar='VARNAME', nargs='+', help='Variable name')
    parser.add_argument('-o', '--output', action='store', dest='fname_out',
            metavar='FIGNAME', help='Output figure name')
    #parser.add_argument('-ds', '--date_start', action='store', dest='date_start',
    #        metavar='STARTDATE',
    #        help='Starting date of input data, in the format of YYYYMMDD')
    #parser.add_argument('-de', '--date_end', action='store', dest='date_end',
    #        metavar='ENDDATE',
    #        help='Ending date of input data, in the format of YYYYMMDD')
    parser.add_argument('--version', action='version', version='%(prog)s: 1.0')
    # parsing arguments and save to args
    args = parser.parse_args()
    
    # check input
    if not args.fname or not args.fname_out:# or not args.vname
        print('Oceananigans netCDF data and output figure name are required. Stop.\n')#, variable name,
        parser.print_help()
        sys.exit(1)
    
    # specify file path
    if sys.platform == 'linux' or sys.platform == 'linux2':
        data_dir = '/glade/work/zhihuaz/Data/FrontalZone/'
        figs_dir = '/glade/u/home/zhihuaz/Projects/TRACE-SEAS/FrontalZone/Figures/'
    elif sys.platform == 'darwin':
        data_dir = '/Users/zhihua/Documents/Work/Research/Projects/TRACE-SEAS/FrontalZone/Data/'
        figs_dir = '/Users/zhihua/Documents/Work/Research/Projects/TRACE-SEAS/FrontalZone/Figures/'
    else:
        print('OS not supported.')
    
    # read data
    fpath = data_dir+args.fname
    ds = xr.open_dataset(fpath, group='average').load()
    ds.close()
    top = xr.open_dataset(fpath, group='slice/top').load()#.chunk('auto')
    top.close()

    # total buoyancy
    ds['Bt']  = (-ds.attrs['M²']  * ds.xC  + ds.B).transpose('zC','xC',...)
    top['bt'] = (-top.attrs['M²'] * top.xC + top.b).squeeze().transpose('yC','xC',...)
    top['w']  = top.w.squeeze().transpose('yC','xC',...)
    
    # Rossby number
    # top['Ro'] = (top['ζ'] / top.f).squeeze()

    # time interval and initial timestamp in unit of hours
    # dhr = int(((ds.time[2] - ds.time[1])/np.timedelta64(1,'h')).data)
    # hr0 = int((ds.time[0]/np.timedelta64(1,'h')).data)

    fig = plt.figure(figsize=(5,6.2), constrained_layout=True)
    gs = gridspec.GridSpec(9,1, figure=fig, left=0.1, right=0.9, bottom=0.1, top=0.9,
                           wspace=0.05, hspace=0.05)
    itime = 0

    ax1 = fig.add_subplot(gs[:6, 0], aspect='equal')
    kw_w1 = {'norm': mcolors.CenteredNorm(),
             'levels': np.linspace(-9e-3,9e-3,19),
             'extend': 'both',
             'cmap': 'PiYG_r'
            }
    kw_b1 = {'levels': np.arange(12,18,0.5)*1e-5,
             'colors': 'k',
             'linewidths': 0.2
            }
    Cw1 = ax1.contourf(top.xC, top.yC, top.w.isel(time=itime), **kw_w1)
    Cb1 = ax1.contour(top.xC, top.yC, top.bt.isel(time=itime), **kw_b1)
    cbar1 = fig.colorbar(Cw1, ax=ax1, fraction=0.027)
    
    ax2 = fig.add_subplot(gs[6:, 0], sharex=ax1)
    kw_u2 = {'norm': mcolors.CenteredNorm(),
             'levels': np.linspace(-9e-3,9e-3,19),
             'extend': 'both',
             'cmap': 'RdBu_r'
             }
    kw_b2 = {'levels': np.arange(-1,18,0.5)*1e-5,
             'colors': 'k',
             'linewidths': 1
             }
    Cu2 = ax2.contourf(ds.xF, ds.zC, ds.U.isel(time=itime), **kw_u2)
    Cb2 = ax2.contour(ds.xC, ds.zC, ds.Bt.isel(time=itime), **kw_b2)
    cbar2 = fig.colorbar(Cu2, ax=ax2, fraction=0.5)

    # def init():
    cbar1.set_label(r'w|$_{z \approx -10 \:m}$ [m $s^{-1}$]', labelpad=-50)
    # cbar1.set_label(r'$\zeta / f$', labelpad=-40)
    cbar1.formatter.set_powerlimits((0, 0))
    cbar1.formatter.set_useMathText(True)
    cbar1.ax.yaxis.set_offset_position('left')
    cbar1.ax.set_yticks([-9e-3, -6e-3, -3e-3, 0, 3e-3, 6e-3, 9e-3])
    ax1.set_title(rf'Time / T$_{{inertial}}$ = {ds.timeTf[itime]:.2f}')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_ylabel('Y [m]')
    
    # cbar2.set_label(r'$\langle w\rangle_{y}$ [m $s^{-1}$]', labelpad=-45)
    cbar2.set_label(r'$\langle u\rangle_{y}$ [m $s^{-1}$]', labelpad=-45)
    cbar2.formatter.set_powerlimits((0, 0))
    cbar2.formatter.set_useMathText(True)
    cbar2.ax.yaxis.set_offset_position('left')
    cbar2.ax.set_yticks([-9e-3, -6e-3, -3e-3, 0, 3e-3, 6e-3, 9e-3])
    ax2.set_ylim(-100,0)
    ax2.set_ylabel('Z [m]')
    ax2.set_xlabel('X [m]')
        # return Cw1, Cb1, Cu2, Cb2

    def update(frame):
        nonlocal Cw1, Cb1, Cu2, Cb2
        # for each frame, get new data and clear old data stored on each artist
        wtop    = top.w.isel(time=frame)
        btop    = top.bt.isel(time=frame)
        u_ymean = ds.U.isel(time=frame)
        b_ymean = ds.Bt.isel(time=frame)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for obj in (Cw1, Cb1, Cu2, Cb2):
                for coll in obj.collections:
                    coll.remove()
        # update the plot
        Cw1 = ax1.contourf(top.xC, top.yC, wtop, **kw_w1)
        Cb1 = ax1.contour(top.xC, top.yC, btop, **kw_b1)
        Cu2 = ax2.contourf(ds.xF, ds.zC, u_ymean, **kw_u2)
        Cb2 = ax2.contour(ds.xC, ds.zC, b_ymean, **kw_b2)
        ax1.set_title(rf'Time / T$_{{inertial}}$ = {ds.timeTf[frame]:.2f}')
        return Cw1, Cb1, Cu2, Cb2

    ani = animation.FuncAnimation(fig=fig, func=update, frames=range(ds.dims['time']), #init_func=init, 
                                  interval=400, repeat_delay=800)#, blit=True
    # save animation
    ani.save(figs_dir+args.fname_out, writer='ffmpeg', fps=5, dpi=200)


if __name__ == "__main__":
    main()
