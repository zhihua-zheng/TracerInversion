#!/usr/bin/env python3
import cmocean
import numpy as np
import dask.array as daska
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import LinearSegmentedColormap
# from mpl_toolkits.mplot3d.axis3d import Axis


class FormatScalarFormatter(tkr.ScalarFormatter):
    def __init__(self, fformat='%1.1f', offset=True, mathText=True):
        self.fformat = fformat
        tkr.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)
    def _set_format(self):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


# if not hasattr(Axis, '_get_coord_info_old'):
#     def _get_coord_info_new(self, renderer):
#         mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
#         mins += deltas / 4
#         maxs -= deltas / 4
#         return mins, maxs, centers, deltas, tc, highs
#     Axis._get_coord_info_old = Axis._get_coord_info
#     Axis._get_coord_info = _get_coord_info_new


def plot_box_frame(ax, xlim, ylim, zlim, azim, cloud=False, **edges_kw):
    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim
    xazm = xmin if azim < -90 else xmax
    ax.plot([xmax, xmax], [ymin, ymax], zmax, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], zmax, **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymax], zmax, **edges_kw)
    ax.plot([xmin, xmax], [ymax, ymax], zmax, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
    ax.plot([xazm, xazm], [ymax, ymax], [zmin, zmax], **edges_kw)
    if not cloud:
        ax.plot([xazm, xazm], [ymin, ymax], zmin, **edges_kw)
        ax.plot([xmin, xmax], [ymin, ymin], zmin, **edges_kw)
    else:
        ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], **edges_kw)


def config_colorbar(pcm, ax=None, sci_notation=True, shrink=1, nbins=None, **kwargs):
    if ax is None: ax = plt.gca()
    cbar = plt.colorbar(pcm, ax=ax, location='right', shrink=shrink, aspect=50, pad=0.02, **kwargs)
    cbar.ax.tick_params(labelsize=6, length=2)
    if sci_notation:
        cbar.formatter.set_powerlimits((0, 0))
        if nbins is not None:
            cbar.ax.locator_params(nbins=5)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_fontsize(7)
        offset_text.set_horizontalalignment('left')
        offset_text.set_verticalalignment('bottom')
        offset_text.set_position((0, 1))


def div_sym_cmap(rcut, base='RdBu_r'):
    # rcut: fraction of cutoff value in the normalized [-1, 1] coordinate
    base_cmap = plt.get_cmap(base).resampled(200)
    colors = base_cmap(np.linspace(0, 1, 200))
    # Position of cutoff in [0, 1] scale of colormap
    frac_neg = (1 - rcut) / 2
    frac_pos = (1 + rcut) / 2
    whitened = (np.linspace(0, 1, 200) >= frac_neg) & (np.linspace(0, 1, 200) <= frac_pos) 
    colors[whitened, :] = np.array([1, 1, 1, 1])
    bname = base if isinstance(base, str) else base.name
    return LinearSegmentedColormap.from_list(bname+'_ds', colors)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def get_coords(da, xvar, yvar, hori_km=False):
    x, y = da[xvar], da[yvar]
    x_is_hori, y_is_hori = xvar.startswith(('x', 'y')), yvar.startswith(('x', 'y'))
    rescale_x, rescale_y = x_is_hori & hori_km, y_is_hori & hori_km
    if xvar == 'time':
        x = x / np.timedelta64(1, 'D')
    else:
        x = x / 1e3 if rescale_x else x
    y = y / 1e3 if rescale_y else y
    return x, y


def pcolor_center(da, ax=None, cmap='RdBu_r', cupr=1, smag=None, mag_mode=None, hori_km=True, rotate=False, **kwargs):
    if ax is None: ax = plt.gca()
    if isinstance(da.data, daska.Array):
        da = da.load()
    xvar, yvar = np.flip(sorted(da.dims)) if rotate else sorted(da.dims)
    da = da.transpose(..., yvar, xvar)
    if smag is None and 'norm' not in kwargs:
        if mag_mode == 'range':
            smag = np.max(np.abs(da))
        else:
            smag = np.abs(da).quantile(0.99, dim=[xvar, yvar])*cupr
    x, y = get_coords(da, xvar, yvar, hori_km=hori_km)
    if 'norm' in kwargs:
        pcm = ax.pcolormesh(x, y, da, cmap=cmap, **kwargs);
    else:
        pcm = ax.pcolormesh(x, y, da, cmap=cmap, vmin=-smag, vmax=smag, **kwargs);
    ax.ticklabel_format(useMathText=True)
    return pcm, smag


def pcolor_limits(da, ax=None, cmap='RdBu_r', clim=None, mag_mode=None, cnorm='linear', hori_km=True, rotate=False, **kwargs):
    if ax is None: ax = plt.gca()
    if isinstance(da.data, daska.Array):
        da = da.load()
    xvar, yvar = np.flip(sorted(da.dims)) if rotate else sorted(da.dims)
    da = da.transpose(..., yvar, xvar)
    if clim is None:
        if 'norm' in kwargs:
            clim = [None, None]
        else:
            if mag_mode == 'range':
                umag = da.max(dim=[xvar, yvar])
                lmag = da.min(dim=[xvar, yvar])
            else:
                umag = da.quantile(0.99, dim=[xvar, yvar])
                lmag = da.quantile(0.01, dim=[xvar, yvar])
            clim = [lmag, umag]
    x, y = get_coords(da, xvar, yvar, hori_km=hori_km)
    pcm = ax.pcolormesh(x, y, da, cmap=cmap, vmin=clim[0], vmax=clim[1], **kwargs);
    ax.ticklabel_format(useMathText=True)
    return pcm, clim


def get_pdf_of_icdf(pdf, bin_area, icdf=[0.3, 0.6, 0.9, 0.995]):
    if isinstance(pdf, daska.Array):
        pdf = pdf.compute()
    pdf = pdf.flatten()
    pdf_sort = np.sort(pdf)[::-1]
    cdf = np.cumsum(pdf_sort) * bin_area
    pdf_of_icdf = []
    for f in icdf:
        idx = np.searchsorted(cdf, f) # where the inquired cdf is located
        pdf_of_icdf.append(pdf_sort[idx])
    return np.flip(pdf_of_icdf)


def create_register_cmaps(extreme=True):
    colorlist = ['xkcd:white', 'xkcd:heather', 'xkcd:bright sky blue', 'xkcd:jade', 'xkcd:golden yellow', 'xkcd:tomato red']
    nodes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cmappdf = LinearSegmentedColormap.from_list('pdf', list(zip(nodes, colorlist)))
    
    colorlist = ['xkcd:cerulean', 'xkcd:white', 'xkcd:red']
    nodes = [0.0, 0.5, 1.0]
    cmapw = LinearSegmentedColormap.from_list('w', list(zip(nodes, colorlist)))

    colorlist = ['xkcd:white', '#c6c629', 'xkcd:cornflower', '#2a3871', 'xkcd:red']
    nodes = [0.0, 0.3, 0.6, 0.8, 1.0]
    cmapc = LinearSegmentedColormap.from_list('c', list(zip(nodes, colorlist)))
    
    colorlist = ['xkcd:purple', 'xkcd:white', 'xkcd:green']
    nodes = [0.0, 0.5, 1.0]
    cmapcp = LinearSegmentedColormap.from_list('cp', list(zip(nodes, colorlist)))
    
    colorlist = ['xkcd:turquoise', 'xkcd:white', 'xkcd:Orange']
    nodes = [0.0, 0.5, 1.0]
    cmapflux = LinearSegmentedColormap.from_list('flux', list(zip(nodes, colorlist)))

    positive_curl = cmocean.tools.crop_by_percent(cmocean.cm.curl, 50, which='min', N=None)
    positive_curl.name = 'positive_curl'

    plt.colormaps.register(cmap=cmappdf)
    plt.colormaps.register(cmap=cmapw)
    plt.colormaps.register(cmap=cmapc)
    plt.colormaps.register(cmap=cmapcp)
    plt.colormaps.register(cmap=cmapflux)
    plt.colormaps.register(cmap=positive_curl)
    if extreme == True:
        cmap_w = div_sym_cmap(0.1, base=cmapw)
        cmap_cp = div_sym_cmap(0.1, base=cmapcp)
        cmap_flux = div_sym_cmap(0.1, base=cmapflux)
        RdBu_r_ds = div_sym_cmap(0.1, base='RdBu_r')
        # balance_ds = div_sym_cmap(0.05, base='cmo.balance')
        plt.colormaps.register(cmap=cmap_w)
        plt.colormaps.register(cmap=cmap_cp)
        plt.colormaps.register(cmap=cmap_flux)
        plt.colormaps.register(cmap=RdBu_r_ds)