#!/usr/bin/env python3
import numpy as np
import matplotlib.ticker as tkr


class FormatScalarFormatter(tkr.ScalarFormatter):
    def __init__(self, fformat='%1.1f', offset=True, mathText=True):
        self.fformat = fformat
        tkr.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)
    def _set_format(self):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def pcolor_center(da, ax, cmap='RdBu_r', cupr=1.2, sym_mag=None, use_km=False, **kwargs):
    xvar, zvar = sorted(da.dims)
    if sym_mag is None:
        sym_mag = np.abs(da).quantile(0.99, dim=[xvar, zvar])*cupr
    if use_km:
        x, z = da[xvar]/1e3, da[zvar]/1e3
    else:
        x, z = da[xvar], da[zvar]
    if xvar == 'time':
        x = x / np.timedelta64(1, 'D')
    pcm = ax.pcolormesh(x, z, da, cmap=cmap, vmin=-sym_mag, vmax=sym_mag, **kwargs);
    ax.ticklabel_format(useMathText=True)
    return pcm, sym_mag


def pcolor_lim(da, ax, cmap='RdBu_r', clim=None, cnorm='linear', **kwargs):
    xvar, zvar = sorted(da.dims)
    if clim is None:
        if 'norm' in kwargs:
            clim = [None, None]
        else:
            umag = da.quantile(0.99, dim=[xvar, zvar])
            lmag = da.quantile(0.01, dim=[xvar, zvar])
            clim = [lmag, umag]
    if xvar == 'time':
        x = da[xvar] / np.timedelta64(1, 'D')
    else:
        x = da[xvar]
    pcm = ax.pcolormesh(x, da[zvar], da, cmap=cmap, vmin=clim[0], vmax=clim[1], **kwargs);
    return pcm