import numpy as np
import xarray as xr
from numpy import pi
# from scipy.special import gammainc
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d
try:
    import mkl
    np.use_fastnumpy = True
except ImportError:
    pass


class TWODimensional_spec(object):
    """ A class that represents a two dimensional spectrum
            for real signals """

    def __init__(self, phi, d1, d2, detrend=True):

        self.phi = phi  # two dimensional real field
        self.d1 = d1
        self.d2 = d2
        self.n2, self.n1 = phi.shape[-2:]
        self.L1 = d1*self.n1
        self.L2 = d2*self.n2

        if detrend:
            self.phi = signal.detrend(self.phi, axis=(-1), type='linear')
            self.phi = signal.detrend(self.phi, axis=(-2), type='linear')
        else:
            pass

        win1 = np.hanning(self.n1)
        win1 = np.sqrt(self.n1 / (win1**2).sum())*win1
        win2 = np.hanning(self.n2)
        win2 = np.sqrt(self.n2 / (win2**2).sum())*win2

        win = win1[np.newaxis, ...] * win2[..., np.newaxis]

        self.phi *= win

        # test eveness
        if (self.n1 % 2):
            self.n1even = False
        else: self.n1even = True

        if (self.n2 % 2):
            self.n2even = False
        else: self.n2even = True

        # calculate frequencies
        self.calc_freq()

        # calculate spectrum
        self.calc_spectrum()

        # calculate total var
        #self.calc_var()

        # calculate isotropic spectrum
        self.ki, self.ispec = calc_ispec(self.k1, self.k2, self.spec, ndim=self.spec.ndim)

    def calc_freq(self):
        """ calculate array of spectral coordinate (frequency or
                wavenumber) in cycles per unit of L """

        # wavenumbers 
        self.k1 = np.fft.fftshift( np.fft.fftfreq(self.n1, self.d1) )
        self.k2 = np.fft.fftshift( np.fft.fftfreq(self.n2, self.d2) )

        self.kk1, self.kk2 = np.meshgrid(self.k1, self.k2)

        self.kappa2 = self.kk1**2 + self.kk2**2
        self.kappa = np.sqrt(self.kappa2)

    def calc_spectrum(self):
        """ calculates the power spectral density """
        self.phih = np.fft.fft2(self.phi)
        self.spec = (self.phih*self.phih.conj()).real * (self.d1*self.d2)**2 / (self.L1*self.L2)
        self.spec = np.fft.fftshift(self.spec)

   # def calc_var(self):
   #     """ compute variance of p from Fourier coefficients ph """
   #     self.var_dens = np.fft.fftshift(self.spec.copy(),axes=0)
        # only half of coefs [0] and [nx/2+1] due to symmetry in real fft2

   #     if self.n1even:
   #         self.var_dens[:,0],self.var_dens[:,-1] = self.var_dens[:,0]/2.,\
   #                 self.var_dens[:,-1]/2.
   #         self.var = self.var_dens.sum()*self.dk1*self.dk2
   #     else:
   #         self.var_dens[:,0],self.var_dens[:,-1] = self.var_dens[:,0]/2.,\
   #                 self.var_dens[:,-1]
   #         self.var = self.var_dens.sum()*self.dk1*self.dk2


def calc_ispec(k, l, E, ndim=2):
    """ Calculates the azimuthally-averaged spectrum

        Parameters
        ===========
        - E is the two-dimensional spectrum
        - k is the wavenumber is the x-direction
        - l is the wavenumber in the y-direction

        Output
        ==========
        - kr: the radial wavenumber
        - Er: the azimuthally-averaged spectrum """

    dk = np.abs(k[2] - k[1])
    dl = np.abs(l[2] - l[1])

    k, l = np.meshgrid(k, l)

    wv = np.sqrt(k**2 + l**2)

    # ignore Nyquist frequency, which is located at the negative side for numpy
    if k.max()>l.max():
        kmax = l.max()
    else:
        kmax = k.max()

    if ndim==3:
        nomg, nl, nk = E.shape
    elif ndim==2:
        nomg = 1

    dkr = np.sqrt(dk**2 + dl**2)
    kr = np.arange(dkr/2, kmax-dkr/2, dkr)
    Er = np.zeros((kr.size,nomg))

    for i in range(kr.size):

        fkr = (wv>kr[i]-dkr/2) & (wv<=kr[i]+dkr/2)
        dth = 2*pi / (fkr.sum()-1)
        if ndim==2:
            Er[i] = np.trapz(E[fkr]*wv[fkr], dx=dth)
        elif ndim==3:
            Er[i] = np.trapz(E[:,fkr]*wv[fkr][None,:], dx=dth)

    return kr, Er.squeeze()


def Gaussian_filter_2d(da, cutoff, dims, truncate=3, **kwargs):
    """ Apply two dimensional low-pass Gaussian filter to an xarray DataArray

        Parameters
        ===========
        cutoff: cutoff scale in physical unit, standard deviation of the Gaussian kernel
    """ 

    dims_list = da.dims
    core_dims = [d for d in dims_list if d[0] in dims]
    daf = da.copy(deep=True)
    for dim in core_dims:
        if dim=='zF' or dim=='zC':
            bc = 'constant'
        else:
            bc = 'wrap'

        spacing = abs(daf[dim].diff(dim)[0].data) 
        daf = xr.apply_ufunc(gaussian_filter1d, daf, input_core_dims=[[dim]], output_core_dims=[[dim]],
                             kwargs=dict(mode=bc, sigma=cutoff/spacing, truncate=truncate),
                             dask='parallelized', **kwargs)
    return daf.transpose(*dims_list)