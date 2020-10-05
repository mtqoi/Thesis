# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:15:23 2019

@author: Matthew Thornton

Calculate signature length for QDS-1 protocol. 
"""

# ================================================
#	Import statements
# ================================================

import qutip as qt 
import multiprocessing as mp
import AQC_background as aqc 
import numpy as np 
import os 
qt.settings.auto_tidyup = False
qt.settings.atol = 1e-18
cachedir = os.getcwd() + "\\" + "cache\\"
if os.path.isdir(cachedir) == False:
	os.mkdir(cachedir)
# os.path.isdir(cachedir)
from joblib import Memory
memory = Memory(cachedir, verbose=0)
from scipy.optimize import fsolve


# ================================================
#	Define global variables
# ================================================
class gv: 
    """ Class to contain global variables.
    """
    alpha=0.5
    T=0.8
    barn = 0.1
    dims = 10
    lim = 6
    mixedQ = True
    
# ================================================
# ================================================
# ================================================
#	Integrated conditional states
# ================================================
# ================================================
# ================================================



class beamsplitter_attack:
    """ Class to contain functions for manipulating and integrating states under beamsplitter attack. Mainly used to allow the integration to be parallelized and called from other files. """
    def __init__(self, alpha=gv.alpha, T=gv.T, dims=gv.dims, lim=gv.lim, delta_r = 0.0, delta_th = 0.0, mixedQ=gv.mixedQ):
        self.alpha = alpha
        self.T = T
        self.dims = dims
        self.lim = lim
        self.mixedQ = mixedQ
        self.delta_r = delta_r
        self.delta_th = delta_th

    def parallel_integrate_mpmath(self):
        """ Integrates over entire phase-space.
        """
        print("Running beamsplitter_attack.parallel_integrate() with alpha = " + str(self.alpha) + ", T = " + str(self.T) + ", dims = " + str(self.dims) + ", lim = " + str(self.lim) + "mixedQ = " + str(self.mixedQ))
		
		# Get parameters which will let us reconstruct the Qobj at the end.
        _teststate = aqc.coh_vac_heterodyne_conditionalstate(0.5, self.alpha, self.T, self.dims, qobjQ=True, mixedQ=False)
        _mydims = _teststate.dims
        _myshape = _teststate.shape
        _newmat = np.zeros(_myshape, dtype=np.complex64)
        _myrange = np.arange(0, _myshape[0])

		# Multiprocessing and integrating
        pool = mp.Pool(mp.cpu_count())
        _newmat = np.array(pool.starmap(aqc.integrate_coh_vac_heterodyne_mpmath, [(i, j, self.alpha, self.T, self.dims, self.lim, self.delta_r, self.delta_th, self.mixedQ) for i in _myrange for j in _myrange]))
        pool.close()

		# Reconstructing state
        _newmat = np.reshape( _newmat, [_myshape[0], _myshape[0]])
        _newstate = qt.Qobj(_newmat, dims = _mydims)
        self.state = _newstate
        return self.state

    def parallel_integrate_onequad(self):
        """ Integrates over first quadrant of phase-space.
        """
        print("Running beamsplitter_attack.parallel_integrate_onequad() with alpha = " + str(self.alpha) + ", T = " + str(self.T) + ", dims = " + str(self.dims) + ", lim = " + str(self.lim) + "mixedQ = " + str(self.mixedQ))
		
		# Get parameters which will let us reconstruct the Qobj at the end.
        _teststate = aqc.coh_vac_heterodyne_conditionalstate(0.5, self.alpha, self.T, self.dims, qobjQ=True, mixedQ=False)
        _mydims = _teststate.dims
        _myshape = _teststate.shape
        _newmat = np.zeros(_myshape, dtype=np.complex64)
        _myrange = np.arange(0, _myshape[0])

		# Multiprocessing and integrating
        pool = mp.Pool(mp.cpu_count())
        _newmat = np.array(pool.starmap(aqc.integrate_coh_vac_heterodyne_onequad, [(i, j, self.alpha, self.T, self.dims, self.lim, self.mixedQ) for i in _myrange for j in _myrange]))
        pool.close()

		# Reconstructing state
        _newmat = np.reshape( _newmat, [_myshape[0], _myshape[0]])
        _newstate = qt.Qobj(_newmat, dims = _mydims)
        self.state = _newstate
        return self.state
    
    def parallel_integrate_onequad_mpmath(self):
        """ Integrates over first quadrant of phase-space.
        """
        print("Running beamsplitter_attack.parallel_integrate_onequad() with alpha = " + str(self.alpha) + ", T = " + str(self.T) + ", dims = " + str(self.dims) + ", lim = " + str(self.lim) + "mixedQ = " + str(self.mixedQ))
		
		# Get parameters which will let us reconstruct the Qobj at the end.
        _teststate = aqc.coh_vac_heterodyne_conditionalstate(0.5, self.alpha, self.T, self.dims, qobjQ=True, mixedQ=False)
        _mydims = _teststate.dims
        _myshape = _teststate.shape
        _newmat = np.zeros(_myshape, dtype=np.complex64)
        _myrange = np.arange(0, _myshape[0])

		# Multiprocessing and integrating
        pool = mp.Pool(mp.cpu_count())
        _newmat = np.array(pool.starmap(aqc.integrate_coh_vac_heterodyne_onequad_mpmath, [(i, j, self.alpha, self.T, self.dims, self.lim, self.delta_r, self.delta_th,  self.mixedQ) for i in _myrange for j in _myrange]))
        pool.close()

		# Reconstructing state
        _newmat = np.reshape( _newmat, [_myshape[0], _myshape[0]])
        _newstate = qt.Qobj(_newmat, dims = _mydims)
        self.state = _newstate
        return self.state
    


class entanglingcloner_attack:
    """Class to contain functions for manipulating and integrating states under entangling-cloner attack. Mainly used to allow the integration to be parallelized and called from other files. 
    """
    def __init__(self, alpha=gv.alpha, T=gv.T, barn=gv.barn, dims=gv.dims, lim=gv.lim, mixedQ=gv.mixedQ):
        self.alpha = alpha
        self.T = T
        self.barn = barn
        self.dims = dims
        self.lim = lim
        self.mixedQ = mixedQ

    def parallel_integrate(self):
        """ Integrates over entire phase-space.
        """
        print("Running entanglingcloner_attack.parallel_integrate() with alpha = " + str(self.alpha) + ", T = " + str(self.T) + ", barn = " + str(self.barn) + " dims = " + str(self.dims) + ", lim = " + str(self.lim) + "mixedQ = " + str(self.mixedQ))

		# Get parameters which will let us reconstruct the Qobj at the end.
        _teststate = aqc.coh_tmsv_heterodyne_conditionalstate(0.5, self.alpha, self.T, self.barn, self.dims, qobjQ=True, mixedQ=False)
        _mydims = _teststate.dims
        _myshape = _teststate.shape
        _newmat = np.zeros(_myshape, dtype=np.complex64)
        _myrange = np.arange(0, _myshape[0])


		# Multiprocessing and integrating
        pool = mp.Pool(mp.cpu_count())
        _newmat = np.array(pool.starmap(aqc.integrate_coh_tmsv_heterodyne, [(i, j, self.alpha, self.T, self.barn, self.dims, self.lim, self.mixedQ) for i in _myrange for j in _myrange]))
        pool.close()

		# Reconstructing state
        _newmat = np.reshape(_newmat, [_myshape[0], _myshape[0]])
        _newstate = qt.Qobj(_newmat, dims=_mydims)
        self.state = _newstate
        return self.state

    def parallel_integrate_onequad(self):
        """ Integrates over first quadrant of phase-space.
        """
        print("Running entanglingcloner_attack.parallel_integrate_onequad() with alpha = " + str(self.alpha) + ", T = " + str(self.T) + ", barn = " + str(self.barn) + " dims = " + str(self.dims) + ", lim = " + str(self.lim) + "mixedQ = " + str(self.mixedQ))

		# Get parameters which will let us reconstruct the Qobj at the end.
        _teststate = aqc.coh_tmsv_heterodyne_conditionalstate(0.5, self.alpha, self.T, self.barn, self.dims, qobjQ=True, mixedQ=False)
        _mydims = _teststate.dims
        _myshape = _teststate.shape
        _newmat = np.zeros(_myshape, dtype=np.complex64)
        _myrange = np.arange(0, _myshape[0])


		# Multiprocessing and integrating
        pool = mp.Pool(mp.cpu_count())
        _newmat = np.array(pool.starmap(aqc.integrate_coh_tmsv_heterodyne_onequad, [(i, j, self.alpha, self.T, self.barn, self.dims, self.lim, self.mixedQ) for i in _myrange for j in _myrange]))
        pool.close()

		# Reconstructing state
        _newmat = np.reshape(_newmat, [_myshape[0], _myshape[0]])
        _newstate = qt.Qobj(_newmat, dims=_mydims)
        self.state = _newstate
        return self.state


# ================================================
# Functions which actually call the classes and their methods.
# ================================================

@memory.cache
def bs_integratedstate_mpmath_PS(alpha, T, dims, lim, delta_r=0.0, delta_th=0.0, mixedQ=True):
	""" Integrate Eve's conditional state over all phase-space, when she performs a beamsplitter attack. Function mainly used for testing.
	Parameters
	----------
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		0 <= T <= 1
		Channel transmittivity.
	dims: :obj:`int`
		Size of underlying hilbert space.
	lim: :obj:`float`
		0 < lim
		Limits of integration. Ideally "lim = infinity".
	mixedQ: :obj:`bool`
		True: Input coherent state is mixed over QPSK alphabet.
		False: It isn't.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Eve's one-mode conditional state.
	"""
	ms = beamsplitter_attack(alpha, T, dims, lim, delta_r=delta_r, delta_th=delta_th, mixedQ=mixedQ)
	return ms.parallel_integrate_mpmath()

@memory.cache
def bs_integratedstate_onequad(alpha, T, dims, lim, mixedQ):
	""" Integrate Eve's conditional state over one quadrature of phase-space, when she performs a beamsplitter attack. 
	Parameters
	----------
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		0 <= T <= 1
		Channel transmittivity.
	dims: :obj:`int`
		Size of underlying hilbert space.
	lim: :obj:`float`
		0 < lim
		Limits of integration. Ideally "lim = infinity".
	mixedQ: :obj:`bool`
		True: Input coherent state is mixed over QPSK alphabet.
		False: It isn't.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Eve's one-mode conditional state.
	"""
	ms = beamsplitter_attack(alpha, T, dims, lim, mixedQ)
	return ms.parallel_integrate_onequad()

@memory.cache
def bs_integratedstate_onequad_mpmath_PS(alpha, T, dims, lim, delta_r=0.0, delta_th=0.0, mixedQ=True):
	""" Integrate Eve's conditional state over one quadrature of phase-space, when she performs a beamsplitter attack. 
	Parameters
	----------
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		0 <= T <= 1
		Channel transmittivity.
	dims: :obj:`int`
		Size of underlying hilbert space.
	lim: :obj:`float`
		0 < lim
		Limits of integration. Ideally "lim = infinity".
	mixedQ: :obj:`bool`
		True: Input coherent state is mixed over QPSK alphabet.
		False: It isn't.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Eve's one-mode conditional state.
	"""
	ms = beamsplitter_attack(alpha, T, dims, lim, delta_r=delta_r, delta_th=delta_th, mixedQ=mixedQ)
	return ms.parallel_integrate_onequad_mpmath()


@memory.cache
def bs_integratedstate_onequad_mpmath(alpha, T, dims, lim, mixedQ):
	""" Integrate Eve's conditional state over one quadrature of phase-space, when she performs a beamsplitter attack. 
	Parameters
	----------
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		0 <= T <= 1
		Channel transmittivity.
	dims: :obj:`int`
		Size of underlying hilbert space.
	lim: :obj:`float`
		0 < lim
		Limits of integration. Ideally "lim = infinity".
	mixedQ: :obj:`bool`
		True: Input coherent state is mixed over QPSK alphabet.
		False: It isn't.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Eve's one-mode conditional state.
	"""
	ms = beamsplitter_attack(alpha, T, dims, lim, mixedQ=mixedQ)
	return ms.parallel_integrate_onequad_mpmath()

@memory.cache
def ec_integratedstate(alpha, T, barn, dims, lim, mixedQ):
	""" Integrate Eve's conditional state over all phase-space, when she performs an entangling-cloner attack. Function mainly used for testing.
	Parameters
	----------
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		0 <= T <= 1
		Channel transmittivity.
	dims: :obj:`int`
		Size of underlying hilbert space.
	lim: :obj:`float`
		0 < lim
		Limits of integration. Ideally "lim = infinity".
	mixedQ: :obj:`bool`
		True: Input coherent state is mixed over QPSK alphabet.
		False: It isn't.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Eve's two-mode conditional state.
	"""
	ms = entanglingcloner_attack(alpha, T, barn, dims, lim, mixedQ)
	return ms.parallel_integrate()

@memory.cache
def ec_integratedstate_onequad(alpha, T, barn, dims, lim, mixedQ):
	""" Integrate Eve's conditional state over one quadrature of phase-space, when she performs an entangling-cloner attack. 
	Parameters
	----------
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		0 <= T <= 1
		Channel transmittivity.
	dims: :obj:`int`
		Size of underlying hilbert space.
	lim: :obj:`float`
		0 < lim
		Limits of integration. Ideally "lim = infinity".
	mixedQ: :obj:`bool`
		True: Input coherent state is mixed over QPSK alphabet.
		False: It isn't.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Eve's two-mode conditional state.
	"""
	ms = entanglingcloner_attack(alpha, T, barn, dims, lim, mixedQ)
	return ms.parallel_integrate_onequad()





# =============================================================================
# Functions for QDS1 signature length
# =============================================================================


def holevo_bs(alpha, T, dims, lim, delta_r = 0.0, delta_th = 0.0):
    """ Eve's Holevo information, under beamsplitter attack.
    Parameters
    ----------
    alpha: :obj:`float`
        Coherent state amplitude.
    T: :obj:`float`
        Channel transmittivity.
    dims: :obj:`int`
        Size of the underlying hilbert space.
    lim: :obj:`float`
        Integration limits.
    Returns
    -------
    :obj:`float`
        Eve's Holevo information.
    """
    _apriori = bs_integratedstate_mpmath_PS(alpha, T, dims, lim, delta_r=delta_r, delta_th=delta_th, mixedQ=True)
    _aposteriori = bs_integratedstate_onequad_mpmath_PS(alpha, T, dims, lim, delta_r=delta_r, delta_th=delta_th, mixedQ=True)
    
    return qt.entropy_vn(_apriori, base=2) - qt.entropy_vn(_aposteriori, base=2)
    

def holevo_ec(alpha, T, barn, dims, lim):
    """ Eve's Holevo information, under entangling-cloner attack.
    Parameters
    ----------
    alpha: :obj:`float`
        Coherent state amplitude.
    T: :obj:`float`
        Channel transmittivity.
    barn: :obj:`float`
        Channel thermal photon number.
    dims: :obj:`int`
        Size of the underlying hilbert space.
    lim: :obj:`float`
        Integration limits.
    Returns
    -------
    :obj:`float`
        Eve's Holevo information.
    """
    _apriori = aqc.coh_tmsv(alpha, T, barn, dims, test=False, mixedQ=True).ptrace([1,2])
    _aposteriori = ec_integratedstate_onequad(alpha, T, barn, dims, lim, mixedQ=True)
    
    return qt.entropy_vn(_apriori, base=2) - qt.entropy_vn(_aposteriori, base=2)
    

def binent(x):
    """ Returns the binary entropy of random variable x.
    Parameters
    ----------
    x: :obj:`float`
        0 <= x <= 1
        Random variable. 
    Returns
    -------
    :obj:`float`
        Binary entropy.
    """
    return -x * np.log2(x) - (1-x) * np.log2(1-x)

def evepe_BS(alpha, T, dims, lim, delta_r=0.0, delta_th=0.0, starting_guess=0.01):
    """ Caclulate Eve's mismatch probability when she performs a beamsplitting attack.
    Parameters
    ----------
    alpha: :obj:`float`
        Amplitude of QPSK alphabet.
    T: :obj:`float`
        0 <= T <= 1
        Channel transmittivity.
    dims: :obj:`int`
        Size of underlying Hilbert space.
    lim: :obj:`float`
        Limits of integration
    starting_guess = 0.01: :obj:`float`
        Parameter to set the starting guess of the nonlinear root solver scipy.special.fsolve()
    Returns
    -------
    :obj:`float`
        Eve's mismatch probability, pe.
        We should find 0 <= pe <= 0.5. If we find pe>0.5 it is probably because scipy.special.fsolve() is finding the wrong root of the binary entropy.
    """
    _holevo = holevo_bs(alpha, T, dims, lim, delta_r=delta_r, delta_th=delta_th)
    
#    Let's check whether Holevo is positive - and if it isn't just return 0 (plus print a message).
    
    if _holevo < 0:
        print("ERROR: Holevo = " + str(_holevo) + ", with alpha = " + str(alpha) + ", T = " + str(T) + ", dims = " + str(dims) + ", lim = " + str(lim))
        _holevo = 0
        
    
    
    _rhs = 1 - _holevo
    
    def f(x):
        """Local function to create the function of which scipy.special.fsolve() will find the roots. Basically just makes it so we are finding the roots of our function as it crosses the x axis.
        """
        return binent(x) - _rhs
    
    return fsolve(f, starting_guess)[0]



def evepe_EC(alpha, T, barn, dims, lim, starting_guess=0.01):
    """ Caclulate Eve's mismatch probability when she performs a beamsplitting attack.
    Parameters
    ----------
    alpha: :obj:`float`
        Amplitude of QPSK alphabet.
    T: :obj:`float`
        0 <= T <= 1
        Channel transmittivity.
    barn: :obj:`float`
        Channel thermal photon number.
    dims: :obj:`int`
        Size of underlying Hilbert space.
    lim: :obj:`float`
        Limits of integration
    starting_guess = 0.01: :obj:`float`
        Parameter to set the starting guess of the nonlinear root solver scipy.special.fsolve()
    Returns
    -------
    :obj:`float`
        Eve's mismatch probability, pe.
        We should find 0 <= pe <= 0.5. If we find pe>0.5 it is probably because scipy.special.fsolve() is finding the wrong root of the binary entropy.
    """
    _holevo = holevo_ec(alpha, T, barn, dims, lim)
    

#    Let's check whether Holevo is positive - and if it isn't just return 0 (plus print a message).
    
    if _holevo < 0:
        print("ERROR: Holevo = " + str(_holevo) + ", with alpha = " + str(alpha) + ", T = " + str(T) + ", dims = " + str(dims) + ", lim = " + str(lim))
        _holevo = 0
        
    
    _rhs = 1 - _holevo
    
    def f(x):
        """Local function to create the function of which scipy.special.fsolve() will find the roots. Basically just makes it so we are finding the roots of our function as it crosses the x axis.
        """
        return binent(x) - _rhs
    
    return fsolve(f, starting_guess)[0]


def secparamg_BS(alpha, T, dims, lim, delta_r=0.0, delta_th=0.0, starting_guess=0.01):
    """ Calculate the security parameter g := perr - pe when Eve uses a beamsplitter attack.
        Parameters
    ----------
    alpha: :obj:`float`
        Amplitude of QPSK alphabet.
    T: :obj:`float`
        0 <= T <= 1
        Channel transmittivity.
    dims: :obj:`int`
        Size of underlying Hilbert space.
    lim: :obj:`float`
        Limits of integration
    delta_r: :obj:`float`
        0 <= delta_r 
        Specifies radial size of postselection region.
    delta_th: :obj:`float`
        0 <= delta_th < pi/4
        Specifies angular size of postselection region.
    starting_guess = 0.01: :obj:`float`
        Parameter to set the starting guess of the nonlinear root solver scipy.special.fsolve()
    Returns
    -------
    :obj:`float`
        Security parameter g := pe - perr.
    """
    _pe = evepe_BS(alpha, T, dims, lim, delta_r=delta_r, delta_th=delta_th, starting_guess=starting_guess)
    _perr = aqc.perr_PS(alpha, T, 0.0, delta_r=delta_r, delta_th=delta_th)
    
    return _pe - _perr


def secparamg_EC(alpha, T, barn, dims, lim, delta_r=0.0, delta_th=0.0, starting_guess=0.01):
    """ Calculate the security parameter g := perr - pe when Eve uses an entangling cloner attack.
        Parameters
    ----------
    alpha: :obj:`float`
        Amplitude of QPSK alphabet.
    T: :obj:`float`
        0 <= T <= 1
        Channel transmittivity.
    barn: :obj:`float`
        Channel thermal photon number.
    dims: :obj:`int`
        Size of underlying Hilbert space.
    lim: :obj:`float`
        Limits of integration
    
    starting_guess = 0.01: :obj:`float`
        Parameter to set the starting guess of the nonlinear root solver scipy.special.fsolve()
    Returns
    -------
    :obj:`float`
        Security parameter g := pe - perr.
    """
    _pe = evepe_EC(alpha, T, barn, dims, lim, starting_guess)
    _perr = aqc.perr_PS(alpha, T, 0.0, delta_r=delta_r, delta_th=delta_th)
    
    return _pe - _perr


def siglength_BS(alpha, T, dims, lim, delta_r=0.0, delta_th=0.0, failprob=0.0001):
    """ Calculate the signature length when Eve uses a beamsplitter attack. If protocol is insecure (i.e. if g < 0) then prints an error message and returns value -1.
    Parameters
    ----------
    alpha: :obj:`float`
        Amplitude of QPSK alphabet.
    T: :obj:`float`
        0 <= T <= 1
        Channel transmittivity.
    dims: :obj:`int`
        Size of underlying Hilbert space.
    lim: :obj:`float`
        Limits of integration
    delta_r: :obj:`float`
        0 <= delta_r 
        Specifies radial size of postselection region.
    delta_th: :obj:`float`
        0 <= delta_th < pi/4
        Specifies angular size of postselection region.
    failprob=0.0001: :obj:`float`
        Total probability that the protocol fails. We want failprob to be small. Default value 0.0001 corresponds to 0.01% chance of failure.
    Returns
    -------
    :obj:`float`
        Signature length L. Returns -1 if the protocol is insecure.
    """
    
    _g = secparamg_BS(alpha, T, dims, lim, delta_r=delta_r, delta_th=delta_th)
    _pnorm = aqc.pnorm(alpha, T, 0.0, delta_r, delta_th)
    
    
    if _g >= 0:
        return (1/_pnorm) * (- 16 / (_g**2)) * np.log(failprob/2) # gotta multiply by 1/_pnorm to correctly take into account states which are thrown away.
    else:
        print("Protocol insecure for alpha = " + str(alpha) + ", T = " + str(T) + ", dims = " + str(dims) + ", lim = " + str(lim))
        return -1


def siglength_EC(alpha, T, barn, dims, lim, delta_r=0.0, delta_th=0.0, failprob=0.0001):
    """ Calculate the signature length when Eve uses a beamsplitter attack. If protocol is insecure (i.e. if g < 0) then prints an error message and returns value -1.
    Parameters
    ----------
    alpha: :obj:`float`
        Amplitude of QPSK alphabet.
    T: :obj:`float`
        0 <= T <= 1
        Channel transmittivity.
    barn: :obj:`float`
        Channel thermal photon number.
    dims: :obj:`int`
        Size of underlying Hilbert space.
    lim: :obj:`float`
        Limits of integration
    delta_r: :obj:`float`
        0 <= delta_r 
        Specifies radial size of postselection region.
    delta_th: :obj:`float`
        0 <= delta_th < pi/4
        Specifies angular size of postselection region.
    failprob=0.0001: :obj:`float`
        Total probability that the protocol fails. We want failprob to be small. Default value 0.0001 corresponds to 0.01% chance of failure.
    Returns
    -------
    :obj:`float`
        Signature length L. Returns -1 if the protocol is insecure.
    """
    
    _g = secparamg_EC(alpha, T, barn, dims, lim, delta_r=delta_r, delta_th=delta_th)
    _pnorm = aqc.pnorm(alpha, T, barn, delta_r, delta_th)
    
    if _g >= 0:
        return (1/_pnorm) * (- 16 / (_g**2)) * np.log(failprob)# gotta multiply by 1/_pnorm to correctly take into account states which are thrown away.
    else:
        print("Protocol insecure for alpha = " + str(alpha) + ", T = " + str(T) + ", dims = " + str(dims) + ", lim = " + str(lim))
        return -1
    
