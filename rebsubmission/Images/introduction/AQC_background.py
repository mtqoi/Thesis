# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:15:23 2019

@author: Matthew Thornton

Background functions required for implementing BS and EC attacks in QDS1, QDS2, QSS
"""


# ================================================
#	Import statements
# ================================================

import qutip as qt 
import scipy.integrate as integrate 
import numpy as np 
qt.settings.auto_tidyup = False
qt.settings.atol = 1e-18
import os
import math
import mpmath as mp
mp.dps=15
mp.pretty=True
from scipy.special import binom as binom

cachedir = os.getcwd() + "\\" + "cache\\"
if os.path.isdir(cachedir) == False:
	os.mkdir(cachedir)
# os.path.isdir(cachedir)
from joblib import Memory
memory = Memory(cachedir, verbose=0)
from scipy.special import erf as erf
from scipy.special import erfc as erfc

thresh = 1E-5


# =============================================================================
# Functions for modelling single coherent state through channel
# =============================================================================
def pb_ak(b, alpha_k, T, barn):
    """ Calculates the probability of a heterodyne measurement giving complex outcome, b, when a coherent state amplitude alpha_k is sent through the channel parameterised by transmittivity T and thermal photon number barn.
    Parameters
    ----------
    b : :obj:`complex`
        Heterodyne outcome b.
    alpha_k : :obj:`complex`
        Complex coherent state amplitude.
    T : :obj:`float`
        Channel transmittivity.
    barn : :obj:`float
        Channel thermal photon number.
    Returns
    -------
    obj:`float`
        Normalized probability of measuring b.    
    """
    
    _numerator =  - np.abs(b - np.sqrt(T) * alpha_k)**2
    _denominator = 1 + (1-T) * barn
    return (1 / (np.pi * _denominator)) * np.exp((_numerator/_denominator))


def pb_ak_mpmath(x, y, alpha_k, T, barn):
    """ Same a pb_ak but uses mpmath library. Same as pb_ak_mpmath2 but split into real and imaginary parts to see if it provides any performance increase.
    """
    _alphaR = mp.re(alpha_k)
    _alphaI = mp.im(alpha_k)

    _numerator = - (x - mp.sqrt(T) * _alphaR)**2 - (y - mp.sqrt(T) * _alphaI)**2
    _denominator = 1 + (1-T) * barn
    return (1 / (mp.pi * _denominator)) * mp.exp((_numerator/_denominator))

def pbak_mpmath_polars_intfun(alpha, T, barn):
    """ pb_ak in polars for integration. Should be the same as pk_ak_mpmath2 but in polars.
    """
    _alphaR = mp.re(alpha)
    _alphaI = mp.im(alpha)

    # _numerator =  - (r * mp.cos(th) - _alphaR * mp.sqrt(T))**2 - (r * mp.sin(th) - _alphaI * mp.sqrt(T))**2
    _denominator = 1 + (1-T)*barn

    return lambda r, th: r * (1/( mp.pi * _denominator)) * mp.exp(- (r * mp.cos(th) - _alphaR * mp.sqrt(T))**2 - (r * mp.sin(th) - _alphaI * mp.sqrt(T))**2/_denominator)


# def pb_ak_mpmath2(b, alpha_k, T, barn):
#     """ Same as pb_ak but uses mpmath library.
#     """
#     _numerator = - mp.fabs(b - mp.sqrt(T) * alpha_k)**2
#     _denominator = 1 + (1-T) * barn
#     return (1 / (mp.pi * _denominator)) * mp.exp((_numerator/_denominator))



# ================================================
# Functions for creating states after channel (before measurement)
# ================================================


def coh_T(N, alpha, T, k, dims):
    """ Returns a coherent state chosen from NPSK alphabet.
    Parameters
    ----------
    N : :obj:`int`
        Number of coherent states in our NPSK alphabet.
    alpha : :obj:`complex`
        Coherent state amplitude input. 
            Note that when e.g. purely real alpha specified, QuTip will return a coherent state whose position expectation value is sqrt(2) * alpha. And analogously for purely imaginary alpha and momentum expectation value.
    T : :obj:`float`
        Beamsplitter transmittivity.
    k : :obj:`int`
        Label 0 <= k <= N-1 for which coherent state to return.
    dims : :obj:`int`
        Dimension of hilbert space. Make sure to choose dims >= 2 |alpha|^2. 
    Returns
    -------
    :obj:`QuTip.Qobj`
        Coherent state.
    """
    if k < N:
        _phi_k = (2 * np.pi) * (k/N)
        _amplitude = alpha  * np.sqrt(T) * np.exp(1.0J * _phi_k)
        return qt.coherent_dm(dims, _amplitude, method="analytic")
    else: 
        print("Invalid k specified. k = " + str(k) + "must be an integer strictly less than " + str(N))
        return 


def one_coh_vac_bs(alphak, T, dims):
    """A single coherent state and vacuum interfere on beamsplitter.
    Parameters
    ----------
    alphak: :obj:`complex`
    	Input coherent state amplitude.
    T: :obj:`float`
    	0 <= T <=1
    	Channel transmittivity.
    dims: :obj:`int`
    	Underlying hilbert space dimension.
    Returns
    -------
    :obj:`QuTip.Qobj`
    	Two-mode output state after coherent state passes through channel.
    """
    prefactor = np.exp(-(1/2) * np.abs(alphak)**2)
    _blankarray = np.zeros(dims, dtype="object")
    
    for n in np.arange(0, dims):
        _blankarray[n] = prefactor * ((alphak**n) / (math.sqrt(math.factorial(n)))) * n1n2out(n, 0, T, dims)
        
    return qt.ket2dm(np.sum(_blankarray))

@memory.cache
def coh_vac_bs(alpha, T, dims, mixedQ=True):
    """Mixes one_coh_vac_bs over the four QPSK alphabet states.
    Parameters
    ----------
    alpha: :obj:`float`
    	Amplitude of QPSK alphabet.
    T: :obj:`float`
    	0 <= T <= 1
    	Channel transmittivity.
    dims: :obj:`int`
    	Underlying hilbert space dimension
    mixedQ: :obj:`bool`
        True: Mix over QPSK alphabet.
        False: Just return a single coherent state with amplitude alpha.
    Returns
    -------
    :obj:`QuTip.Qobj`
    	Normalized two-mode output state, after the mixture of QPSK alphabet coherent states passes through the channel.
    """

    if mixedQ==False:
        return one_coh_vac_bs(alpha, T, dims)
    elif mixedQ==True:
        _statesarray = np.zeros(4, dtype="object")
        _statesarray[0] = one_coh_vac_bs(1.0 * alpha, T, dims)
        _statesarray[1] = one_coh_vac_bs(1.0J * alpha, T, dims)
        _statesarray[2] = one_coh_vac_bs(-1.0 * alpha, T, dims)
        _statesarray[3] = one_coh_vac_bs(-1.0J * alpha, T, dims)
        
        return (1/4) * np.sum(_statesarray)



####################
# Creating the TMSV
####################

def barn2zeta(barn):
    """
    Converts average thermal photon number, barn,  in the thermal state (reduced state of tmsv) to the squeezing parameter \zeta required to give this level of barn.
    barn = np.sinh(zeta)**2
    therefore zeta = np.arcsinh(np.sqrt(barn)).
    Parameters
    ----------
    barn: :obj:`float`
    	Thermal photon number.
    Returns
    -------
    :obj:`float`
    	Squeezing parameter zeta which gives the correct barn.
    """
    return np.arcsinh(np.sqrt(barn))

def b11(T):
    """Calculate elements of beamsplitter matrix. See Ulf Eq.5.9.
    Parameters
    ----------
    T:   :obj:`float`
        The beamsplitter transmittivity.
    Returns
    -------
    :obj:`float`
        Element of beamsplitter matrix.
    """
    
    return np.sqrt(T)

def b21(T):
    """Calculate elements of beamsplitter matrix. See Ulf Eq.5.9.
    Parameters
    ----------
    T:   :obj:`float`
        The beamsplitter transmittivity.
    Returns
    -------
    :obj:`float`
        Element of beamsplitter matrix.
    """
    
    return np.sqrt(1-T)

def b12(T):
    """Calculate elements of beamsplitter matrix. See Ulf Eq.5.9.
    Parameters
    ----------
    T:   :obj:`float`
        The beamsplitter transmittivity.
    Returns
    -------
    :obj:`float`
        Element of beamsplitter matrix.
    """
    
    return -np.sqrt(1-T)

def b22(T):
    """Calculate elements of beamsplitter matrix. See Ulf Eq.5.9.
    Parameters
    ----------
    T:   :obj:`float`
        The beamsplitter transmittivity.
    Returns
    -------
    :obj:`float`
        Element of beamsplitter matrix.
    """
    
    return np.sqrt(T)



def sumcoeff(n1, n2, k, l, T):
    """Gives the coefficient which multiplies |k+l, n1+n2-k-l> in my expression for the beamsplitter transform. See Ulf Eq.5.43.
    Parameters
    ----------
    n1 : :obj:`int`
        Number of photons incident on beamsplitter in mode 1.
    n2 : :obj:`int`
        Number of photons incident on beamsplitter in mode 2.
    k : :obj:`int`
        Dummy-index counting mode 1. Must be <= n1.
    l : :obj:`int`
        Dummy-index counting mode 2. Must be <= n2.
    T : :obj:`float`
        The beamsplitter transmittivity.
    Returns
    -------
    :obj:`float`
        Coefficient which multiplies |k+l, n1+n2-k-l>.
    """
    
    numerator = binom(n1, k) * binom(n2, l) * (b11(T)**k) * (b12(T)**l) * (b21(T)**(n1-k)) * (b22(T)**(n2-l)) * math.sqrt(math.factorial(k + l)) * math.sqrt(math.factorial(n1 + n2 - k - l))
    denominator = math.sqrt(math.factorial(n1) * math.factorial(n2))
    return numerator/denominator

def ketkl(n1, n2, k, l, dims):
    """
Returns ket corresponding to (k+l) photons in mode 1 and n1 + n2 - k - l photons in mode 2.
    Parameters
    ----------
    n1 : :obj:`int`
        Number of photons incident on beamsplitter in mode 1.
    n2 : :obj:`int`
        Number of photons incident on beamsplitter in mode 2.
    k : :obj:`int`
        Dummy-index counting mode 1. Must be <= n1.
    l : :obj:`int`
        Dummy-index counting mode 2. Must be <= n2.
    dims : :obj:`int`
        Hilbert-space dimension on each mode. Require dims > n1+n2.
    Returns
    -------
    :obj:`QuTip.Qobj`
        Two-mode quantum object containing (k+l) photons in mode 1 and (n1+n2-k-l) photons in mode 2.
    """
    
    ket1 = qt.basis(dims, k + l)
    ket2 = qt.basis(dims, n1 + n2 - k - l)
    return qt.tensor(ket1, ket2)


def outstatekl(n1, n2, k, l, T, dims):
    """ Gives product of sumcoeff * |k+l, n1+n2-k-l>
    Parameters
    ----------
    n1 : :obj:`int`
        Number of photons incident on beamsplitter in mode 1.
    n2 : :obj:`int`
        Number of photons incident on beamsplitter in mode 2.
    k : :obj:`int`
        Dummy-index counting mode 1. Must be <= n1.
    l : :obj:`int`
        Dummy-index counting mode 2. Must be <= n2.
    T : :obj:`float`
        The beamsplitter transmittivity
    dims : :obj:`int`
        Hilbert-space dimension on each mode. Require dims > n1+n2.
    Returns
    -------
    :obj:`QuTip.Qobj`
        Product of :obj:`int` sumcoeff and :obj:`QuTip.Qobj` ketkl.
    Useful for generating state after beamsplitter. See sumcoeff, ketkl, n1n2out.
    """
    
    return sumcoeff(n1, n2, k, l, T) * ketkl(n1, n2, k, l, dims)



@memory.cache
def n1n2out(n1, n2, T, dims):
    """ Returns the state after beamsplitter transmittivity T when the state |n1, n2> is incident on the beamsplitter. See Ulf. Eq.5.43.
    Function n1n2out is decorated with memory.cache from joblib package to allow for cacheing and persistence. This will greatly improve performance when repeated calls to n1n2out are required, e.g. when integrating for calculations of aposteriori entropies. It will also allow persistence between kernel restarts.
    Parameters
    ----------
    n1 : :obj:`int`
        Number of photons incident on beamsplitter in mode 1.
    n2 : :obj:`int`
        Number of photons incident on beamsplitter in mode 2.
    T : :obj:`float`
        The beamsplitter transmittivity
    dims : :obj:`int`
        Hilbert-space dimension on each mode. Require dims > n1+n2.
    Returns
    -------
    :obj:`QuTip.Qobj`
        Output of beamsplitter.
    """
    _blankarray = np.zeros([n2+1, n1+1], dtype="object")
    
    if dims <= n1+n2:
        print("Dims less than n1+n2. Error.")
        return None
    else:
        for k in np.arange(n1+1):
            for l in np.arange(n2+1):
                _blankarray[l][k] = outstatekl(n1, n2, k, l, T, dims)
        return np.sum(_blankarray)


@memory.cache
def tmsv(barn, dims):
    """ Calculates the two-mode TMSV function.
	Parameters
	----------
	barn: :obj:`float`
		Desired thermal photon umber.
	dims: :obj:`int`
		Hilbert space size of each mode. Total hilbert space size is dims**2.
	Returns
	-------
	:obj:`QuTip.Qobj`
		TMSV state.
	"""
    _zeta = barn2zeta(barn)
    _prefactor = 1 / (np.cosh(_zeta)**2)
    _blankarray = np.zeros([dims, dims], dtype=object)
    
    for n in np.arange(0, dims-1):
        for m in np.arange(0, dims-1):
            _blankarray[n, m] = (np.tanh(_zeta)**(n + m)) * qt.tensor(qt.basis(dims, n), qt.basis(dims, n)) * qt.tensor(qt.basis(dims, m), qt.basis(dims, m)).trans()
    
    return np.sum(_prefactor * _blankarray)  


@memory.cache
def coh_tmsv_one(alpha, T, barn, dims, test=False):
    """Calculates the three-mode output state when a coherent state and one arm of a tmsv interfere on beamsplitter. Function coh_tmsv is decorated with memory.cache from joblib package to allow for cacheing and persistence. This will greatly improve performance when repeated calls to coh_tmsv are required, e.g. when integrating for calculations of aposteriori entropies. It will also allow persistence between kernel restarts.
    Parameters
    ----------
    alpha : :obj:`complex` 
        Amplitude of input coherent state.
    T : :obj:`float`
        Beampspitter transmittivity.
    barn : :obj:`float`
        Number of thermal photons in one-mode reduced state of incident TMSV.
    dims : :obj:`int`
        Hilbert space dimension.
    test: :obj:`bool`
        Specify whether to test normalization of output state.
    Returns
    -------
    :obj:`QuTip.Qobj`
        Quantum-object containing three-mode output state.
    """
    
    zeta = barn2zeta(barn)
    
    prefactor = np.exp(-np.abs(alpha)**2)/(np.cosh(zeta)**2)
    
    _blankarray = np.zeros([dims, dims, dims, dims], dtype="object")
    
    for n in np.arange(dims):
        for m in np.arange(dims):
            for nprime in np.arange(dims):
                for mprime in np.arange(dims):
                    if n + nprime < dims and m + mprime < dims:
                        _blankarray[n][m][nprime][mprime] = ((alpha**(n) * np.conj(alpha)**(m))/(math.sqrt(math.factorial(n) * math.factorial(m)))) * (np.tanh(zeta)**(nprime+mprime)) * qt.tensor(n1n2out(n, nprime, T, dims), qt.basis(dims, nprime)) * qt.tensor(n1n2out(m, mprime, T, dims), qt.basis(dims, mprime)).trans()
                    else:
                        None
    _state =  prefactor * np.sum(_blankarray)
    if test == True:
        assert _state.ptrace(0).norm() >= 1.0 - thresh and _state.ptrace(1).norm() >= 1.0 - thresh and _state.ptrace(2).norm() >= 1.0-thresh
        return _state
    else:
         return _state

@memory.cache
def coh_tmsv(alpha, T, barn, dims, test=False, mixedQ=True):
    """ Returns either coh_tmsv, or coh_tmsv mixed over QPSK alphabet states.
    Parameters
    ----------
    alpha : :obj:`complex` 
        Amplitude of input coherent state.
    T : :obj:`float`
        Beampspitter transmittivity.
    barn : :obj:`float`
        Number of thermal photons in one-mode reduced state of incident TMSV.
    dims : :obj:`int`
        Hilbert space dimension.
    test: :obj:`bool`
        Specify whether to test normalization of output state.
    mixedQ: :obj:`bool`
        True: mix over four QPSK alphabet states.
        False: just put a single coherent state, amplitude alpha, into the channel.
    Returns
    -------
    :obj:`QuTip.Qobj`
        Quantum-object containing three-mode output state.
    """
    if mixedQ==False:
        return coh_tmsv_one(alpha, T, barn, dims, test)
    elif mixedQ==True:
        _statesarray = np.zeros(4, dtype="object")
        _statesarray[0] = coh_tmsv_one(1.0 * alpha, T, barn, dims, test)
        _statesarray[1] = coh_tmsv_one(1.0J * alpha, T, barn, dims, test)
        _statesarray[2] = coh_tmsv_one(-1.0 * alpha, T, barn, dims, test)
        _statesarray[3] = coh_tmsv_one(-1.0J * alpha, T, barn, dims, test)
        
        return (1/4) * np.sum(_statesarray)


# ================================================
# ================================================
# ================================================
# Functions for measurements
# ================================================
# ================================================
# ================================================

# ================================================
# Define my measurement projectors
# ================================================


def hetproj_onemode(b, dims):
	""" One-mode projector onto coherent state |b><b|. Required for computing heterodyne outcome probablities.
	The probability of heterodyne measurement returning complex amplitude b is then given by:
		(hetproj_onemode(b, dims) * rho * hetproj_onemode(b, dims).dag()).tr()
	Parameters
	----------
	b: :obj:`complex`
		Complex amplitude on which to project.
	dims: :obj:`int`
		Hilbert space size.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Quantum-object containing the projector.
	"""
	return (1/np.sqrt(np.pi)) * qt.coherent_dm(dims, b, method="analytic")


def hetproj_twomode(b, dims):
    """ Gives me the two-mode projector for performing heterodyne detection on mode 1.
	Parameters
	----------
	b: :obj:`complex`
		Complex amplitude on which to project.
	dims: :obj:`int`
		Hilbert space size.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Quantum-object containing the projector.
    """
    _PiB = (1/np.sqrt(np.pi)) * qt.coherent_dm(dims, b, method="analytic")
    _id = qt.qeye(dims)
    return qt.tensor(_PiB, _id)



def hetproj_threemode(b, dims):
    """ Gives me the three-mode projector for performing heterodyne detection on mode 1.
   	Parameters
	----------
	b: :obj:`complex`
		Complex amplitude on which to project.
	dims: :obj:`int`
		Hilbert space size.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Quantum-object containing the projector.
    """
    _PiB = (1/np.sqrt(np.pi)) * qt.coherent_dm(dims, b, method="analytic")
    _id = qt.qeye(dims)
    return qt.tensor(_PiB, _id, _id)



# ================================================
# Performing the measurements
# ================================================

#########
# TMSV
#########

def tmsv_heterodyne_outcome(b, barn, dims):
    """Probability of receiving heterodyne outcome b when the first mode of a TMSV is measured.
	Parameters
	----------
	b: :obj:`complex`
		Heterodyne outcome.
	barn: :obj:`float`
		Number of thermal photons in my TMSV
	dims: :obj:`int`
		Underlying dimension of hilbert-space of one arm of the TMSV.
	Returns
	-------
	:obj:`float`
		Real-valued probability of measuring b.
	"""
    _rho = tmsv(barn, dims)
    
    _Pi = hetproj_twomode(b, dims)
    
    _projected = (_Pi * _rho * _Pi.dag())
    _outcome = _projected.ptrace(0).tr()
    
    return np.real(_outcome)

def tmsv_heterodyne_conditionalstate(b, barn, dims, qobjQ=True):
	""" Normalized conditional state when the first mode of a TMSV is measured by heterodyne detection and receives outcome b.
	Parameters
	----------
	b: :obj:`complex`
		Heterodyne outcome.
	barn: :obj:`float`
		Number of thermal photons in my TMSV
	dims: :obj:`int`
		Underlying dimension of hilbert-space of one arm of the TMSV.
	qobjQ: :obj:`bool`
		True: output is Qobj.
		False: output is numpy array.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Normalized one-mode conditional state.
	or
	:obj:`numpy.array`	
		Numpy array corresponding to normalized one-mode conditional state.
	"""
	_rho = tmsv(barn, dims)
	_Pi = hetproj_twomode(b, dims)

	_projected = (_Pi * _rho * _Pi.dag())
	_outcome = _projected.ptrace(0).tr()
	_condstate = _projected.ptrace(1)
	_normalized = (1 / _outcome) * _condstate

	return _normalized

#########
# coh-tmsv (Entangling-cloner attack)
#########

@memory.cache
def coh_tmsv_heterodyne_outcome(b, alphak, T, barn, dims, mixedQ=True):
    """Mix a coherent state and TMSV on a beamsplitter, then perform heterodyne measurement on the first output mode. What is the probablity of receiving heterodyne outcome b?
	Parameters
	----------
	b: :obj:`complex`
		Heterodyne outcome.
	alphak: :obj:`complex`
		Input coherent state amplitude.
	T: :obj:`float`
		0 <= T <=1. 
		Channel transmittivity.
	barn: :obj:`float`
		Number of thermal photons in my TMSV
	dims: :obj:`int`
		Underlying dimension of hilbert-space of one arm of the TMSV.
    mixedQ: :obj:`bool`
        True: mix coherent states over QPSK alphabet.
        False: don't.
	Returns
	-------
	:obj:`float`
		Real-valued probability of measuring b.
	"""
    _rho = coh_tmsv(alphak, T, barn, dims, mixedQ)
    
    _Pi = hetproj_threemode(b, dims)
    
    _projected = (_Pi * _rho * _Pi.dag())
    _outcome = _projected.ptrace(0).tr()
    
    return np.real(_outcome)

@memory.cache
def coh_tmsv_heterodyne_conditionalstate(b, alphak, T, barn, dims, qobjQ=True, mixedQ=True):
    """ Mix a coherent state and TMSV on a beamsplitter, then perform heterodyne measurement on the first output mode. What is the probablity of receiving heterodyne outcome b?
	Parameters
	----------
	b: :obj:`complex`
		Heterodyne outcome.
	alphak: :obj:`complex`
		Input coherent state amplitude.
	T: :obj:`float`
		0 <= T <=1. 
		Channel transmittivity.
	barn: :obj:`float`
		Number of thermal photons in my TMSV
	dims: :obj:`int`
		Underlying dimension of hilbert-space of one arm of the TMSV.
	qobjQ: :obj:`bool`
		True: output is Qobj.
		False: output is numpy array.
    mixedQ: :obj:`bool`
        True: mix coherent states over QPSK alphabet.
        False: don't.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Two-mode output conditional state.
	or
	:obj:`numpy.array`
		Numpy array corresponding to two-mode output conditional state.
	"""
    _rho = coh_tmsv(alphak, T, barn, dims, mixedQ=mixedQ)
    _Pi = hetproj_threemode(b, dims)
    
    _projected = (_Pi * _rho * _Pi.dag())
    _outcome = np.real(_projected.ptrace(0).tr())
    
#    if qobjQ==True:
#    	return (_projected.ptrace([1,2])/_outcome)
    if qobjQ==True:
        return _projected.ptrace([1,2])
    elif qobjQ==False:
    	return (_projected.ptrace([1,2])/_outcome).full()


#########
# coh-vac (Beamsplitter attack)
#########


#@memory.cache
def coh_vac_heterodyne_outcome(b, alpha, T, dims, mixedQ=True):
    """ Mix QPSK-mixed coherent states and vacuum on a beamsplitter, then perform heterodyne measurement on the first output mode. What is the probablity of receiving heterodyne outcome b?
	Parameters
	----------
	b: :obj:`complex`
		Heterodyne outcome.
	alphak: :obj:`complex`
		Input coherent state amplitude.
	T: :obj:`float`
		0 <= T <=1. 
		Channel transmittivity.
	dims: :obj:`int`
		Underlying dimension of hilbert-space of one arm of the TMSV.
    mixedQ: :obj:`bool`
        True: mix coherent states over QPSK alphabet.
        False: don't.
	Returns
	-------
	:obj:`float`
		Real-valued probability of measuring b.
	"""
    _rho = coh_vac_bs(alpha, T, dims, mixedQ)
    
    _Pi = hetproj_twomode(b, dims)
    
    _projected = (_Pi * _rho * _Pi.dag())
    _outcome = _projected.ptrace(0).tr()
    
    return np.real(_outcome)

@memory.cache
def coh_vac_heterodyne_conditionalstate(b, alpha, T, dims, qobjQ=True, mixedQ=True):
    """ Mix QPSK-mixed coherent states and vacuum on a beamsplitter, then perform heterodyne maesurement on the first ouput mode. If measurement outcome b is received, what is the conditional state in the other mode?
	Parameters
	----------
	b: :obj:`complex`
		Heterodyne outcome.
	alphak: :obj:`complex`
		Input coherent state amplitude.
	T: :obj:`float`
		0 <= T <=1. 
		Channel transmittivity.
	dims: :obj:`int`
		Underlying dimension of hilbert-space of one arm of the TMSV.
	qobjQ: :obj:`bool`
		True: output is a Qobj
		False: output is a numpy array.
    mixedQ: :obj:`bool`
        True: mix coherent states over QPSK alphabet.
        False: don't.
	Returns
	-------
	:obj:`QuTip.Qobj`
		Normalized one-mode conditional state.
	or
	:obj:`numpy.array`
		Numpy array corresponding to normalized one-mode conditional state.
	"""
    _rho = coh_vac_bs(alpha, T, dims, mixedQ)
    
    _Pi = hetproj_twomode(b, dims)
    
    _projected = (_Pi * _rho * _Pi.dag())
    _outcome = _projected.ptrace(0).tr()
#    
    if qobjQ==True:
    	return (_projected.ptrace(1)/_outcome)
#    if qobjQ==True:
#        return _projected.ptrace(1)
    elif qobjQ==False:
    	return (_projected.ptrace(1)/_outcome).full()




# ================================================
# Integrating over phase space regions.
# 	The following functions are only really for QDS-1.
# ================================================   



########
# TMSV
########

#@memory.cache
def integrate_tmsv_heterodyne(i, j, barn, dims, lim):
    """ Integrate matrix element i, j of tmsv_heterodyne_conditionalstate() over all phase space.
	Parameters
	----------
	i, j: :obj:`int`
		0 <= i, j < dims
		Specifies which matrix element I am interested in.
	barn: :obj:`float`
		Thermal photon number of my TMSV.
	dims: :obj:`int`
		Underlying hilbert space dimension.
	lim: :obj:`float`
		0 < lim
		x and y limit of my integration.
	Returns
	-------
	:obj:`complex`
		Complex value of matrix element i, j after integration, ready to be made into a quantum state.
	"""
    _intreal =  integrate.dblquad(lambda x, y: tmsv_heterodyne_outcome(x + 1.0J * y, barn, dims) * np.real( tmsv_heterodyne_conditionalstate(x + 1.0J * y, barn, dims, qobjQ=False)[i, j]), -lim, lim, lambda y: -lim, lambda y: lim)[0]
    
    _intimag =  integrate.dblquad(lambda x, y: tmsv_heterodyne_outcome(x + 1.0J * y, barn, dims) * np.imag( tmsv_heterodyne_conditionalstate(x + 1.0J * y, barn, dims, qobjQ=False)[i, j]), -lim, lim, lambda y: -lim, lambda y: lim)[0]
    
    return _intreal + 1.0J * _intimag
    


########
# coh-tmsv (entangling cloner)
########

#@memory.cache
def integrate_coh_tmsv_heterodyne(i, j, alpha, T,  barn, dims, lim, mixedQ=True):
    """ Integrate matrix element i, j of coh_tmsv_heterodyne_conditionalstate() over all phase space.
	Parameters
	----------
	i, j: :obj:`int`
		0 <= i, j < dims
		Specifies which matrix element I am interested in.
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		Channel transmittivity.
	barn: :obj:`float`
		Thermal photon number of my TMSV.
	dims: :obj:`int`
		Underlying hilbert space dimension.
	lim: :obj:`float`
		0 < lim
		x and y limit of my integration.
    mixedQ: :obj:`bool`
        True: mix coherent states over QPSK alphabet.
        False: don't.
	Returns
	-------
	:obj:`complex`
		Complex value of matrix element i, j after integration, ready to be made into a quantum state.
	"""

    _intreal =  integrate.dblquad(lambda x, y: coh_tmsv_heterodyne_outcome(x + 1.0J * y, alpha, T, barn, dims, mixedQ) * np.real(coh_tmsv_heterodyne_conditionalstate(x + 1.0J * y, alpha, T, barn, dims, qobjQ=False, mixedQ=mixedQ)[i, j]), -lim, lim, lambda y: -lim, lambda y: lim)[0]
    _intimag =  integrate.dblquad(lambda x, y: coh_tmsv_heterodyne_outcome(x + 1.0J * y, alpha, T, barn, dims, mixedQ) * np.imag(coh_tmsv_heterodyne_conditionalstate(x + 1.0J * y, alpha, T, barn, dims, qobjQ=False, mixedQ=mixedQ)[i, j]), -lim, lim, lambda y: -lim, lambda y: lim)[0]
    return _intreal + _intimag 

def probfactor(alpha, T, barn):
    """ Probability factor, useful for re-normalizing states after integrating over just one quadrature. Found by integrating one coherent state over just one quadrature, in mathematica.
    """
    return (1/4) * (1 + erf(np.sqrt( T / (1 + (1-T) * barn)) * alpha))

#@memory.cache
def integrate_coh_tmsv_heterodyne_onequad(i, j, alpha, T,  barn, dims, lim, mixedQ=True):
    """ Integrate matrix element i, j of coh_tmsv_heterodyne_conditionalstate() over one quadrature of phase space.
	Parameters
	----------
	i, j: :obj:`int`
		0 <= i, j < dims
		Specifies which matrix element I am interested in.
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		Channel transmittivity.
	barn: :obj:`float`
		Thermal photon number of my TMSV.
	dims: :obj:`int`
		Underlying hilbert space dimension.
	lim: :obj:`float`
		0 < lim
		x and y limit of my integration.
    mixedQ: :obj:`bool`
        True: mix coherent states over QPSK alphabet.
        False: don't.
	Returns
	-------
	:obj:`complex`
		Complex value of matrix element i, j after integration, ready to be made into a quantum state.
	"""
    
    if mixedQ==True:
        _probfactor = (1/4)
    elif mixedQ==False:
        _probfactor = probfactor(alpha, T, 0.0)
        
    _intreal =  (1/_probfactor) * integrate.dblquad(lambda x, y: coh_tmsv_heterodyne_outcome(x + 1.0J * y, alpha, T, barn, dims, mixedQ) * np.real(coh_tmsv_heterodyne_conditionalstate(x + 1.0J * y, alpha, T, barn, dims, qobjQ=False, mixedQ=mixedQ)[i, j]), 0, lim, lambda y: 0, lambda y: lim)[0]
    
    _intimag =  (1/_probfactor) * integrate.dblquad(lambda x, y: coh_tmsv_heterodyne_outcome(x + 1.0J * y, alpha, T, barn, dims, mixedQ) * np.imag(coh_tmsv_heterodyne_conditionalstate(x + 1.0J * y, alpha, T, barn, dims, qobjQ=False, mixedQ=mixedQ)[i, j]), 0, lim, lambda y: 0, lambda y: lim)[0]
    
    return _intreal + 1.0J * _intimag

########
# coh-vac (beamsplitter)
########

#@memory.cache
#def integrate_coh_vac_heterodyne(i, j, alpha, T, dims, lim, mixedQ=True):
#    """ Integrate matrix element i, j of coh_vac_heterodyne_conditionalstate() over all phase space.
#	Parameters
#	----------
#	i, j: :obj:`int`
#		0 <= i, j < dims
#		Specifies which matrix element I am interested in.
#	alpha: :obj:`complex`
#		Coherent state amplitude.
#	T: :obj:`float`
#		Channel transmittivity.
#	dims: :obj:`int`
#		Underlying hilbert space dimension.
#	lim: :obj:`float`
#		0 < lim
#		x and y limit of my integration.
#    mixedQ: :obj:`bool`
#        True: mix coherent states over QPSK alphabet.
#        False: don't.
#	Returns
#	-------
#	:obj:`complex`
#		Complex value of matrix element i, j after integration, ready to be made into a quantum state.
#	"""
#    _intreal = integrate.dblquad(lambda x, y: coh_vac_heterodyne_outcome(x + 1.0J * y, alpha, T, dims, mixedQ) * np.real(coh_vac_heterodyne_conditionalstate(x + 1.0J*y, alpha, T, dims, qobjQ=False, mixedQ=mixedQ)[i,j]), -lim, lim, lambda y: -lim, lambda y: lim)[0]
#    _intimag = integrate.dblquad(lambda x, y: coh_vac_heterodyne_outcome(x + 1.0J * y, alpha, T, dims, mixedQ) * np.imag(coh_vac_heterodyne_conditionalstate(x + 1.0J*y, alpha, T, dims, qobjQ=False, mixedQ=mixedQ)[i,j]), -lim, lim, lambda y: -lim, lambda y: lim)[0]
#    return _intreal + 1.0J * _intimag
#
    


def intfun_coh_vac_heterodyne_onequad_mpmath(i, j, alpha, T, dims, mixedQ=True):
        return lambda r, th: r * coh_vac_heterodyne_outcome(float(r * mp.cos(th)) + 1.0J*float(r * mp.sin(th)), alpha, T, dims, mixedQ=mixedQ) * coh_vac_heterodyne_conditionalstate(float(r * mp.cos(th)) + 1.0J*float(r * mp.sin(th)), alpha, T, dims, qobjQ=False, mixedQ=mixedQ)[i, j]




    



def integrate_coh_vac_heterodyne_onequad(i, j, alpha, T, dims, lim, mixedQ=True):
    """ Integrate matrix element i, j of coh_vac_heterodyne_conditionalstate() over one quadrature of phase space.
	Parameters
	----------
	i, j: :obj:`int`
		0 <= i, j < dims
		Specifies which matrix element I am interested in.
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		Channel transmittivity.
	dims: :obj:`int`
		Underlying hilbert space dimension.
	lim: :obj:`float`
		0 < lim
		x and y limit of my integration.
    mixedQ: :obj:`bool`
        True: mix coherent states over QPSK alphabet.
        False: don't.
	Returns
	-------
	:obj:`complex`
		Complex value of matrix element i, j after integration, ready to be made into a quantum state.
	"""
    
    if mixedQ==True:
        _probfactor = (1/4)
    elif mixedQ==False:
        _probfactor = probfactor(alpha, T, 0.0)
    
    
    _intreal =  (1/_probfactor) * integrate.dblquad(lambda x, y: coh_vac_heterodyne_outcome(x + 1.0J * y, alpha, T, dims, mixedQ) * np.real(coh_vac_heterodyne_conditionalstate(x + 1.0J*y, alpha, T, dims, qobjQ=False, mixedQ=mixedQ)[i,j]), 0, lim, lambda y: 0, lambda y: lim)[0]
    
    _intimag =  (1/_probfactor) * integrate.dblquad(lambda x, y: coh_vac_heterodyne_outcome(x + 1.0J * y, alpha, T, dims, mixedQ) * np.imag(coh_vac_heterodyne_conditionalstate(x + 1.0J*y, alpha, T, dims, qobjQ=False, mixedQ=mixedQ)[i,j]), 0, lim, lambda y: 0, lambda y: lim)[0]
    
    return _intreal + 1.0J * _intimag



# =============================================================================
# Misc functions
# =============================================================================


# I've included this function here because it will be shared between both QDS1 and QDS2.
def perr(alpha, T, barn):
    """ Honest mismatch probability. Can be calcualted by integrating pb_ak() over half of phase-space.
    NOTE: trying to do the integration in scipy gives consistently incorrect answers. I have done the integrations in Mathematica and compared to the following formula, and they are in complete agreement. See p60 of my yellow book for the formulae, including how they should be displayed in the paper (for consistency with the extra np.sqrt(2) which needs to multiply alpha).
    Parameters
    ----------
    alpha: :obj:`complex`
        Coherent state amplitude.
    T: :obj:`float`
        Channel transmittivity.
    barn: :obj:`float`
        Thermal photon number of the channel.
    Returns
    -------
    :obj:`float`
        perr, honest mismatch rate.
    """
    return (1/2) * erfc( np.sqrt(T / 2) * np.sqrt(2) * alpha / np.sqrt(1 + (1-T) * barn))

def perr_mpmath2(alpha, T, barn):
    """ Honest mismatch probability (no postselection). Calculated by integrating pb_ak() over half of phase space.
    Parameters
    ----------
    alpha: :obj:`float`
        Magnitude of amplitude of QPSK alphabet.
    T: :obj:`float`
        Channel transmittivity
    barn: :obj:`float`
        Thermal photon number of the channel.
    Rturns
    :obj:`float`
        perr, honest mismatch rate, no postselection.
    """
    f = lambda x, y: pb_ak_mpmath(x + 1.0J*y, alpha, T, barn)
    return float(mp.quad(f, [-mp.inf, 0], [-mp.inf, mp.inf]))


def perr_mpmath(alpha, T, barn):
    """ Same as perr_mpmath2 but splits pbak into real and imaginary parts, to see if it gives a performance increase.

    Honest mismatch probability (no postselection). Calculated by integrating pb_ak() over half of phase space.
    Parameters
    ----------
    alpha: :obj:`float`
        Magnitude of amplitude of QPSK alphabet.
    T: :obj:`float`
        Channel transmittivity
    barn: :obj:`float`
        Thermal photon number of the channel.
    Rturns
    :obj:`float`
        perr, honest mismatch rate, no postselection.
    """
    f = lambda x, y: pb_ak_mpmath(x, y, alpha, T, barn)
    return float(mp.quad(f, [-mp.inf, 0], [-mp.inf, mp.inf]))



def pnorm(alpha, T, barn, delta_r, delta_th):
    """ Probability that a heterodyne measurement is accepted (i.e. it does _not_ lie within \\mathcal{R} postselection region). Function used primarily for normalization of perr and 1-perr later on.
    Parameters
    ----------
    alpha: :obj:`complex`
        Complex amplitude of the sent coherent state.
    T: :obj:`float`
        Channel transmittivity.
    delta_r: :obj:`float`
        r cooridinate for defining postselection region
        0 <= delta_r
    delta_th: :obj:`float`
        angular coordinate for defining postselection region. 
        0 <= delta_th <= pi/4.
    Returns
    -------
    :obj:`mpmath.mpf`
        Probability that a measurement outcome is accepted. 
        Object type is mpf() format.
    """
    f = pbak_mpmath_polars_intfun(alpha, T, barn) # the function which I will integrate over.
    _int1 = mp.quad(f, [delta_r, mp.inf], [delta_th, (mp.pi/2)-delta_th])
    _int2 = mp.quad(f, [delta_r, mp.inf], [(mp.pi/2) + delta_th, mp.pi-delta_th])
    _int3 = mp.quad(f, [delta_r, mp.inf], [mp.pi + delta_th, (3 * mp.pi/2) - delta_th])
    _int4 = mp.quad(f, [delta_r, mp.inf], [(3 * mp.pi/2) + delta_th, 2*mp.pi-delta_th])
    _inttot = _int1 + _int2 + _int3 + _int4
    return float(_inttot)

def perr_PS(alpha, T, barn, delta_r, delta_th):
    """ Honest mismatch probability, conditioned on the fact that the measurement was accepted. If delta_r == delta_th == 0 (i.e. no postselection region) then just return analytically known solution.
    Paremeters
    ----------
    alpha: :obj:`float`
        Magnitude of QPSK alphabet.
    T: :obj:`float`
        Channel transmittivity.
    barn: :obj:`float`
        Channel thermal photon number.
    delta_r: :obj:`float`
        r cooridinate for defining postselection region
        0 <= delta_r
    delta_th: :obj:`float`
        angular coordinate for defining postselection region. 
        0 <= delta_th <= pi/4.
    Returns
    -------
    :obj:`float`
        Normalized probability of mismatch.
    """
    
    if delta_r == 0 and delta_th == 0:
        return perr(alpha, T, barn)
    else:
        f = pbak_mpmath_polars_intfun(alpha, T, barn) # the function which I will integrate over
        _norm = pnorm(alpha, T, barn, delta_r, delta_th)
        _int1 = mp.quad(f, [delta_r, mp.inf], [(mp.pi/2) + delta_th, mp.pi - delta_th])
        _int2 = mp.quad(f, [delta_r, mp.inf], [mp.pi + delta_th, (3 * mp.pi/2) - delta_th])
    
        return float((_int1 + _int2) / _norm)


def notperr_PS(alpha, T, barn, delta_r, delta_th):
    """ Probability there is no mismatch, conditioned on the fact that the measurement was accepted. Function primarily used for testing.
    Paremeters
    ----------
    alpha: :obj:`float`
        Magnitude of QPSK alphabet.
    T: :obj:`float`
        Channel transmittivity.
    barn: :obj:`float`
        Channel thermal photon number.
    delta_r: :obj:`float`
        r cooridinate for defining postselection region
        0 <= delta_r
    delta_th: :obj:`float`
        angular coordinate for defining postselection region. 
        0 <= delta_th <= pi/4.
    Returns
    -------
    :obj:`float`
        Normalized probability of mismatch.
    """
    f = pbak_mpmath_polars_intfun(alpha, T, barn) # the function which I will integrate over
    _norm = pnorm(alpha, T, barn, delta_r, delta_th)
    _int1 = mp.quad(f, [delta_r, mp.inf], [(3 * mp.pi/2) + delta_th, (2 * mp.pi) - delta_th])
    _int2 = mp.quad(f, [delta_r, mp.inf], [delta_th, (mp.pi/2) - delta_th])

    return float((_int1 + _int2) / _norm)


def req_barn2(T, xi):
    """ Calculates the channel thermal photon number required to give an excess noise xi in Bob's measurement outcomes. Uses Ref.[59] of Papanastasiou2018.
    NOTE: the xi at Bob is now correct! The function req_barn() (from QKD_reproducing_papa.py) forgets that the vacuum variance is normalized to (1/2) instead of 1.
    Parameters
    ----------
    T : :obj:`float`
        Channel transmittivity.
    xi : :obj:`float`
        Excess noise at Bob, in percent.
    Returns
    -------
    :obj:`float`
        Thermal photon number which will give xi at Bob.    
    """
    
#    return T * xi / (1-T)
    return xi / (1-T)



def db2T(db):
    """ Converts loss in decibels to a transmittivity T.
    Parameters
    ----------
    db : :obj:`float`
        Absolute-value of loss in dB.
    Returns
    -------
    :obj:`float`
        Channel transmittivity T.
    """
    
    return 10**(-db/10)


def T2db(T): 
    """ Converts transmittivity T to loss in decibels
    Parameters
    ----------
    T: :obj:`float`
        Channel transmittivity T.
    Returns
    -------
    :obj:`float`
        Absolute-value of loss in dB.
    """
    
    return - 10 * np.log10(T)


#@memory.cache
def integrate_coh_vac_heterodyne_onequad_mpmath(i, j, alpha, T, dims, lim, delta_r = 0.0, delta_th = 0.0, mixedQ=True):
    """ Integrate matrix element i, j of coh_vac_heterodyne_conditionalstate() over one quadrature of phase space.
	Parameters
	----------
	i, j: :obj:`int`
		0 <= i, j < dims
		Specifies which matrix element I am interested in.
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		Channel transmittivity.
	dims: :obj:`int`
		Underlying hilbert space dimension.
	lim: :obj:`float`
		0 < lim
		x and y limit of my integration.
    mixedQ: :obj:`bool`
        True: mix coherent states over QPSK alphabet.
        False: don't.
	Returns
	-------
	:obj:`complex`
		Complex value of matrix element i, j after integration, ready to be made into a quantum state.
	"""
    if mixedQ==True:
        _probfactor = (1/4)
    elif mixedQ==False:
        _probfactor = probfactor(alpha, T, 0.0)

    
    f = intfun_coh_vac_heterodyne_onequad_mpmath(i, j, alpha, T, dims, mixedQ=mixedQ) # the function to integrate 
    
    _int = mp.quad(f, [delta_r, lim], [delta_th, (mp.pi/2)-delta_th], method="gauss-legendre") # do the integration 
    
    _pnorm = pnorm(alpha, T, 0.0, delta_r, delta_th)
    
    return complex(_int) * (1/_probfactor) * (1/_pnorm)# normalize the integration with respect to the section of phase space I want to integrate over.


def integrate_coh_vac_heterodyne_mpmath(i, j, alpha, T, dims, lim, delta_r = 0.0, delta_th = 0.0, mixedQ=True):
    """ Integrate matrix element i, j of coh_vac_heterodyne_conditionalstate() over all of phase space
	Parameters
	----------
	i, j: :obj:`int`
		0 <= i, j < dims
		Specifies which matrix element I am interested in.
	alpha: :obj:`complex`
		Coherent state amplitude.
	T: :obj:`float`
		Channel transmittivity.
	dims: :obj:`int`
		Underlying hilbert space dimension.
	lim: :obj:`float`
		0 < lim
		x and y limit of my integration.
    mixedQ: :obj:`bool`
        True: mix coherent states over QPSK alphabet.
        False: don't.
	Returns
	-------
	:obj:`complex`
		Complex value of matrix element i, j after integration, ready to be made into a quantum state.
	"""

    
    f = intfun_coh_vac_heterodyne_onequad_mpmath(i, j, alpha, T, dims, mixedQ=mixedQ) # the function to integrate 
    
    _int1 = mp.quad(f, [delta_r, lim], [delta_th, (mp.pi/2)-delta_th], method="gauss-legendre") 
    _int2 = mp.quad(f, [delta_r, lim], [(mp.pi/2)+delta_th, (mp.pi)-delta_th], method="gauss-legendre") 
    _int3 = mp.quad(f, [delta_r, lim], [mp.pi+delta_th, (3*mp.pi/2)-delta_th], method="gauss-legendre") 
    _int4 = mp.quad(f, [delta_r, lim], [(3*mp.pi/2)+delta_th, (2*mp.pi)-delta_th], method="gauss-legendre") 
    
    _int = _int1 + _int2 + _int3 + _int4
    
    _pnorm = pnorm(alpha, T, 0.0, delta_r, delta_th)
    
    return complex(_int)  * (1/_pnorm)# normalize the integration with respect to the section of phase space I want to integrate over.




def coh_tmsv_het_conditionalstate_byhand(beta, alpha, T, barn, dims):
    _zeta = barn2zeta(barn)
    
    _prefactor = (1/np.pi) * np.exp(-np.abs(beta)**2) * np.exp(-np.abs(alpha)**2) * (1/(np.cosh(_zeta)**2))
    
    _mylist = []
    
    def mydenominator(n1, n2, m1, m2, k1, k2, l1, l2):
        _denom = math.factorial(k1) * math.factorial(k2) * math.factorial(l1) * math.factorial(l2) * math.factorial(n1-k1) * math.factorial(n2 - k2) * math.factorial(m1 - l1) * math.factorial(m2 - l2)
        return 1/_denom
    
    def myfun(n1, n2, m1, m2, k1, k2, l1, l2, beta, alpha, T, zeta):
        return alpha**n1 * np.conj(alpha)**m1 * (np.tanh(zeta)**(n2 + m2)) * beta**(k1 + k2) * np.conj(beta)**(l1 + l2) * np.sqrt(math.factorial(n2) * math.factorial(m2)) * mydenominator(n1, n2, m1, m2, k1, k2, l1, l2) * np.sqrt(math.factorial(n1 + n2 - k1 - k2) * math.factorial(m1 + m2 - l1 - l2)) * np.sqrt(T)**(k1 + l1) * np.sqrt(1-T)**(n1 + m1 - k1 - l1) * (-np.sqrt(1-T))**(k2 + l2) * np.sqrt(T)**(n2+m2-k2-l2)
    
    def mystate(n1, n2, m1, m2, k1, k2, l1, l2):
        _mode1 = qt.basis(2*dims, n1+n2-k1-k2) * qt.basis(2*dims, m1+m2-l1-l2).dag()
        _mode2 = qt.basis(2*dims, n2) * qt.basis(2*dims, m2).dag()
        return qt.tensor(_mode1, _mode2)
    
    for n1 in np.arange(dims):
        for n2 in np.arange(dims):
            for m1 in np.arange(dims):
                for m2 in np.arange(dims):
                    for k1 in np.arange(n1):
                        for k2 in np.arange(n2):
                            for l1 in np.arange(m1): 
                                for l2 in np.arange(m2):
                                    _mylist.append(myfun(n1, n2, m1, m2, k1, k2, l1, l2, beta, alpha, T, _zeta) * mystate(n1, n2, m1, m2, k1, k2, l1, l2))
    return _prefactor * np.array(_mylist, dtype="object").sum()
    
    
    
    
    
    
    
    
    