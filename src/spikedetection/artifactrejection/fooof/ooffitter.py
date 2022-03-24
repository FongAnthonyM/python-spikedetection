""" ooffitter.py

"""
# Package Header #
from ...header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
from typing import Any, Callable, NamedTuple
import warnings

# Third-Party Packages #
from baseobjects import BaseObject, singlekwargdispatchmethod, search_sentinel
from fooof import FOOOF
import numpy as np
from scipy.optimize import curve_fit

# Local Packages #


# Definitions #
class FitError(Exception):
    """Error for a failure to fit."""


class MeanErrors(NamedTuple):
    """A data class for containing mean errors."""
    mae: np.ndarray
    mse: np.ndarray
    rmse: np.ndarray


def calculate_mean_errors(a: np.ndarray, b: np.ndarray) -> MeanErrors:
    """Calculates the mean errors between two arrays.

    Args:
        a: The primary array.
        b: The secondary array.

    Returns:.
        The mean errors between the two arrays
    """
    difference = a - b

    mae = np.abs(difference).mean()
    mse = (difference ** 2).mean()
    rmse = np.sqrt(mse)

    return MeanErrors(mae, mse, rmse)


def expo_function(xs, *args):
    """Exponential fitting function, for fitting aperiodic component with a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *args : float
        Parameters (offset, knee, exp) that define Lorentzian function:
        y = 10^offset * (1/(knee + x^exp))

    Returns
    -------
    ys : 1d array
        Output values for exponential function.
    """

    ys = np.zeros_like(xs)

    offset, knee, exp = args

    ys = ys + offset - np.log10(knee + xs**exp)

    return ys


def expo_nk_function(xs, *args):
    """Exponential fitting function, for fitting aperiodic component without a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *args : float
        Parameters (offset, exp) that define Lorentzian function:
        y = 10^off * (1/(x^exp))

    Returns
    -------
    ys : 1d array
        Output values for exponential function, without a knee.
    """

    ys = np.zeros_like(xs)

    offset, exp = args

    ys = ys + offset - np.log10(xs**exp)

    return ys


# Classes #
class OOFFitter(BaseObject):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    fitting_methods: dict[str, Callable] = {"fixed": expo_nk_function, "knee": expo_function}

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, init: bool = True) -> None:
        # New Attributes #

        # Percentile threshold, to select points from a flat spectrum for an initial aperiodic fit
        #   Points are selected at a low percentile value to restrict to non-peak points
        self._oof_percentile_thresh = 0.025
        # Guess parameters for aperiodic fitting, [offset, knee, exponent]
        #   If offset guess is None, the first value of the power spectrum is used as offset guess
        #   If exponent guess is None, the abs(log-log slope) of first & last points is used
        self._oof_guess = (None, 0, None)
        # Bounds for aperiodic fitting, as: ((offset_low_bound, knee_low_bound, exp_low_bound),
        #                                    (offset_high_bound, knee_high_bound, exp_high_bound))
        # By default, aperiodic fitting is unbound, but can be restricted here, if desired
        #   Even if fitting without knee, leave bounds for knee (they are dropped later)
        self._oof_bounds = ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))
        # The maximum number of calls to the curve fitting function
        self._maxfev = 5000
        
        self.axis: int = 0
        
        self.power_spectra_method: Callable[..., np.ndarray] = self.fft

        self.fitting_mode: str = "fixed"
        self._fitting_method: Callable = None
        
        # Object Construction #
        if init:
            self.construct()

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, init: bool = True) -> None:
        pass

    @singlekwargdispatchmethod("method")
    def set_fitting_method(self, method: Callable | str) -> None:
        """Sets the fitting method of this object.

        Args:
            method: The method or name of method to use for fitting.
        """
        raise TypeError(f"{type(method)} can not be used to set the fitting method of {type(self)}")

    @set_fitting_method.register(Callable)
    def _(self, method: Callable) -> None:
        """Sets the fitting method of this object.

        Args:
            method: The method or name of method to use for fitting.
        """
        self._fitting_method = method

    @set_fitting_method.register
    def _(self, method: str) -> None:
        """Sets the fitting method of this object.

        Args:
            method: The method or name of method to use for fitting.
        """
        method = self.fitting_methods.get(method, search_sentinel)
        if method is search_sentinel:
            raise ValueError("The provide method name is not a fitting method.")
        else:
            self._fitting_method = method

    def fft(self, data: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Uses the fast fourier transform to calculate the power spectra of the given signals in a numpy array.

        Args:
            data: The data to find the power spectra of.
            axis: The axis over which to compute the power spectra.

        Returns:
            The power spectra of the given data.
        """
        axis = axis if axis is not None else self.axis
        f_transform = np.fft.rfft(data, axis=axis)
        return np.square(np.abs(f_transform))

    def generate_oof(self, freqs, *fit_params):
        """Generate aperiodic values.

        Parameters
        ----------
        freqs : 1d array
            Frequency vector to create aperiodic component for.
        fit_params : list of float
            Parameters that define the aperiodic component.


        Returns
        -------
        ap_vals : 1d array
            Aperiodic values, in log10 spacing.
        """
        return self._fitting_method(freqs, *fit_params)

    def _simple_oof_fit(self, freqs, power_spectrum):
        """Fit the aperiodic component of the power spectrum.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.
        """

        # Get the guess parameters and/or calculate from the data, as needed.
        # Note that these are collected as lists, to concatenate with or without knee later
        off_guess = [power_spectrum[0] if not self._oof_guess[0] else self._oof_guess[0]]
        kne_guess = [self._oof_guess[1]] if self.fitting_mode == 'knee' else []
        exp_guess = [np.abs(self.power_spectrum[-1] - self.power_spectrum[0] /
                            np.log10(self.freqs[-1]) - np.log10(self.freqs[0]))
                     if not self._oof_guess[2] else self._oof_guess[2]]
        guess = np.array([off_guess + kne_guess + exp_guess])

        # Ignore warnings that are raised in curve_fit.
        # A runtime warning can occur while exploring parameters in curve fitting.
        # This doesn't affect outcome - it won't settle on an answer that does this.
        # It happens if / when b < 0 & |b| > x**2, as it leads to log of a negative number.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof_params, _ = curve_fit(
                    self._fitting_method,
                    freqs,
                    power_spectrum,
                    p0=guess,
                    maxfev=self._maxfev,
                    bounds=self._oof_bounds,
                )
        except RuntimeError:
            raise FitError("Model fitting failed due to not finding parameters in "
                           "the simple one over f fit.")

        return oof_params
    
    def _robust_oof_fit(self, freqs, power_spectrum):
        """Fit the aperiodic component of the power spectrum robustly, ignoring outliers.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.

        Raises
        ------
        FitError
            If the fitting encounters an error.
        """

        # Do a quick, initial aperiodic fit
        popt = self._simple_oof_fit(freqs, power_spectrum)
        initial_fit = self._fitting_method(freqs, *popt)

        # Flatten power_spectrum based on initial aperiodic fit
        flatspec = power_spectrum - initial_fit
        flatspec[flatspec < 0] = 0  # Flatten outliers that drop below 0

        # Use percentile threshold, in terms of # of points, to extract and re-fit
        perc_thresh = np.percentile(flatspec, self._oof_percentile_thresh)
        perc_mask = flatspec <= perc_thresh
        freqs_ignore = freqs[perc_mask]
        spectrum_ignore = power_spectrum[perc_mask]

        # Second aperiodic fit - using results of first fit as guess parameters
        # Ignore warnings that are raised in curve_fit.
        # A runtime warning can occur while exploring parameters in curve fitting.
        # This doesn't affect outcome - it won't settle on an answer that does this.
        # It happens if / when b < 0 & |b| > x**2, as it leads to log of a negative number.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof_params, _ = curve_fit(
                    self._fitting_method,
                    freqs_ignore,
                    spectrum_ignore,
                    p0=popt,
                    maxfev=self._maxfev,
                    bounds=self._oof_bounds,
                )
        except RuntimeError:
            raise FitError("Model fitting failed due to not finding "
                           "parameters in the robust one over f fit.")
        except TypeError:
            raise FitError("Model fitting failed due to sub-sampling in the robust one over f fit.")

        return oof_params

    def fit(self, freqs=None, power_spectrum=None, freq_range=None):
        """Fit the full power spectrum as a combination of periodic and aperiodic components.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power spectrum, in linear space.
        power_spectrum : 1d array, optional
            Power values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectrum to. If not provided, keeps the entire range.

        Raises
        ------
        NoDataError
            If no data is available to fit.
        FitError
            If model fitting fails to fit. Only raised in debug mode.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """
        oof_params = self._robust_oof_fit(freqs, power_spectrum)
        oof_curve = self._fitting_method(freqs, oof_params)

        r_val = np.corrcoef(power_spectrum, oof_curve)
        r_squared = r_val[0][1] ** 2
        errors = calculate_mean_errors(power_spectrum, oof_curve)

