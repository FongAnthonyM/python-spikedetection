""" svmaudit.py

"""
# Package Header #
from src.spikedetection.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
import pathlib
import pickle

# Third-Party Packages #
from baseobjects import singlekwargdispatchmethod
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import scale

# Local Packages #
from ...dataclasses import FitSpectrumCurve, FitSpectrumCurves
from .goodnessauditinterface import GoodnessAuditInterface


# Definitions #
# Classes #
class SVMAudit(GoodnessAuditInterface):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self, svm: SVC | None = None,
        path: str | pathlib.Path | None = None,
        probability: bool = False,
        init: bool = True,
    ) -> None:
        # New Attributes #
        self._path: pathlib.Path | None = None
        self._probability: bool = False

        self._svm: SVC | None = None

        # Object Construction #
        if init:
            self.construct(svm=svm, path=path, probability=probability)

    @property
    def path(self) -> pathlib.Path | None:
        return self._path

    @path.setter
    def path(self, value: str | pathlib.Path) -> None:
        if isinstance(value, str):
            self._path = pathlib.Path(value)
        else:
            self._path = value
            
    @property
    def probability(self):
        return self._path
    
    @probability.setter
    def probability(self, value: bool) -> None:
        if self.svm is not None and not self.svm.probability and value:
            raise ValueError("The provided SVM cannot calculate probabilities.")
        else:
            self._probability = value

    @property
    def svm(self):
        return self._svm

    @svm.setter
    def svm(self, value: SVC) -> None:
        if not value.probability and self._probability:
            raise ValueError("The provided _svm cannot calculate probabilities.")
        else:
            self._svm = value

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        svm: SVC | None = None,
        path: str | pathlib.Path | None = None,
        probability: bool | None = None,
    ) -> None:
        if svm is not None:
            self.svm = svm
            
        if path is not None:
            self.path = path

        if probability is not None:
            self.probability = probability
            
        if self.svm is None and self._path is not None:
            self.load_svm()

    def load_svm(self, path: str | pathlib.Path | None = None) -> None:
        if path is not None:
            self.path = path
        
        with self.path.open("rb") as file_object:
            self.svm = pickle.load(file_object)
    
    # Auditing
    @singlekwargdispatchmethod("info")
    def run_audit(self, info: np.ndarray | FitSpectrumCurve | FitSpectrumCurves) -> np.ndarray:
        """Runs the audit to determine the goodness of the fit.

        Args:
            info: Either a FitSepctrumCurve data object or the R squared values of the curves.

        Returns:
            The goodness values of the curves.
        """
        raise ValueError(f"{self.__class__} cannot run an audit with a {type(info)}.")

    @run_audit.register
    def __run_audit(self, info: np.ndarray) -> np.ndarray:
        """Runs the audit to determine the goodness of the fits.

        Args:
            info: The R squared values to determine their goodness.

        Returns:
            The goodness values of the curves.
        """
        return self.goodness_audit(metrics_=info)

    @run_audit.register
    def __run_audit(self, info: FitSpectrumCurve) -> np.ndarray:
        """Runs the audit to determine the goodness of the fit.

        Args:
            info: The curve and its metrics that will determine its goodness.

        Returns:
            The goodness values of the curve.
        """
        metrics_ = np.zeros((5,))
        metrics_[0] = info.r_squared
        metrics_[1] = info.normal_entropy
        metrics_[2] = info.mae
        metrics_[3] = info.mse
        metrics_[4] = info.rmse

        metrics_ = scale(metrics_)

        return self.goodness_audit(metrics_=metrics_)

    @run_audit.register
    def __run_audit(self, info: FitSpectrumCurves) -> np.ndarray:
        """Runs the audit to determine the goodness of the fits.

        Args:
            info: The curves and their metrics that will determine their goodness.

        Returns:
            The goodness values of the curves.
        """
        channels = len(info.r_squared)

        metrics_ = np.zeros((channels, 5))
        metrics_[:, 0] = info.r_squared
        metrics_[:, 1] = info.normal_entropy
        metrics_[:, 2] = info.mae
        metrics_[:, 3] = info.mse
        metrics_[:, 4] = info.rmse

        metrics_ = scale(metrics_)

        return self.goodness_audit(metrics_=metrics_)

    def goodness_audit(self, metrics_: np.ndarray) -> np.ndarray:
        if self._probability:
            return self._svm.predict_proba(metrics_)
        else:
            return self._svm.predict(metrics_)
