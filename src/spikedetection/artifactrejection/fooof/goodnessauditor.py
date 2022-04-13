""" goodnessauditor.py

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
import abc
from collections.abc import MutableMapping, Hashable, Iterator, Iterable
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject, singlekwargdispatchmethod
from baseobjects.types_ import AnyCallable
import numpy as np

# Local Packages #
from .dataclasses import FitSpectrumCurve, FitSpectrumCurves


# Definitions #
# Classes #
class GoodnessAuditInterface(BaseObject):
    """An abstract class and interface for goodness audit objects."""
    # Magic Methods #
    # Callable
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """When this auditor is called, run the audit.

        Args:
            *args: The arguments to pass to the audit.
            **kwargs: The keyword arguments to pass to the audit.

        Returns:
            The results of the audit.
        """
        return self.run_audit(*args, **kwargs)

    # Instance Methods #
    # Auditing
    @abc.abstractmethod
    def run_audit(self, *args, **kwargs) -> Any:
        """Runs a goodness audit on the provided input.

        Args:
            *args: The arguments to run the audit.
            **kwargs: The keyword arguments to run the audit

        Returns:
            The results of the audit.
        """
        pass


class RSquaredBoundsAudit(GoodnessAuditInterface):
    """A goodness audit which uses R squared to determine goodness, particularly within a bounded R squared range.

    The goodness values are R squared values rescaled with the minimum and maximum values being 0 and 1 respectively.
    Any R squared values outside the bounds are clipped to the minimum and maximum values.

    Class Attributes:
        default_minimum: The default lowest bound of the R squared to be considered 0 goodness.
        default_maximum: The default highest bound of the R squared to be considered 1 goodness.

    Attributes:
        minimum: The lowest bound of the R squared to be considered 0 goodness.
        maximum: The highest bound of the R squared to be considered 1 goodness.

    Args:
        min_: The minimum bound of the R squared range for goodness.
        max_: The maximum bound of the R squared range for goodness.
        init: Determines if this object will be constructed.
    """
    default_minimum: float = 0.30
    default_maximum: float = 0.65

    # Magic Methods #
    # Construction/Destruction
    def __init__(self, min_: float | None = None, max_: float | None = None, init: bool = True) -> None:
        # New Attributes #
        self.minimum: float = self.default_minimum
        self.maximum: float = self.default_maximum

        # Object Construction #
        if init:
            self.construct(min_=min_, max_=max_)

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, min_: float | None, max_: float | None = None,) -> None:
        """Constructs this object.

        Args:
            min_: The minimum bound of the r_squared range for goodness.
            max_: The maximum bound of the r_squared range for goodness.
        """
        if min_ is not None:
            self.minimum = min_

        if max_ is not None:
            self.maximum = max_

    # Auditing
    @singlekwargdispatchmethod("info")
    def run_audit(self, info: float | np.ndarray | FitSpectrumCurve | FitSpectrumCurves) -> float | np.ndarray:
        """Runs the audit to determine the goodness of the fit.

        Args:
            info: Either a FitSepctrumCurve data object or the R squared values of the curves.

        Returns:
            The goodness values of the curves.
        """
        raise ValueError(f"{self.__class__} cannot run an audit with a {type(info)}.")

    @run_audit.register
    def __run_audit(self, info: float) -> float:
        """Runs the audit to determine the goodness of the fit.

        Args:
            info: The R squared value to determine its goodness.

        Returns:
            The goodness value of the curve.
        """
        return self.single_goodness_audit(r_squared=info)

    @run_audit.register
    def __run_audit(self, info: np.ndarray) -> np.ndarray:
        """Runs the audit to determine the goodness of the fits.

        Args:
            info: The R squared values to determine their goodness.

        Returns:
            The goodness values of the curves.
        """
        return self.multiple_goodness_audit(r_squared=info)

    @run_audit.register
    def __run_audit(self, info: FitSpectrumCurve) -> float:
        """Runs the audit to determine the goodness of the fit.

        Args:
            info: The curve and its metrics that will determine its goodness.

        Returns:
            The goodness values of the curve.
        """
        return self.single_goodness_audit(r_squared=info.r_squared)

    @run_audit.register
    def __run_audit(self, info: FitSpectrumCurves) -> np.ndarray:
        """Runs the audit to determine the goodness of the fits.

        Args:
            info: The curves and their metrics that will determine their goodness.

        Returns:
            The goodness values of the curves.
        """
        return self.multiple_goodness_audit(r_squared=info.r_squared)

    def single_goodness_audit(self, r_squared: float) -> float:
        """Evaluates the goodness for a single fit based on the r_squared value.

        Args:
            r_squared: A single R squared value to evaluate the goodness.

        Returns:
            The goodness value as a scalar between 0 and 1.
        """
        if r_squared <= self.minimum:
            return 0.0
        elif r_squared >= self.maximum:
            return 1.0
        else:
            return (r_squared - self.minimum) / (self.maximum - self.minimum)

    def multiple_goodness_audit(self, r_squared: np.ndarray) -> np.ndarray:
        """Evaluates the goodness for multiple fits based on the r_squared values.

        Args:
            r_squared: R squared values to evaluate the goodness of each fit.

        Returns:
            The goodness values as an array of scalars between 0 and 1.
        """
        return (np.clip(r_squared, self.minimum, self.maximum) - self.minimum) / (self.maximum - self.minimum)


class GoodnessAuditor(BaseObject, MutableMapping):
    """An object which evaluates the goodness of a fit curve based on a chosen audit.

    The Auditor contains several audits within it, so the method of determining goodness can be changed efficiently.

    Class Attributes:
        default_audits: The default audits to add when an object is created.

    Attributes:
        audits: The audits to store in this auditor.
        _audit: The audit to run when run_audit is called.

    Args:
        dict_: A dictionary of audits to add to this auditor.
        init: Determines if this object will be constructed.
        **kwargs: Audits to add as keyword arguments.
    """
    default_audits: dict[Hashable, GoodnessAuditInterface] = {}

    # Class Methods #
    @classmethod
    def fromkeys(cls, iterable: Iterable[Hashable], value: AnyCallable | None = None) -> "GoodnessAuditor":
        """Creates a GoodnessAuditor object with specified keys set to a default audit or None.

        Args:
            iterable: The keys put in audits.
            value: The default audit or None to fill the keys with.

        Returns:
            A GoodnessAuditor object with the keys.
        """
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        dict_: dict[Hashable, GoodnessAuditInterface] | None = None,
        /,
        init: bool = True,
        **kwargs: GoodnessAuditInterface,
    ) -> None:
        # New Attributes #
        self.audits: dict[Hashable, GoodnessAuditInterface] = self.default_audits.copy()
        self._audit: GoodnessAuditInterface | None = None

        # Object Construction #
        if init:
            self.construct(dict_, **kwargs)

    # Container Methods
    def __len__(self) -> int:
        """Returns the mount of audits in this auditor."""
        return len(self.audits)

    def __getitem__(self, key: Hashable) -> Any:
        """Gets an audit within this auditor."""
        if key in self.audits:
            return self.audits[key]
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __setitem__(self, key: Hashable, item: Any) -> None:
        """Sets an audit within this auditor."""
        self.audits[key] = item

    def __delitem__(self, key: Hashable) -> Any:
        """Deletes and audit within this auditor."""
        del self.audits[key]

    def __iter__(self) -> Iterator:
        """Returns an iterator of the audits within this auditor."""
        return iter(self.audits)

    def __contains__(self, key: Hashable) -> bool:
        """Checks if an audit is within this auditor."""
        return key in self.audits

    # Callable
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """When this auditor is called, run the audit.

        Args:
            *args: The arguments to pass to the audit.
            **kwargs: The keyword arguments to pass to the audit.

        Returns:
            The results of the audit.
        """
        return self._audit(*args, **kwargs)

    # Representation
    def __repr__(self) -> str:
        """The representation of this object as a string."""
        return repr(self.audits)

    # Logic Operators
    def __or__(self, other: Any) -> "GoodnessAuditor":
        """The or operator as a union operation."""
        new = self.copy()
        if isinstance(other, GoodnessAuditor):
            return new.audits.update(other.audits)
        if isinstance(other, dict):
            return new.audits.update(other)
        return NotImplemented

    def __ror__(self, other: Any) -> "GoodnessAuditor":
        """The reverse or operator as a union operation."""
        new = self.copy()
        if isinstance(other, GoodnessAuditor):
            return new.audits.update(other.audits)
        if isinstance(other, dict):
            return new.audits.update(other)
        return NotImplemented

    def __ior__(self, other: Any) -> "GoodnessAuditor":
        """The assign or operator as a union operation"""
        if isinstance(other, GoodnessAuditor):
            self.audits |= other.audits
        else:
            self.audits |= other
        return self

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        dict_: dict[Hashable, GoodnessAuditInterface] | None = None,
        /,
        **kwargs: GoodnessAuditInterface,
    ) -> None:
        """Constructs this object.

        Args:
            dict_: A dictionary of audits to add to this auditor.
            **kwargs: Audits to add as keyword arguments.
        """
        if dict_ is not None:
            self.update(dict)

        if kwargs:
            self.update(kwargs)

    # Auditing
    def run_audit(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the audit.

        Args:
            *args: The arguments to pass to the audit.
            **kwargs: The keyword arguments to pass to the audit.

        Returns:
            The results of the audit.
        """
        return self._audit(*args, **kwargs)

    def set_audit(self, name: Hashable, audit: GoodnessAuditInterface | None = None) -> None:
        """Sets the audit to be run. It can audit in this auditor or another to be added.

        Args:
            name: The name of the audit to set as the run audit.
            audit: The audit to add if given.
        """
        if audit is None:
            self._audit = self.audits[name]
        else:
            self.audits[name] = audit
            self._audit = audit
