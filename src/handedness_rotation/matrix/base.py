from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from cartesian_axis import Axis, CoordinateHandedness
from rotation import RotationMatrix as RotationMatrixBase

from .mixin import (
    RotationMatrixConvertMixin,
    RotationMatrixFactoryMixin,
    RotationMatrixSpecialMixin,
)


@dataclass(frozen=True)
class RotationMatrix(
    RotationMatrixConvertMixin,
    RotationMatrixSpecialMixin["RotationMatrix"],
    RotationMatrixFactoryMixin,
    RotationMatrixBase,
):
    """
    A 3x3 proper rotation matrix (SO(3)) tagged with a coordinate handedness.

    The tag only records which frame convention this matrix is meant for; it does
    not perform any axis flip when exchanging data with another pipeline.
    Composition via @ requires both operands to share the same tag.

    Attributes
    ----------
    value: np.ndarray
        The rotation matrix as a 3x3 matrix.
    coordinate_handedness: CoordinateHandedness
        The coordinate handedness (CoordinateHandedness.RIGHT or CoordinateHandedness.LEFT).

    Raises
    ------
    ValueError:
        If value is not a 3x3 orthogonal matrix with determinant +1.
    """

    coordinate_handedness: CoordinateHandedness

    @property
    def is_determinant_correct(self) -> bool:
        """
        True if this matrix is a proper rotation (orthogonal with determinant +1).

        Returns
        -------
        bool:
            True if determinant is +1 within tolerance, False otherwise.
        """
        return bool(np.isclose(np.linalg.det(self.value), 1.0, atol=1e-6))

    def extract_axis_rotation(
        self,
        axis: Axis,
    ) -> np.ndarray:
        """
        Extract the column of the rotation matrix for the given Cartesian axis.

        Parameters
        ----------
        axis: Axis
            The axis to extract (X, Y, or Z).

        Returns
        -------
        np.ndarray:
            The corresponding column of the rotation matrix, shape (3,).
        """
        match axis:
            case Axis.X:
                return self.value[:, 0]
            case Axis.Y:
                return self.value[:, 1]
            case Axis.Z:
                return self.value[:, 2]
