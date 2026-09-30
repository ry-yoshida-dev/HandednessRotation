from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore
from cartesian_axis import Axis, CoordinateHandedness
from units import Angle, AngleUnit

from ...order import IntrinsicRotationOrder, ExtrinsicRotationOrder
from ..protocol import HandedRotationMatrixProtocol, handed_rotation_matrix_ctor

if TYPE_CHECKING:
    from ...euler import EulerAngles

_R = TypeVar("_R", bound=HandedRotationMatrixProtocol)


class RotationMatrixConvertMixin:
    """Conversions to other representations (handedness tag, Euler angles)."""

    @staticmethod
    def _axis_reflection_matrix(axis: Axis) -> np.ndarray:
        match axis:
            case Axis.X:
                return np.diag([-1.0, 1.0, 1.0])
            case Axis.Y:
                return np.diag([1.0, -1.0, 1.0])
            case Axis.Z:
                return np.diag([1.0, 1.0, -1.0])

    def to_opposite_handedness(self: _R, *, flip_axis: Axis = Axis.Z) -> _R:
        """
        Represent the same proper rotation under the opposite handedness tag using one axis flip.

        Computes R' = D @ R @ D where D reflects flip_axis (orthogonal, det(D) = -1, D @ D = I).
        Then det(R') = +1 still holds. Which axis to flip is a convention (often Z in graphics);
        match it to the change-of-basis between your left-handed and right-handed setups.

        Parameters
        ----------
        flip_axis : Axis
            Coordinate axis reflected by D.

        Returns
        -------
        _R
            Same orthogonal SO(3) map with flipped coordinate_handedness metadata.
        """
        reflection: np.ndarray = RotationMatrixConvertMixin._axis_reflection_matrix(flip_axis)
        new_handedness: CoordinateHandedness
        match self.coordinate_handedness:
            case CoordinateHandedness.RIGHT:
                new_handedness = CoordinateHandedness.LEFT
            case CoordinateHandedness.LEFT:
                new_handedness = CoordinateHandedness.RIGHT
        return handed_rotation_matrix_ctor(type(self))(
            value=reflection @ self.value @ reflection,
            coordinate_handedness=new_handedness,
        )

    def to_euler_angles(
        self: _R,
        order: IntrinsicRotationOrder | ExtrinsicRotationOrder,
        unit: AngleUnit,
    ) -> EulerAngles:
        """
        Euler angles via SciPy (internally right-handed).

        angle progression follows SciPy for this matrix as SO(3); it does not reinterpret signs for
        a left-handed coordinate_handedness tag on its own.
        """
        from ...euler import EulerAngles

        r = Rotation.from_matrix(self.value)
        euler_angles: Angle = Angle(
            value=r.as_euler(order.value, degrees=False),
            unit=AngleUnit.RADIAN,
        )
        euler_angles.convert_unit(unit)
        return EulerAngles(
            value=np.asarray(euler_angles.value, dtype=np.float64),
            order=order,
            unit=unit,
        )
