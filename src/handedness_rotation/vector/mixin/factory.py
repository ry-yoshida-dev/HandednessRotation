from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt
from cartesian_axis import CoordinateHandedness
from rotation import RotationMatrix as RotationMatrixBase
from rotation import RotationVector as RotationVectorBase
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from ..protocol import HandedRotationVectorProtocol, handed_rotation_vector_ctor

_R = TypeVar("_R", bound=HandedRotationVectorProtocol)


class RotationVectorFactoryMixin:
    """
    Factory classmethods for handedness-tagged rotation vectors.

    Keep this mixin **plain** (do not subclass ``rotation`` vector mixins): the
    upstream package already defines ``RotationVectorFactoryMixin`` on its
    container; inheriting that type makes overrides clash with static analysis.
    Combine with ``rotation.RotationVector`` (or a subclass) in the final class
    bases so ``value`` validation, ``angle``, and ``rotation_matrix`` stay
    available. Implementations delegate to ``rotation`` for validation and conversion,
    then attach ``coordinate_handedness`` (metadata only).
    """

    @classmethod
    def from_matrix(
        cls: type[_R],
        rotation_matrix: npt.ArrayLike,
        *,
        coordinate_handedness: CoordinateHandedness = CoordinateHandedness.RIGHT,
        validate: bool = True,
    ) -> _R:
        """
        Build a rotation vector from a 3×3 rotation matrix.

        Delegates axis–angle extraction and SO(3) validation to
        :meth:`RotationVectorBase.from_matrix` (same rules as ``rotation``:
        by default the matrix is checked to be orthogonal with determinant +1,
        matching :class:`rotation.RotationMatrix`). Pass ``validate=False``
        only when the matrix is already known to be a valid rotation.

        Parameters
        ----------
        rotation_matrix : array_like
            Source rotation matrix with shape (3, 3).
        coordinate_handedness : CoordinateHandedness, default ``RIGHT``
            Convention tag stored on the result (does not change the numeric map).
        validate : bool, default True
            If True, run the same validation as ``rotation`` (see upstream docs).

        Returns
        -------
        RotationVector
            Rotation vector tagged with ``coordinate_handedness``.

        Raises
        ------
        ValueError
            If ``rotation_matrix`` is not shape (3, 3), or if ``validate`` is True
            and the matrix is not a proper rotation.
        """
        matrix = np.asarray(rotation_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Invalid rotation matrix: must be a shape (3, 3) array, got shape {matrix.shape}."
            )
        if validate:
            RotationMatrixBase.validate_rotation_matrix_array(matrix)
        return handed_rotation_vector_ctor(cls)(
            value=Rotation.from_matrix(matrix).as_rotvec(),
            coordinate_handedness=coordinate_handedness,
        )

    @classmethod
    def zero_vector(
        cls: type[_R],
        *,
        coordinate_handedness: CoordinateHandedness = CoordinateHandedness.RIGHT,
    ) -> _R:
        """
        Identity rotation (zero rotation vector) with a handedness tag.

        Equivalent to :meth:`RotationVectorBase.zero_vector` for ``value``,
        plus the requested ``coordinate_handedness``.

        Parameters
        ----------
        coordinate_handedness : CoordinateHandedness, default ``RIGHT``
            Convention tag stored on the result.

        Returns
        -------
        RotationVector
            Zero vector ``(0, 0, 0)`` with shape (3,) and the given tag.
        """
        base = RotationVectorBase.zero_vector()
        return handed_rotation_vector_ctor(cls)(
            value=np.asarray(base.value, dtype=np.float64),
            coordinate_handedness=coordinate_handedness,
        )

    @classmethod
    def from_axis_angle(
        cls: type[_R],
        axis: npt.ArrayLike,
        angle: float,
        *,
        coordinate_handedness: CoordinateHandedness = CoordinateHandedness.RIGHT,
    ) -> _R:
        """
        Build a rotation vector from an axis and angle in radians.

        ``axis`` is normalized to unit length; a zero-length axis yields the
        identity rotation (same behavior as :meth:`RotationVectorBase.from_axis_angle`).

        Parameters
        ----------
        axis : array_like
            Rotation axis (any non-zero 3-vector; will be normalized).
        angle : float
            Rotation angle in radians.
        coordinate_handedness : CoordinateHandedness, default ``RIGHT``
            Convention tag stored on the result.

        Returns
        -------
        RotationVector
            ``value = (axis / ||axis||) * angle`` (or zero if axis vanishes),
            tagged with ``coordinate_handedness``.
        """
        if np.isclose(float(np.linalg.norm(np.asarray(axis, dtype=np.float64))), 0.0):
            return handed_rotation_vector_ctor(cls)(
                value=np.zeros(3, dtype=np.float64),
                coordinate_handedness=coordinate_handedness,
            )
        base = RotationVectorBase.from_axis_angle(axis, angle)
        return handed_rotation_vector_ctor(cls)(
            value=np.asarray(base.value, dtype=np.float64),
            coordinate_handedness=coordinate_handedness,
        )
