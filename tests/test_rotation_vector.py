"""Tests for the handedness-tagged RotationVector."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from handedness_rotation import CoordinateHandedness, RotationVector


class TestRotationVectorValidation:
    def test_rejects_invalid_shape(self) -> None:
        """
        Verify only shape (3,) is accepted.

        Input: length-4 array.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            RotationVector(value=np.zeros(4), coordinate_handedness=CoordinateHandedness.RIGHT)


class TestRotationVectorFactories:
    @pytest.mark.parametrize("handedness", list(CoordinateHandedness))
    def test_zero_vector(self, handedness: CoordinateHandedness) -> None:
        """
        Verify zero_vector is the identity rotation with the requested tag.

        Input: each handedness.
        Output: (0, 0, 0), identity matrix, and that tag.
        """
        rotation_vector: RotationVector = RotationVector.zero_vector(coordinate_handedness=handedness)
        np.testing.assert_array_equal(rotation_vector.value, np.zeros(3))
        np.testing.assert_allclose(rotation_vector.rotation_matrix, np.eye(3))
        assert rotation_vector.coordinate_handedness is handedness

    def test_from_axis_angle_normalizes_axis(self) -> None:
        """
        Verify the axis is normalized and scaled by the angle.

        Input: axis (0, 0, 2), angle π/2, LEFT tag.
        Output: (0, 0, π/2) with LEFT tag.
        """
        rotation_vector: RotationVector = RotationVector.from_axis_angle(
            [0.0, 0.0, 2.0],
            np.pi / 2.0,
            coordinate_handedness=CoordinateHandedness.LEFT,
        )
        assert isinstance(rotation_vector, RotationVector)
        np.testing.assert_allclose(rotation_vector.value, [0.0, 0.0, np.pi / 2.0])
        assert rotation_vector.coordinate_handedness is CoordinateHandedness.LEFT

    def test_from_axis_angle_defaults_to_right(self) -> None:
        """
        Verify the default handedness is RIGHT.

        Output: RIGHT tag.
        """
        rotation_vector: RotationVector = RotationVector.from_axis_angle([1.0, 0.0, 0.0], 0.1)
        assert rotation_vector.coordinate_handedness is CoordinateHandedness.RIGHT

    def test_from_matrix(self, rotation_z_90_matrix: NDArray[np.float64]) -> None:
        """
        Verify axis–angle extraction from a matrix.

        Input: 90° Z rotation, LEFT tag.
        Output: (0, 0, π/2) with LEFT tag.
        """
        rotation_vector: RotationVector = RotationVector.from_matrix(
            rotation_z_90_matrix,
            coordinate_handedness=CoordinateHandedness.LEFT,
        )
        assert isinstance(rotation_vector, RotationVector)
        np.testing.assert_allclose(rotation_vector.value, [0.0, 0.0, np.pi / 2.0], atol=1e-12)
        assert rotation_vector.coordinate_handedness is CoordinateHandedness.LEFT

    def test_from_matrix_rejects_reflection(self) -> None:
        """
        Verify improper matrices are rejected by default.

        Input: diag(1, 1, -1).
        Output: ValueError.
        """
        with pytest.raises(ValueError):
            RotationVector.from_matrix(np.diag([1.0, 1.0, -1.0]))

    def test_matrix_round_trip(self, generic_rotation_matrix: NDArray[np.float64]) -> None:
        """
        Verify matrix → vector → matrix reproduces the original rotation.

        Input: generic rotation.
        Output: identical matrix.
        """
        rotation_vector: RotationVector = RotationVector.from_matrix(generic_rotation_matrix)
        np.testing.assert_allclose(rotation_vector.rotation_matrix, generic_rotation_matrix, atol=1e-10)
