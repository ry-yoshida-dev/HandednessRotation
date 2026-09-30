"""Tests for the handedness-tagged Quaternion."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray
from rotation import QuaternionFormat

from handedness_rotation import CoordinateHandedness, Quaternion


class TestQuaternion:
    def test_stores_handedness(self) -> None:
        """
        Verify the handedness tag is stored alongside the base fields.

        Input: identity quaternion (WXYZ), LEFT tag.
        Output: LEFT tag and WXYZ format.
        """
        quaternion: Quaternion = Quaternion(
            value=np.array([1.0, 0.0, 0.0, 0.0]),
            format=QuaternionFormat.WXYZ,
            coordinate_handedness=CoordinateHandedness.LEFT,
        )
        assert quaternion.coordinate_handedness is CoordinateHandedness.LEFT
        assert quaternion.format is QuaternionFormat.WXYZ

    def test_rejects_unnormalized_value(self) -> None:
        """
        Verify base-class normalization check still applies.

        Input: (2, 0, 0, 0).
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="normalized"):
            Quaternion(
                value=np.array([2.0, 0.0, 0.0, 0.0]),
                format=QuaternionFormat.WXYZ,
                coordinate_handedness=CoordinateHandedness.RIGHT,
            )

    @pytest.mark.parametrize(
        ("value", "quaternion_format"),
        [
            ([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)], QuaternionFormat.WXYZ),
            ([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)], QuaternionFormat.XYZW),
        ],
    )
    def test_rotation_matrix(
        self,
        rotation_z_90_matrix: NDArray[np.float64],
        value: list[float],
        quaternion_format: QuaternionFormat,
    ) -> None:
        """
        Verify the inherited rotation_matrix for both component layouts.

        Input: 90° Z rotation as WXYZ and XYZW.
        Output: 90° Z rotation matrix.
        """
        quaternion: Quaternion = Quaternion(
            value=np.array(value),
            format=quaternion_format,
            coordinate_handedness=CoordinateHandedness.RIGHT,
        )
        np.testing.assert_allclose(quaternion.rotation_matrix, rotation_z_90_matrix, atol=1e-12)

    def test_is_frozen(self) -> None:
        """
        Verify instances are immutable.

        Input: attempt to reassign coordinate_handedness.
        Output: FrozenInstanceError.
        """
        quaternion: Quaternion = Quaternion(
            value=np.array([1.0, 0.0, 0.0, 0.0]),
            format=QuaternionFormat.WXYZ,
            coordinate_handedness=CoordinateHandedness.RIGHT,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(quaternion, "coordinate_handedness", CoordinateHandedness.LEFT)
