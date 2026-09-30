"""Shared fixtures for handedness_rotation package tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from handedness_rotation import Axis, RotationMatrix


@pytest.fixture
def rotation_z_90_matrix() -> NDArray[np.float64]:
    """
    Return a 90° counter-clockwise rotation about +Z.

    Returns
    -------
    NDArray[np.float64]
        Rotation matrix mapping +X to +Y.
    """
    return np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def generic_rotation_matrix() -> NDArray[np.float64]:
    """
    Return a rotation with a non-zero angle about every axis.

    Returns
    -------
    NDArray[np.float64]
        Product Rx(0.3) @ Ry(-0.5) @ Rz(1.1).
    """
    rotation_x: RotationMatrix = RotationMatrix.from_axis_angle(axis=Axis.X, angle=0.3)
    rotation_y: RotationMatrix = RotationMatrix.from_axis_angle(axis=Axis.Y, angle=-0.5)
    rotation_z: RotationMatrix = RotationMatrix.from_axis_angle(axis=Axis.Z, angle=1.1)
    return np.asarray((rotation_x @ rotation_y @ rotation_z).value, dtype=np.float64)
