"""Tests for EulerIndexMapper."""

from __future__ import annotations

import numpy as np
import pytest
from cartesian_axis import AxisOrientation
from numpy.typing import NDArray

from handedness_rotation import (
    Axis,
    EulerIndexMapper,
    ExtrinsicRotationOrder,
    IntrinsicRotationOrder,
    RotationAxis,
)


class TestEulerIndexMapper:
    def test_identity_mapping_for_matching_order(self) -> None:
        """
        Verify XYZ with forward=X, right=Y, up=Z maps roll/pitch/yaw to 0/1/2.

        Input: intrinsic XYZ, X-forward Y-right Z-up.
        Output: {ROLL: 0, PITCH: 1, YAW: 2}.
        """
        mapper: EulerIndexMapper = EulerIndexMapper(
            rotation_order=IntrinsicRotationOrder.XYZ,
            axis_orientation=AxisOrientation(forward=Axis.X, right=Axis.Y, up=Axis.Z),
        )
        assert mapper.euler_index_mapper == {
            RotationAxis.ROLL: 0,
            RotationAxis.PITCH: 1,
            RotationAxis.YAW: 2,
        }

    @pytest.mark.parametrize(
        "rotation_order",
        [IntrinsicRotationOrder.YXZ, ExtrinsicRotationOrder.YXZ],
    )
    def test_mapping_for_y_up_orientation(
        self,
        rotation_order: IntrinsicRotationOrder | ExtrinsicRotationOrder,
    ) -> None:
        """
        Verify indices follow the order for a Z-forward, X-right, Y-up frame.

        Input: YXZ (intrinsic and extrinsic), forward=Z, right=X, up=Y.
        Output: {YAW: 0, PITCH: 1, ROLL: 2}.
        """
        mapper: EulerIndexMapper = EulerIndexMapper(
            rotation_order=rotation_order,
            axis_orientation=AxisOrientation(forward=Axis.Z, right=Axis.X, up=Axis.Y),
        )
        assert mapper.euler_index_mapper == {
            RotationAxis.YAW: 0,
            RotationAxis.PITCH: 1,
            RotationAxis.ROLL: 2,
        }

    def test_call_reorders_to_roll_pitch_yaw(self) -> None:
        """
        Verify __call__ reorders angles into [roll, pitch, yaw].

        Input: ZYX angles (yaw, pitch, roll) = (30, 20, 10), X-forward Y-right Z-up.
        Output: [10, 20, 30].
        """
        mapper: EulerIndexMapper = EulerIndexMapper(
            rotation_order=IntrinsicRotationOrder.ZYX,
            axis_orientation=AxisOrientation(forward=Axis.X, right=Axis.Y, up=Axis.Z),
        )
        reordered: NDArray[np.float64] = mapper(np.array([30.0, 20.0, 10.0]))
        np.testing.assert_array_equal(reordered, [10.0, 20.0, 30.0])

    def test_static_builder_matches_instance(self) -> None:
        """
        Verify get_euler_index_mapper equals the mapping stored at construction.

        Input: extrinsic zxy, X-forward Y-right Z-up.
        Output: identical dictionaries.
        """
        axis_orientation: AxisOrientation = AxisOrientation(forward=Axis.X, right=Axis.Y, up=Axis.Z)
        mapper: EulerIndexMapper = EulerIndexMapper(
            rotation_order=ExtrinsicRotationOrder.ZXY,
            axis_orientation=axis_orientation,
        )
        assert mapper.euler_index_mapper == EulerIndexMapper.get_euler_index_mapper(
            rotation_order=ExtrinsicRotationOrder.ZXY,
            axis_orientation=axis_orientation,
        )
