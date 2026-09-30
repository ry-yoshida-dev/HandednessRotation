"""Tests for EulerAngles."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from units import Angle, AngleUnit

from handedness_rotation import (
    Axis,
    EulerAngles,
    ExtrinsicRotationOrder,
    IntrinsicRotationOrder,
    RotationMatrix,
)


class TestEulerAnglesValidation:
    @pytest.mark.parametrize("shape", [(2,), (4,), (3, 1), (1, 3)])
    def test_rejects_invalid_shape(self, shape: tuple[int, ...]) -> None:
        """
        Verify only shape (3,) is accepted.

        Input: arrays with other shapes.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="shape"):
            EulerAngles(value=np.zeros(shape), order=IntrinsicRotationOrder.XYZ, unit=AngleUnit.RADIAN)

    @pytest.mark.parametrize(
        ("unit", "is_degrees"),
        [(AngleUnit.DEGREE, True), (AngleUnit.RADIAN, False)],
    )
    def test_is_degrees(self, unit: AngleUnit, is_degrees: bool) -> None:
        """
        Verify is_degrees reflects the unit.

        Input: degree and radian units.
        Output: True for degrees only.
        """
        euler_angles: EulerAngles = EulerAngles(value=np.zeros(3), order=IntrinsicRotationOrder.XYZ, unit=unit)
        assert euler_angles.is_degrees is is_degrees


class TestEulerAnglesRotationMatrix:
    @pytest.mark.parametrize("order", [*IntrinsicRotationOrder, *ExtrinsicRotationOrder])
    def test_zero_angles_give_identity(self, order: IntrinsicRotationOrder | ExtrinsicRotationOrder) -> None:
        """
        Verify zero angles compose to the identity.

        Input: zeros, every order.
        Output: identity matrix.
        """
        euler_angles: EulerAngles = EulerAngles(value=np.zeros(3), order=order, unit=AngleUnit.RADIAN)
        np.testing.assert_allclose(euler_angles.rotation_matrix, np.eye(3))

    def test_intrinsic_multiplies_on_the_right(self) -> None:
        """
        Verify intrinsic XYZ equals Rx(a) @ Ry(b) @ Rz(c).

        Input: (0.3, -0.5, 1.1) rad, intrinsic XYZ.
        Output: right-multiplied product of elementary rotations.
        """
        angles: NDArray[np.float64] = np.array([0.3, -0.5, 1.1])
        expected: NDArray[np.float64] = (
            RotationMatrix.from_axis_angle(axis=Axis.X, angle=0.3).value
            @ RotationMatrix.from_axis_angle(axis=Axis.Y, angle=-0.5).value
            @ RotationMatrix.from_axis_angle(axis=Axis.Z, angle=1.1).value
        )
        euler_angles: EulerAngles = EulerAngles(value=angles, order=IntrinsicRotationOrder.XYZ, unit=AngleUnit.RADIAN)
        np.testing.assert_allclose(euler_angles.rotation_matrix, expected, atol=1e-12)

    def test_extrinsic_multiplies_on_the_left(self) -> None:
        """
        Verify extrinsic xyz equals Rz(c) @ Ry(b) @ Rx(a).

        Input: (0.3, -0.5, 1.1) rad, extrinsic xyz.
        Output: left-multiplied product of elementary rotations.
        """
        angles: NDArray[np.float64] = np.array([0.3, -0.5, 1.1])
        expected: NDArray[np.float64] = (
            RotationMatrix.from_axis_angle(axis=Axis.Z, angle=1.1).value
            @ RotationMatrix.from_axis_angle(axis=Axis.Y, angle=-0.5).value
            @ RotationMatrix.from_axis_angle(axis=Axis.X, angle=0.3).value
        )
        euler_angles: EulerAngles = EulerAngles(value=angles, order=ExtrinsicRotationOrder.XYZ, unit=AngleUnit.RADIAN)
        np.testing.assert_allclose(euler_angles.rotation_matrix, expected, atol=1e-12)

    @pytest.mark.parametrize("intrinsic_order", list(IntrinsicRotationOrder))
    def test_intrinsic_equals_reversed_extrinsic(self, intrinsic_order: IntrinsicRotationOrder) -> None:
        """
        Verify intrinsic ABC with (a, b, c) equals extrinsic cba with (c, b, a).

        Input: every intrinsic order and its reversed extrinsic counterpart.
        Output: identical matrices.
        """
        angles: NDArray[np.float64] = np.array([0.3, -0.5, 1.1])
        extrinsic_order: ExtrinsicRotationOrder = ExtrinsicRotationOrder(intrinsic_order.as_lower[::-1])
        intrinsic: EulerAngles = EulerAngles(value=angles, order=intrinsic_order, unit=AngleUnit.RADIAN)
        extrinsic: EulerAngles = EulerAngles(value=angles[::-1].copy(), order=extrinsic_order, unit=AngleUnit.RADIAN)
        np.testing.assert_allclose(intrinsic.rotation_matrix, extrinsic.rotation_matrix, atol=1e-12)

    @pytest.mark.parametrize("order", [*IntrinsicRotationOrder, *ExtrinsicRotationOrder])
    def test_degrees_match_radians(self, order: IntrinsicRotationOrder | ExtrinsicRotationOrder) -> None:
        """
        Verify degree input is converted before composing.

        Input: (10, -20, 30) degrees and the same angles in radians.
        Output: identical matrices.
        """
        degrees: NDArray[np.float64] = np.array([10.0, -20.0, 30.0])
        in_degrees: EulerAngles = EulerAngles(value=degrees, order=order, unit=AngleUnit.DEGREE)
        in_radians: EulerAngles = EulerAngles(value=np.deg2rad(degrees), order=order, unit=AngleUnit.RADIAN)
        np.testing.assert_allclose(in_degrees.rotation_matrix, in_radians.rotation_matrix, atol=1e-12)

    def test_accepts_integer_degrees(self) -> None:
        """
        Verify integer arrays are converted to float64.

        Input: integer (0, 0, 90) degrees, intrinsic XYZ.
        Output: 90° Z rotation with dtype float64.
        """
        euler_angles: EulerAngles = EulerAngles(
            value=np.array([0, 0, 90]),
            order=IntrinsicRotationOrder.XYZ,
            unit=AngleUnit.DEGREE,
        )
        rotation_matrix: NDArray[np.float64] = euler_angles.rotation_matrix
        assert rotation_matrix.dtype == np.float64
        np.testing.assert_allclose(
            rotation_matrix,
            RotationMatrix.from_axis_angle(axis=Axis.Z, angle=np.pi / 2.0).value,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        ("unit", "units_per_degree"),
        [(AngleUnit.ARCMINUTE, 60.0), (AngleUnit.ARCSECOND, 3600.0)],
    )
    def test_sub_degree_units(self, unit: AngleUnit, units_per_degree: float) -> None:
        """
        Verify arcminute and arcsecond inputs are converted before composing.

        Input: (10, -20, 30) degrees expressed in arcminutes / arcseconds.
        Output: same matrix as the degree input.
        """
        degrees: NDArray[np.float64] = np.array([10.0, -20.0, 30.0])
        in_sub_degree: EulerAngles = EulerAngles(
            value=degrees * units_per_degree,
            order=IntrinsicRotationOrder.XYZ,
            unit=unit,
        )
        in_degrees: EulerAngles = EulerAngles(value=degrees, order=IntrinsicRotationOrder.XYZ, unit=AngleUnit.DEGREE)
        np.testing.assert_allclose(in_sub_degree.rotation_matrix, in_degrees.rotation_matrix, atol=1e-12)


class TestEulerAnglesComponents:
    def test_roll_pitch_yaw_read_components(self) -> None:
        """
        Verify roll, pitch and yaw expose value[0], value[1] and value[2].

        Input: (10, 20, 30) degrees.
        Output: length-1 Angle objects carrying each component and the unit.
        """
        euler_angles: EulerAngles = EulerAngles(
            value=np.array([10.0, 20.0, 30.0]),
            order=IntrinsicRotationOrder.XYZ,
            unit=AngleUnit.DEGREE,
        )
        components: list[Angle] = [euler_angles.roll, euler_angles.pitch, euler_angles.yaw]
        for component, expected in zip(components, [10.0, 20.0, 30.0], strict=True):
            np.testing.assert_allclose(component.value, [expected])
            assert component.unit is AngleUnit.DEGREE
