"""Tests for the handedness-tagged RotationMatrix."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from numpy.typing import NDArray
from rotation import RotationMatrix as RotationMatrixBase
from units import AngleUnit

from handedness_rotation import (
    Axis,
    CoordinateHandedness,
    EulerAngles,
    ExtrinsicRotationOrder,
    IntrinsicRotationOrder,
    RotationMatrix,
)


class TestRotationMatrixValidation:
    def test_stores_value_and_handedness(self, rotation_z_90_matrix: NDArray[np.float64]) -> None:
        """
        Verify construction keeps the value and the handedness tag.

        Input: 90° Z rotation tagged LEFT.
        Output: identical value and LEFT tag.
        """
        rotation_matrix: RotationMatrix = RotationMatrix(
            value=rotation_z_90_matrix,
            coordinate_handedness=CoordinateHandedness.LEFT,
        )
        np.testing.assert_allclose(rotation_matrix.value, rotation_z_90_matrix)
        assert rotation_matrix.coordinate_handedness is CoordinateHandedness.LEFT
        assert rotation_matrix.is_determinant_correct

    def test_rejects_reflection(self) -> None:
        """
        Verify improper orthogonal matrices are rejected.

        Input: diag(1, 1, -1).
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="determinant"):
            RotationMatrix(
                value=np.diag([1.0, 1.0, -1.0]),
                coordinate_handedness=CoordinateHandedness.RIGHT,
            )

    def test_rejects_invalid_shape(self) -> None:
        """
        Verify non-(3, 3) arrays are rejected.

        Input: 2×2 identity.
        Output: ValueError.
        """
        with pytest.raises(ValueError):
            RotationMatrix(value=np.eye(2), coordinate_handedness=CoordinateHandedness.RIGHT)

    def test_is_frozen(self) -> None:
        """
        Verify instances are immutable.

        Input: attempt to reassign coordinate_handedness.
        Output: FrozenInstanceError.
        """
        rotation_matrix: RotationMatrix = RotationMatrix.unit_matrix()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(rotation_matrix, "coordinate_handedness", CoordinateHandedness.LEFT)


class TestRotationMatrixExtractAxisRotation:
    @pytest.mark.parametrize(
        ("axis", "column_index"),
        [(Axis.X, 0), (Axis.Y, 1), (Axis.Z, 2)],
    )
    def test_returns_column(
        self,
        rotation_z_90_matrix: NDArray[np.float64],
        axis: Axis,
        column_index: int,
    ) -> None:
        """
        Verify the column for each axis is returned.

        Input: 90° Z rotation and each Axis.
        Output: matching column of the matrix.
        """
        rotation_matrix: RotationMatrix = RotationMatrix(
            value=rotation_z_90_matrix,
            coordinate_handedness=CoordinateHandedness.RIGHT,
        )
        np.testing.assert_allclose(
            rotation_matrix.extract_axis_rotation(axis),
            rotation_z_90_matrix[:, column_index],
        )


class TestRotationMatrixFactories:
    @pytest.mark.parametrize("handedness", list(CoordinateHandedness))
    def test_unit_matrix(self, handedness: CoordinateHandedness) -> None:
        """
        Verify unit_matrix returns the identity with the requested tag.

        Input: each handedness.
        Output: identity matrix with that tag.
        """
        rotation_matrix: RotationMatrix = RotationMatrix.unit_matrix(handedness)
        np.testing.assert_allclose(rotation_matrix.value, np.eye(3))
        assert rotation_matrix.coordinate_handedness is handedness

    def test_unit_matrix_defaults_to_right(self) -> None:
        """
        Verify the default handedness is RIGHT.

        Output: RIGHT tag.
        """
        assert RotationMatrix.unit_matrix().coordinate_handedness is CoordinateHandedness.RIGHT

    @pytest.mark.parametrize(
        ("axis", "source", "expected"),
        [
            (Axis.X, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            (Axis.Y, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            (Axis.Z, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ],
    )
    def test_from_axis_angle_follows_right_hand_rule(
        self,
        axis: Axis,
        source: list[float],
        expected: list[float],
    ) -> None:
        """
        Verify +90° about each axis rotates the next axis onto the one after it.

        Input: +π/2 about X, Y, Z.
        Output: Y→Z, Z→X, X→Y respectively.
        """
        rotation_matrix: RotationMatrix = RotationMatrix.from_axis_angle(axis=axis, angle=np.pi / 2.0)
        np.testing.assert_allclose(rotation_matrix.value @ np.array(source), expected, atol=1e-12)

    @pytest.mark.parametrize("axis", list(Axis))
    def test_from_axis_angle_keeps_rotation_axis_fixed(self, axis: Axis) -> None:
        """
        Verify the rotation axis is an eigenvector with eigenvalue 1.

        Input: arbitrary angle about each axis.
        Output: axis unit vector unchanged.
        """
        rotation_matrix: RotationMatrix = RotationMatrix.from_axis_angle(axis=axis, angle=0.7)
        unit_vector: NDArray[np.float64] = np.eye(3)[axis.to_index]
        np.testing.assert_allclose(rotation_matrix.value @ unit_vector, unit_vector, atol=1e-12)

    def test_from_axis_angle_sets_handedness(self) -> None:
        """
        Verify the handedness tag is attached.

        Input: LEFT tag.
        Output: LEFT tag on the result.
        """
        rotation_matrix: RotationMatrix = RotationMatrix.from_axis_angle(
            axis=Axis.Z,
            angle=0.1,
            coordinate_handedness=CoordinateHandedness.LEFT,
        )
        assert rotation_matrix.coordinate_handedness is CoordinateHandedness.LEFT

    def test_from_approximate_matrix_with_svd(self, rotation_z_90_matrix: NDArray[np.float64]) -> None:
        """
        Verify SVD projection recovers a nearby valid rotation.

        Input: noisy 90° Z rotation, LEFT tag.
        Output: matrix close to the clean rotation, LEFT tag.
        """
        noise: NDArray[np.float64] = 1e-3 * np.random.default_rng(0).standard_normal((3, 3))
        fitted: RotationMatrix = RotationMatrix.from_approximate_matrix_with_SVD(
            rotation_z_90_matrix + noise,
            coordinate_handedness=CoordinateHandedness.LEFT,
        )
        np.testing.assert_allclose(fitted.value, rotation_z_90_matrix, atol=1e-2)
        assert fitted.coordinate_handedness is CoordinateHandedness.LEFT

    def test_from_approximate_matrix_by_qr(self, rotation_z_90_matrix: NDArray[np.float64]) -> None:
        """
        Verify QR projection returns a proper rotation close to the input.

        Input: noisy 90° Z rotation.
        Output: valid rotation close to the clean rotation up to column signs.
        """
        noise: NDArray[np.float64] = 1e-4 * np.random.default_rng(1).standard_normal((3, 3))
        fitted: RotationMatrix = RotationMatrix.from_approximate_matrix_by_qr(rotation_z_90_matrix + noise)
        assert fitted.is_determinant_correct
        np.testing.assert_allclose(np.abs(fitted.value), np.abs(rotation_z_90_matrix), atol=1e-2)

    @pytest.mark.parametrize(
        "approximate",
        [RotationMatrix.from_approximate_matrix_with_SVD, RotationMatrix.from_approximate_matrix_by_qr],
    )
    def test_approximation_corrects_reflection(
        self,
        approximate: Callable[[npt.ArrayLike], RotationMatrix],
    ) -> None:
        """
        Verify reflections are turned into proper rotations.

        Input: diag(1, 1, -1).
        Output: determinant +1.
        """
        fitted: RotationMatrix = approximate(np.diag([1.0, 1.0, -1.0]))
        assert np.isclose(np.linalg.det(fitted.value), 1.0)


class TestRotationMatrixComposition:
    def test_matmul_same_handedness(self, rotation_z_90_matrix: NDArray[np.float64]) -> None:
        """
        Verify composition of matching tags multiplies the matrices.

        Input: two 90° Z rotations tagged LEFT.
        Output: 180° Z rotation tagged LEFT.
        """
        rotation_matrix: RotationMatrix = RotationMatrix(
            value=rotation_z_90_matrix,
            coordinate_handedness=CoordinateHandedness.LEFT,
        )
        composed: RotationMatrix = rotation_matrix @ rotation_matrix
        assert isinstance(composed, RotationMatrix)
        np.testing.assert_allclose(composed.value, np.diag([-1.0, -1.0, 1.0]), atol=1e-12)
        assert composed.coordinate_handedness is CoordinateHandedness.LEFT

    def test_matmul_rejects_mismatched_handedness(self) -> None:
        """
        Verify composition with different tags fails.

        Input: RIGHT @ LEFT.
        Output: ValueError.
        """
        with pytest.raises(ValueError, match="coordinate_handedness"):
            RotationMatrix.unit_matrix(CoordinateHandedness.RIGHT) @ RotationMatrix.unit_matrix(
                CoordinateHandedness.LEFT
            )

    def test_matmul_rejects_base_rotation_matrix(self) -> None:
        """
        Verify composition with an untagged upstream matrix fails.

        Input: handedness RotationMatrix @ rotation.RotationMatrix.
        Output: TypeError.
        """
        with pytest.raises(TypeError, match="handedness"):
            RotationMatrix.unit_matrix() @ RotationMatrixBase(value=np.eye(3))

    def test_matmul_rejects_unrelated_object(self) -> None:
        """
        Verify composition with an unrelated operand defers and then fails.

        Input: handedness RotationMatrix @ object().
        Output: TypeError.
        """
        operand: object = object()
        with pytest.raises(TypeError):
            RotationMatrix.unit_matrix() @ operand


class TestRotationMatrixToOppositeHandedness:
    @pytest.mark.parametrize(
        ("handedness", "expected_handedness"),
        [
            (CoordinateHandedness.RIGHT, CoordinateHandedness.LEFT),
            (CoordinateHandedness.LEFT, CoordinateHandedness.RIGHT),
        ],
    )
    def test_flips_tag(
        self,
        handedness: CoordinateHandedness,
        expected_handedness: CoordinateHandedness,
    ) -> None:
        """
        Verify the tag is swapped.

        Input: identity with each tag.
        Output: the opposite tag.
        """
        converted: RotationMatrix = RotationMatrix.unit_matrix(handedness).to_opposite_handedness()
        assert converted.coordinate_handedness is expected_handedness

    @pytest.mark.parametrize("flip_axis", list(Axis))
    def test_value_is_conjugated_by_reflection(
        self,
        generic_rotation_matrix: NDArray[np.float64],
        flip_axis: Axis,
    ) -> None:
        """
        Verify the value equals D @ R @ D with D reflecting flip_axis.

        Input: generic rotation and each flip axis.
        Output: conjugated proper rotation.
        """
        reflection: NDArray[np.float64] = np.eye(3)
        reflection[flip_axis.to_index, flip_axis.to_index] = -1.0
        converted: RotationMatrix = RotationMatrix(
            value=generic_rotation_matrix,
            coordinate_handedness=CoordinateHandedness.RIGHT,
        ).to_opposite_handedness(flip_axis=flip_axis)
        np.testing.assert_allclose(converted.value, reflection @ generic_rotation_matrix @ reflection)
        assert converted.is_determinant_correct

    def test_default_flip_axis_is_z(self, generic_rotation_matrix: NDArray[np.float64]) -> None:
        """
        Verify the default flip axis is Z.

        Input: generic rotation.
        Output: same value as flip_axis=Axis.Z.
        """
        rotation_matrix: RotationMatrix = RotationMatrix(
            value=generic_rotation_matrix,
            coordinate_handedness=CoordinateHandedness.RIGHT,
        )
        np.testing.assert_allclose(
            rotation_matrix.to_opposite_handedness().value,
            rotation_matrix.to_opposite_handedness(flip_axis=Axis.Z).value,
        )

    def test_round_trip_is_identity(self, generic_rotation_matrix: NDArray[np.float64]) -> None:
        """
        Verify converting twice restores the original matrix and tag.

        Input: generic rotation tagged RIGHT.
        Output: original value and RIGHT tag.
        """
        rotation_matrix: RotationMatrix = RotationMatrix(
            value=generic_rotation_matrix,
            coordinate_handedness=CoordinateHandedness.RIGHT,
        )
        restored: RotationMatrix = rotation_matrix.to_opposite_handedness().to_opposite_handedness()
        np.testing.assert_allclose(restored.value, generic_rotation_matrix, atol=1e-12)
        assert restored.coordinate_handedness is CoordinateHandedness.RIGHT


class TestRotationMatrixToEulerAngles:
    @pytest.mark.parametrize("order", [*IntrinsicRotationOrder, *ExtrinsicRotationOrder])
    @pytest.mark.parametrize("unit", list(AngleUnit))
    def test_round_trip_through_euler_angles(
        self,
        generic_rotation_matrix: NDArray[np.float64],
        order: IntrinsicRotationOrder | ExtrinsicRotationOrder,
        unit: AngleUnit,
    ) -> None:
        """
        Verify matrix → Euler → matrix reproduces the original rotation.

        Input: generic rotation, every order, radian and degree.
        Output: EulerAngles with the requested order/unit whose matrix matches.
        """
        euler_angles: EulerAngles = RotationMatrix(
            value=generic_rotation_matrix,
            coordinate_handedness=CoordinateHandedness.RIGHT,
        ).to_euler_angles(order=order, unit=unit)
        assert euler_angles.order is order
        assert euler_angles.unit is unit
        np.testing.assert_allclose(euler_angles.rotation_matrix, generic_rotation_matrix, atol=1e-10)

    @pytest.mark.parametrize(
        ("unit", "units_per_degree"),
        [(AngleUnit.DEGREE, 1.0), (AngleUnit.ARCMINUTE, 60.0), (AngleUnit.ARCSECOND, 3600.0)],
    )
    def test_angles_are_expressed_in_requested_unit(
        self,
        rotation_z_90_matrix: NDArray[np.float64],
        unit: AngleUnit,
        units_per_degree: float,
    ) -> None:
        """
        Verify the returned values are scaled to the requested unit.

        Input: 90° Z rotation, intrinsic XYZ, degree / arcminute / arcsecond.
        Output: [0, 0, 90 × units_per_degree].
        """
        euler_angles: EulerAngles = RotationMatrix(
            value=rotation_z_90_matrix,
            coordinate_handedness=CoordinateHandedness.RIGHT,
        ).to_euler_angles(order=IntrinsicRotationOrder.XYZ, unit=unit)
        np.testing.assert_allclose(euler_angles.value, [0.0, 0.0, 90.0 * units_per_degree], atol=1e-8)

    def test_known_angles_about_z(self, rotation_z_90_matrix: NDArray[np.float64]) -> None:
        """
        Verify a pure Z rotation maps to a single non-zero Euler angle.

        Input: 90° Z rotation, intrinsic XYZ, degrees.
        Output: [0, 0, 90].
        """
        euler_angles: EulerAngles = RotationMatrix(
            value=rotation_z_90_matrix,
            coordinate_handedness=CoordinateHandedness.RIGHT,
        ).to_euler_angles(order=IntrinsicRotationOrder.XYZ, unit=AngleUnit.DEGREE)
        np.testing.assert_allclose(euler_angles.value, [0.0, 0.0, 90.0], atol=1e-10)
