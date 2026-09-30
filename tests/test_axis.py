"""Tests for axis enums."""

from __future__ import annotations

from handedness_rotation import ExtrinsicAxis3D, IntrinsicAxis3D, RotationAxis


class TestAxisEnums:
    def test_intrinsic_axis_values_are_uppercase(self) -> None:
        """
        Verify intrinsic axes use SciPy's uppercase (intrinsic) letters.

        Output: "X", "Y", "Z".
        """
        assert [axis.value for axis in IntrinsicAxis3D] == ["X", "Y", "Z"]

    def test_extrinsic_axis_values_are_lowercase(self) -> None:
        """
        Verify extrinsic axes use SciPy's lowercase (extrinsic) letters.

        Output: "x", "y", "z".
        """
        assert [axis.value for axis in ExtrinsicAxis3D] == ["x", "y", "z"]

    def test_rotation_axis_members(self) -> None:
        """
        Verify the semantic rotation axes.

        Output: roll, pitch, yaw.
        """
        assert [axis.value for axis in RotationAxis] == ["roll", "pitch", "yaw"]
