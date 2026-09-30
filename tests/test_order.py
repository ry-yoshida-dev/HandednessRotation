"""Tests for IntrinsicRotationOrder and ExtrinsicRotationOrder."""

from __future__ import annotations

import pytest

from handedness_rotation import Axis, ExtrinsicRotationOrder, IntrinsicRotationOrder


class TestIntrinsicRotationOrder:
    @pytest.mark.parametrize("order", list(IntrinsicRotationOrder))
    def test_value_is_uppercase_scipy_sequence(self, order: IntrinsicRotationOrder) -> None:
        """
        Verify values are uppercase so SciPy interprets them as intrinsic.

        Input: every intrinsic order.
        Output: value equals the member name.
        """
        assert order.value == order.name

    def test_iterates_cartesian_axes(self) -> None:
        """
        Verify iteration yields Axis members in sequence.

        Input: ZYX.
        Output: [Axis.Z, Axis.Y, Axis.X].
        """
        axes: list[Axis] = list(IntrinsicRotationOrder.ZYX)
        assert axes == [Axis.Z, Axis.Y, Axis.X]

    def test_case_accessors(self) -> None:
        """
        Verify as_upper and as_lower.

        Input: XZY.
        Output: "XZY" and "xzy".
        """
        assert IntrinsicRotationOrder.XZY.as_upper == "XZY"
        assert IntrinsicRotationOrder.XZY.as_lower == "xzy"


class TestExtrinsicRotationOrder:
    @pytest.mark.parametrize("order", list(ExtrinsicRotationOrder))
    def test_value_is_lowercase_scipy_sequence(self, order: ExtrinsicRotationOrder) -> None:
        """
        Verify values are lowercase so SciPy interprets them as extrinsic.

        Input: every extrinsic order.
        Output: value equals the lowercased member name.
        """
        assert order.value == order.name.lower()

    def test_iterates_cartesian_axes(self) -> None:
        """
        Verify iteration yields uppercase Axis members despite lowercase values.

        Input: yzx.
        Output: [Axis.Y, Axis.Z, Axis.X].
        """
        axes: list[Axis] = list(ExtrinsicRotationOrder.YZX)
        assert axes == [Axis.Y, Axis.Z, Axis.X]

    def test_case_accessors(self) -> None:
        """
        Verify as_upper and as_lower.

        Input: zxy.
        Output: "ZXY" and "zxy".
        """
        assert ExtrinsicRotationOrder.ZXY.as_upper == "ZXY"
        assert ExtrinsicRotationOrder.ZXY.as_lower == "zxy"

    @pytest.mark.parametrize("order", list(ExtrinsicRotationOrder))
    def test_same_axes_as_intrinsic_counterpart(self, order: ExtrinsicRotationOrder) -> None:
        """
        Verify each extrinsic order iterates the same axes as its intrinsic namesake.

        Input: every extrinsic order.
        Output: identical axis lists.
        """
        assert list(order) == list(IntrinsicRotationOrder[order.name])
