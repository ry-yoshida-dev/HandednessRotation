from __future__ import annotations

from typing import Protocol, TypeVar, cast

import numpy as np
from cartesian_axis import CoordinateHandedness


class HandedRotationMatrixProtocol(Protocol):
    """Structural surface used by matrix mixins (no rotation.RotationMatrix subclassing)."""

    @property
    def value(self) -> np.ndarray: ...

    @property
    def coordinate_handedness(self) -> CoordinateHandedness: ...

    def __init__(
        self,
        *,
        value: np.ndarray,
        coordinate_handedness: CoordinateHandedness,
    ) -> None: ...


THandedOut = TypeVar(
    "THandedOut",
    bound=HandedRotationMatrixProtocol,
    covariant=True,
)


class HandedRotationMatrixCls(Protocol[THandedOut]):
    """Class object that instantiates THandedOut with the handedness dataclass kwargs."""

    def __call__(
        self,
        *,
        value: np.ndarray,
        coordinate_handedness: CoordinateHandedness,
    ) -> THandedOut: ...


_SubclassHanded = TypeVar("_SubclassHanded", bound=HandedRotationMatrixProtocol)


def handed_rotation_matrix_ctor(
    cls: type[_SubclassHanded],
) -> HandedRotationMatrixCls[_SubclassHanded]:
    """Narrow ``type[_R]`` for ``(*, value: ndarray, coordinate_handedness: ...) -> _R``."""
    return cast(HandedRotationMatrixCls[_SubclassHanded], cls)
