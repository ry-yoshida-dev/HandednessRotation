from __future__ import annotations

from typing import Protocol, TypeVar, cast

from cartesian_axis import CoordinateHandedness
from rotation.types import FloatArray
from rotation.vector.protocol import RotationVectorLike


class HandedRotationVectorProtocol(RotationVectorLike, Protocol):
    """Structural surface for vector mixins (extends upstream ``RotationVectorLike``)."""

    @property
    def coordinate_handedness(self) -> CoordinateHandedness: ...

    def __init__(
        self,
        *,
        value: FloatArray,
        coordinate_handedness: CoordinateHandedness,
    ) -> None: ...


THandedVectorOut = TypeVar(
    "THandedVectorOut",
    bound=HandedRotationVectorProtocol,
    covariant=True,
)


class HandedRotationVectorCls(Protocol[THandedVectorOut]):
    def __call__(
        self,
        *,
        value: FloatArray,
        coordinate_handedness: CoordinateHandedness,
    ) -> THandedVectorOut: ...


_SubclassHanded = TypeVar("_SubclassHanded", bound=HandedRotationVectorProtocol)


def handed_rotation_vector_ctor(
    cls: type[_SubclassHanded],
) -> HandedRotationVectorCls[_SubclassHanded]:
    return cast(HandedRotationVectorCls[_SubclassHanded], cls)
