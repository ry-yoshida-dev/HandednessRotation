# handedness_rotation

## Overview

Rotation containers tagged with `CoordinateHandedness`, and the axis and order types used by Euler angles. Public names
are re-exported from [`__init__.py`](./__init__.py).

## Components

| Component | Description |
| --------- | ----------- |
| [matrix/](./matrix/README.md) | `RotationMatrix`: SO(3) matrix with a handedness tag |
| [axis/](./axis/README.md) | Single rotation axes: intrinsic, extrinsic, and roll / pitch / yaw |
| [order/](./order/README.md) | Euler rotation orders |
| [euler/](./euler/README.md) | `EulerAngles` and `EulerIndexMapper` |
| [vector/](./vector/README.md) | `RotationVector`: axis-angle vector with a handedness tag |
| [quaternion/](./quaternion/README.md) | `Quaternion`: normalized quaternion with a handedness tag |
| [py.typed](./py.typed) | Marker declaring the package as typed for static analysers |
