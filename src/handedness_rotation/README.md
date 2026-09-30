# handedness_rotation

## Overview

Extends [rotation](https://github.com/ry-yoshida-private/Rotation) with **`CoordinateHandedness`** (and related axis types from [cartesian-axis-3d](https://github.com/ry-yoshida-private/CartesianAxis3D)) on rotation containers. Quaternions, rotation vectors, and matrices share the same tagging pattern: handedness is recorded on the object and enforced where the API composes or constructs tagged types.

Public re-exports from `handedness_rotation` are defined in [`__init__.py`](./__init__.py).

## Components

| Area | Path | Description |
| ---- | ---- | ----------- |
| Matrix | [`matrix/`](./matrix/README.md) | `RotationMatrix` — SO(3) matrix + handedness; factories (identity, axis–angle, QR/SVD cleanup), conversion helpers, `@` composition with matching handedness |
| Axis | [`axis/`](./axis/README.md) | `IntrinsicAxis3D`, `ExtrinsicAxis3D` (SciPy Euler letters), `RotationAxis` (roll / pitch / yaw) |
| Order | [`order/`](./order/README.md) | `IntrinsicRotationOrder`, `ExtrinsicRotationOrder` — length-3 axis sequences for Euler triples |
| Euler | [`euler/`](./euler/README.md) | `EulerAngles` — triple + order + `AngleUnit`; builds 3×3 matrix like SciPy; `EulerIndexMapper` — roll/pitch/yaw index mapping via `AxisOrientation` |
| Vector | [`vector/`](./vector/README.md) | `RotationVector` — axis–angle vector + handedness; factories from matrix / axis–angle |
| Quaternion | [`quaternion/`](./quaternion/README.md) | `Quaternion` — normalized quaternion + format + handedness |

## Dependencies

Runtime use expects **`rotation`**, **`cartesian-axis-3d`** (imported as `cartesian_axis`), **`numpy`**, **`scipy`**, **`typing_extensions`**, and **`units`** (for `EulerAngles`). Declared in the repository root `pyproject.toml`; `requirements.txt` mirrors the same lines.
