# tests

## Overview

Pytest suite for the `handedness_rotation` package: handedness tagging, factories, composition rules, handedness conversion, and Euler angle handling.

Expected values are derived from known rotations and algebraic identities (for example, intrinsic `ABC` equals extrinsic `cba` with reversed angles) rather than from SciPy directly.

## Components

| Component | Description |
|-----------|-------------|
| [conftest.py](./conftest.py) | Shared fixtures (90° Z rotation, generic rotation matrix) |
| [test_axis.py](./test_axis.py) | `IntrinsicAxis3D`, `ExtrinsicAxis3D`, `RotationAxis` values |
| [test_order.py](./test_order.py) | `IntrinsicRotationOrder` / `ExtrinsicRotationOrder` values, iteration, case accessors |
| [test_rotation_matrix.py](./test_rotation_matrix.py) | `RotationMatrix` validation, factories, `@` composition, `to_opposite_handedness`, `to_euler_angles` |
| [test_euler_angles.py](./test_euler_angles.py) | `EulerAngles` validation, matrix composition order, unit handling, component accessors |
| [test_euler_index_mapper.py](./test_euler_index_mapper.py) | `EulerIndexMapper` index mapping and reordering to roll / pitch / yaw |
| [test_rotation_vector.py](./test_rotation_vector.py) | `RotationVector` validation, factories, matrix round trip |
| [test_quaternion.py](./test_quaternion.py) | `Quaternion` handedness tag, validation, matrix conversion |

## Examples

```bash
cd /path/to/HandednessRotation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
