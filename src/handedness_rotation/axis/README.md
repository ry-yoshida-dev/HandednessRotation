# axis

## Overview

Single-axis enums for 3D rotations, using the Euler letters of `scipy.spatial.transform.Rotation`.

## Components

| Component | Description |
| --------- | ----------- |
| [intrinsic.py](./intrinsic.py) | `IntrinsicAxis3D`: intrinsic axis (uppercase value) |
| [extrinsic.py](./extrinsic.py) | `ExtrinsicAxis3D`: extrinsic axis (lowercase value) |
| [rotation.py](./rotation.py) | `RotationAxis`: roll, pitch, and yaw |

## Examples

```python
from handedness_rotation import ExtrinsicAxis3D, IntrinsicAxis3D

assert IntrinsicAxis3D.Z.value == "Z"
assert ExtrinsicAxis3D.Z.value == "z"
```
