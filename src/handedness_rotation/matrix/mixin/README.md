# mixin

## Overview

Behavior mixins composed into [`RotationMatrix`](../base.py).

## Components

| Component | Description |
| --------- | ----------- |
| [factory.py](./factory.py) | `RotationMatrixFactoryMixin`: identity, axis-angle, and orthogonalization of approximate matrices |
| [convert.py](./convert.py) | `RotationMatrixConvertMixin`: conversion to the opposite handedness and to Euler angles |
| [special.py](./special.py) | `RotationMatrixSpecialMixin`: `@` composition with matching handedness |

## Examples

```python
import numpy as np
from cartesian_axis import Axis, CoordinateHandedness
from units import AngleUnit

from handedness_rotation import IntrinsicRotationOrder, RotationMatrix

right_handed = RotationMatrix.from_axis_angle(
    Axis.Z, np.pi / 4, coordinate_handedness=CoordinateHandedness.RIGHT
)
left_handed = right_handed.to_opposite_handedness(flip_axis=Axis.Z)
euler = right_handed.to_euler_angles(IntrinsicRotationOrder.ZYX, AngleUnit.DEGREE)
```
