# euler

## Overview

Euler angles with their rotation order and angle unit, and the mapping from storage order to roll / pitch / yaw.

## Components

| Component | Description |
| --------- | ----------- |
| [euler_angles.py](./euler_angles.py) | `EulerAngles`: three angles, an order, and an `AngleUnit`, convertible to a rotation matrix |
| [euler_index_mapper.py](./euler_index_mapper.py) | `EulerIndexMapper`: reorders angles to `[roll, pitch, yaw]` for an `AxisOrientation` |

## Examples

```python
import numpy as np
from cartesian_axis import Axis, AxisOrientation
from units import AngleUnit

from handedness_rotation import EulerAngles, EulerIndexMapper, IntrinsicRotationOrder

euler = EulerAngles(
    value=np.array([5.0, 10.0, 15.0], dtype=np.float64),
    order=IntrinsicRotationOrder.ZYX,
    unit=AngleUnit.DEGREE,
)
R = euler.rotation_matrix

mapper = EulerIndexMapper(
    rotation_order=euler.order,
    axis_orientation=AxisOrientation(forward=Axis.X, right=Axis.Y, up=Axis.Z),
)
roll_pitch_yaw = mapper(euler.value)
```
