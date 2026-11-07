# matrix

## Overview

`RotationMatrix`: a proper 3x3 rotation matrix from `rotation` with a `coordinate_handedness` tag.

## Components

| Component | Description |
| --------- | ----------- |
| [base.py](./base.py) | `RotationMatrix` and axis column extraction |
| [protocol.py](./protocol.py) | Structural types the mixins are typed against |
| [mixin/](./mixin/README.md) | Factories, conversions, and `@` composition |

## Examples

```python
import numpy as np
from cartesian_axis import Axis, CoordinateHandedness

from handedness_rotation import RotationMatrix

h = CoordinateHandedness.RIGHT
R_z = RotationMatrix.from_axis_angle(Axis.Z, np.pi / 6, coordinate_handedness=h)
R_y = RotationMatrix.from_axis_angle(Axis.Y, np.pi / 4, coordinate_handedness=h)
R = R_z @ R_y  # requires the same handedness on both operands

x_body = R.extract_axis_rotation(Axis.X)
```
