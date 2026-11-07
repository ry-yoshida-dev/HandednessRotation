# HandednessRotation

## Overview

`handedness_rotation` extends the 3D rotation containers of [rotation](https://github.com/ry-yoshida-dev/Rotation) with a
`CoordinateHandedness` tag from [cartesian-axis-3d](https://github.com/ry-yoshida-dev/CartesianAxis3D). The numeric
values follow `rotation`; the tag records the frame convention, and composition and factories require matching tags.

| Representation | Class |
| --- | --- |
| Rotation matrix | `RotationMatrix` |
| Quaternion | `Quaternion` |
| Rotation vector | `RotationVector` |
| Euler angles | `EulerAngles` |

## Installation

```bash
pip install .
```

## Usage

```python
import numpy as np
from cartesian_axis import Axis, CoordinateHandedness

from handedness_rotation import RotationMatrix

R = RotationMatrix(
    value=np.eye(3),
    coordinate_handedness=CoordinateHandedness.RIGHT,
)

x_axis = R.extract_axis_rotation(Axis.X)
```
