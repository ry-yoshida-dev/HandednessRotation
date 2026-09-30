# vector

## Overview

**`RotationVector`** is the axis–angle vector representation from **`rotation.RotationVector`**, plus **`coordinate_handedness`**. Direction is the rotation axis, magnitude is the angle in radians (same as upstream). Factories delegate to `rotation` and then attach the handedness tag.

## Example

Construct from axis–angle or from a validated 3×3 rotation matrix; `rotation_matrix` matches **`rotation`** (scipy `Rotation.from_rotvec` under the hood).

```python
import numpy as np
from cartesian_axis import CoordinateHandedness

from handedness_rotation import RotationVector

h = CoordinateHandedness.RIGHT
omega = RotationVector.from_axis_angle(
    axis=np.array([0.0, 0.0, 1.0]),
    angle=np.pi / 3,
    coordinate_handedness=h,
)
R = omega.rotation_matrix  # (3, 3) float64

omega2 = RotationVector.from_matrix(R, coordinate_handedness=h)
```

## Components

| Module | Description |
| ------ | ----------- |
| [`rotation_vector.py`](./rotation_vector.py) | `RotationVector` dataclass |
| [`mixin/factory.py`](./mixin/factory.py) | `RotationVectorFactoryMixin` — `from_matrix`, `from_axis_angle`, `zero_vector`, with handedness parameter |

See [`../matrix`](../matrix/README.md) and **`rotation`** for SO(3) validation rules on matrix-based construction.
