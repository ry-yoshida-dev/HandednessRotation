# vector

## Overview

`RotationVector`: an axis-angle vector from `rotation` with a `coordinate_handedness` tag.

## Components

| Component | Description |
| --------- | ----------- |
| [rotation_vector.py](./rotation_vector.py) | `RotationVector` |
| [protocol.py](./protocol.py) | Structural types the mixin is typed against |
| [mixin/](./mixin/README.md) | Factories |

## Examples

```python
import numpy as np
from cartesian_axis import CoordinateHandedness

from handedness_rotation import RotationVector

h = CoordinateHandedness.RIGHT
omega = RotationVector.from_axis_angle(axis=np.array([0.0, 0.0, 1.0]), angle=np.pi / 3, coordinate_handedness=h)
recovered = RotationVector.from_matrix(omega.rotation_matrix, coordinate_handedness=h)
```
