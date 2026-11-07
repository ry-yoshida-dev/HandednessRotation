# quaternion

## Overview

`Quaternion`: a normalized quaternion from `rotation` with a `coordinate_handedness` tag.

## Components

| Component | Description |
| --------- | ----------- |
| [container.py](./container.py) | `Quaternion`: `value`, `format`, and `coordinate_handedness` |

## Examples

```python
import numpy as np
from cartesian_axis import CoordinateHandedness
from rotation import QuaternionFormat

from handedness_rotation import Quaternion

q = Quaternion(
    value=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    format=QuaternionFormat.WXYZ,
    coordinate_handedness=CoordinateHandedness.RIGHT,
)
R = q.rotation_matrix
```
