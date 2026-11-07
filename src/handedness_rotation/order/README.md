# order

## Overview

Euler rotation orders: which axis each of the three angles is about, in the conventions of
`scipy.spatial.transform.Rotation`. They are the `order` of [`EulerAngles`](../euler/euler_angles.py).

## Components

| Component | Description |
| --------- | ----------- |
| [intrinsic.py](./intrinsic.py) | `IntrinsicRotationOrder`: six intrinsic axis sequences |
| [extrinsic.py](./extrinsic.py) | `ExtrinsicRotationOrder`: six extrinsic axis sequences |

## Examples

```python
from handedness_rotation import IntrinsicRotationOrder

axes = list(IntrinsicRotationOrder.ZYX)  # cartesian_axis.Axis values in sequence
```
