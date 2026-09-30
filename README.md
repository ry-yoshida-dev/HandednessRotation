# HandednessRotation

`HandednessRotation` is a Python package for 3D rotations that carries **coordinate handedness** (`CoordinateHandedness`) and **axis conventions** (`Axis`, `AxisOrientation`) alongside matrix, quaternion, rotation-vector, and Euler representations.

It builds on [rotation](https://github.com/ry-yoshida-private/Rotation) and [cartesian-axis-3d](https://github.com/ry-yoshida-private/CartesianAxis3D). Numeric maps (for example `rotation_matrix`) match the upstream `rotation` types; `coordinate_handedness` is **metadata** (composition and factories require matching tags unless documented otherwise).

## Installation

From a clone of this repository. **`pyproject.toml`** declares **NumPy**, **SciPy**, **`typing_extensions`**, and Git installs of **`units`**, **`cartesian-axis-3d`**, and **`rotation`**; **`requirements.txt`** is the same runtime stack for an optional `pip install -r` pass (CI, strict venv setup, and so on).

```bash
pip install -r requirements.txt
pip install -e .
```

Skip the first line if you only want **`pip`** to resolve dependencies from **`pyproject.toml`**.

## Example

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

## Package layout

Subpackages and entry points are summarized in [`src/handedness_rotation/README.md`](src/handedness_rotation/README.md) (matrix, axis, order, euler, vector, quaternion).
