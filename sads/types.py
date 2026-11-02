from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
BoolArray: TypeAlias = NDArray[np.bool_]
PathLike: TypeAlias = str | Path
Distribution: TypeAlias = Literal["normal", "student"]
StressSignal: TypeAlias = Literal["stlfsi", "realized_vol"]
TailForecast: TypeAlias = tuple[FloatArray, FloatArray, FloatArray]
