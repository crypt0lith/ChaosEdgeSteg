from typing import Hashable, Literal, Sequence, Union

import numpy as np

type ArrayBase[_ShapeT_co: tuple, _SCT = np.generic] = Union[
    np.ndarray[_ShapeT_co, np.dtype[_SCT]], np.ndarray[tuple[int, ...], np.dtype[_SCT]]
]
type ArrayIndices[_Dim = int] = ArrayBase[tuple[_Dim], np.int64]
type Array3dIndex[_Dim = int] = TupleOf3[ArrayIndices[_Dim]]
type Array3d[_SCT = np.generic] = ArrayBase[tuple[int, int, Literal[3]], _SCT]
type TupleOf3[_T] = tuple[_T, _T, _T]
type SupportsEntropy = Sequence[Hashable]
type GrayscaleArray = ArrayBase[tuple[int, int], np.uint8]
