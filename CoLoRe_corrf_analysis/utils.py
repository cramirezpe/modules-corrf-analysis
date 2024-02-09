"""Module defining utils."""

from __future__ import annotations

from typing import TYPE_CHECKING
from scipy.ndimage import uniform_filter1d

import numpy as np

if TYPE_CHECKING:
    from typing import Optional, Tuple, TypeVar

    T1 = TypeVar("T1", bound=np.generic)
    T2 = TypeVar("T2", bound=np.generic)
    DTypeVar = TypeVar("DTypeVar", bound=np.dtype)
    DTypeVar2 = TypeVar("DTypeVar2", bound=np.dtype)
    DTypeVar3 = TypeVar("DTypeVar3", bound=np.dtype)


def weighted_uniform_filter1d(
    data: np.ndarray[Tuple[T1], DTypeVar],
    size: int,
    weights: Optional[np.ndarray[Tuple[T1], DTypeVar2]]=None,
    axis: int = -1,
    output: Optional[np.ndarray[Tuple[T1], DTypeVar3] | DTypeVar3] = None,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
) -> np.ndarray[Tuple[T1], DTypeVar3]:
    if weights is None:
        weights = np.ones_like(data, dtype=int)

    data_filtered = uniform_filter1d(
        data * weights,
        size=size,
        axis=axis,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
    )
    weight_filtered = uniform_filter1d(
        weights,
        size=size,
        axis=axis,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
    )

    w = weight_filtered > 0
    data_filtered[w] /= weight_filtered[w]
    
    return data_filtered