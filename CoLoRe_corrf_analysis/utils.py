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

def rebin_1d(
    original_array: np.ndarray[Tuple[T1], DTypeVar], 
    grid_array: np.ndarray[Tuple[T2], DTypeVar], 
    data: np.ndarray[Tuple[T1], DTypeVar], 
    errors: Optional[np.ndarray[Tuple[T1], DTypeVar]]=None,
) -> Tuple[np.ndarray[Tuple[T1], DTypeVar]]:
    """Rebin a 1D array.

    Args:
        original_array (np.ndarray): original array
        grid_array (np.ndarray): grid array
        data (np.ndarray): data array
        error (Optional[np.ndarray], optional): error array. Defaults to None.

    Returns: 
        Tuple[np.ndarray[Tuple[T1], DTypeVar]]: rebinned data and error if errors provided
            rebinned data and counts if errors not provided.
    """
    bins = find_bins(original_array, grid_array)

    data = np.bincount(
        bins, 
        weights=data / errors**2 if errors is not None else data, 
        minlength=len(grid_array)
    )
    ivar = np.bincount(
        bins, 
        weights=1 / errors**2 if errors is not None else np.ones_like(original_array), 
        minlength=len(grid_array)
    )

    w = ivar > 0 
    data[w] /= ivar[w]

    if errors is not None:
        errors = 1 / np.sqrt(ivar)
        return data, errors
    else:
        counts = ivar
        return data, ivar


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

def find_bins(
    original_array: np.ndarray[Tuple[T1], DTypeVar],
    grid_array: np.ndarray[Tuple[T2], DTypeVar],
) -> np.ndarray[Tuple[T1], DTypeVar]:
    """Find correspondent bins of the original array elements in the grid_array positions."""
    idx = np.searchsorted(grid_array, original_array)
    np.clip(idx, 0, len(grid_array) - 1, out=idx)

    prev_index_closer = (grid_array[idx - 1] - original_array) ** 2 <= (
        grid_array[idx] - original_array
    ) ** 2
    return idx - prev_index_closer