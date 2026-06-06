import numpy as np
from numpy.typing import NDArray

def rust_smith_normal_form(
    data: NDArray[np.int64],
    modulus: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]: ...
def rust_crt_snf(
    data: NDArray[np.int64],
    modulus: int,
    factors: list[tuple[int, int]],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]: ...
