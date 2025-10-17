import struct
import numpy as np
from numpy.typing import NDArray

def write_tensor(array: NDArray[np.float32], file: str) -> None:
    r"""
    Writes an array to a bin file.

    :param array: NDArray of array.
    :param file: bim file to write to.
    """
    # set up list array
    if array.ndim > 3:
        raise ValueError("Dimension error: maximally 3D arrays.")
    dims = [1] * (3 - array.ndim) + list(array.shape)
    flat: list[float] = array.flatten().tolist()

    # write bin
    with open(file=file, mode="wb") as f:
        f.write(struct.pack("3Q", dims[1], dims[2], dims[0]))
        f.write(struct.pack(f"{len(flat)}f", *flat))
    return None


def write_label(label: int, file: str) -> None:
    r"""
    Writes a label to a bin file.

    :param label: label.
    :param file: bin file to write to.
    """
    # check label
    if label < 0:
        raise ValueError("Invalid label: label must be positive.")

    # write bin
    with open(file=file, mode="wb") as f:
        f.write(struct.pack("Q", label))
    return None
