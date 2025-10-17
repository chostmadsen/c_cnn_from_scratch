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
    if array.ndim > 3: raise ValueError("Dimension error: maximally 3D arrays.")
    dims: list[int] = [1] * (3 - array.ndim) + list(array.shape)
    flat: list[float] = array.flatten().tolist()

    # write bin
    with open(file=file, mode="wb") as f:
        f.write(struct.pack("3Q", dims[1], dims[2], dims[0]))
        f.write(struct.pack(f"{len(flat)}f", *flat))
    return None


def write_dense(weights: NDArray[np.float32], biases: NDArray[np.float32], file: str) -> None:
    r"""
    Writes a dense layer to a bin file.

    :param weights: NDArray of weights.
    :param biases: NDArray of biases.
    :param file: bim file to write to.
    """
    # weights
    w_dims: list[int] = [1] + list(weights.T.shape)
    w_flat: list[float] = weights.T.flatten().tolist()

    # biases
    b_dims: list[int] = [1, 1] + [list(biases.shape)[0]]
    b_flat: list[float] = biases.flatten().tolist()

    # write bin
    with open(file=file, mode="wb") as f:
        f.write(struct.pack("3Q", w_dims[1], w_dims[2], w_dims[0]))
        f.write(struct.pack(f"{len(w_flat)}f", *w_flat))
        f.write(struct.pack("3Q", b_dims[1], b_dims[2], b_dims[0]))
        f.write(struct.pack(f"{len(b_flat)}f", *b_flat))
    return None


def write_conv(kernels: NDArray[np.float32], biases: NDArray[np.float32], stride: tuple[int, int], file: str) -> None:
    r"""
    Writes a convolutional layer to a bin file.

    :param kernels: NDArray of kernels.
    :param biases: NDArray of biases.
    :param stride: kernel stride.
    :param file: bin file to write to.
    """
    # kernel
    num: int = int(kernels.shape[0])
    k_dims: list[int] = list(kernels.shape)[1:]
    k_flat: list[list[float]] = [k.flatten().tolist() for k in kernels]

    # biases
    b_flat: list[float] = biases.flatten().tolist()

    # write bin
    with open(file=file, mode="wb") as f:
        f.write(struct.pack("1Q", num))
        for idx in range(num):
            f.write(struct.pack("5Q", k_dims[1], k_dims[2], k_dims[0], stride[0], stride[1]))
            f.write(struct.pack(f"1f", b_flat[idx]))
            f.write(struct.pack(f"{len(k_flat[idx])}f", *k_flat[idx]))
    return None

def write_pool(dims: tuple[int, int], stride: tuple[int, int], file: str) -> None:
    r"""
    Writes a pooling kernel to a bin file.

    :param dims: kernel dims.
    :param stride: kernel stride.
    :param file: bin file to write to.
    """
    with (open(file=file, mode="wb")) as f:
        f.write(struct.pack("4Q", dims[0], dims[1], stride[0], stride[1]))
    return None


def write_label(label: int, file: str) -> None:
    r"""
    Writes a label to a bin file.

    :param label: label.
    :param file: bin file to write to.
    """
    # check label
    if label < 0: raise ValueError("Invalid label: label must be positive.")

    # write bin
    with open(file=file, mode="wb") as f:
        f.write(struct.pack("Q", label))
    return None
