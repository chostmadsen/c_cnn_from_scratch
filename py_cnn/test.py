import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn


def print_tens(tens: list) -> None:
    # really awful function; will error with any slight modification of tens
    # don't use for anything other than quick debugging
    tens_l = np.array(tens).squeeze().tolist()
    for ten in tens_l:
        for mat in ten:
            print(" [", end="")
            for i, elm in enumerate(mat):
                print(f"{elm:.6f}\t", end="")
                if i == 0: print("\t", end="")
            print("]")
        print("\n")
    print(np.array(tens).shape)
    return None


def main() -> None:
    r"""
    Main test func.
    """
    # input params
    m: int = 12
    n: int = 12
    o: int = 2

    # kernel params
    m_k: int = 3
    n_k: int = 3
    o_k: int = 2
    m_k_stride: int = 1
    n_k_stride: int = 1

    # conv params
    num: int = 4

    # pooling params
    m_p: int = 2
    n_p: int = 2
    m_p_stride: int = 2
    n_p_stride: int = 2

    # set up arrays
    tens_arr: NDArray = np.arange(o * m * n).reshape(1, o, m, n)
    kern_arr: NDArray = np.arange(num * m_k * n_k * o_k).reshape(num, o_k, m_k, n_k)
    kern_bias: NDArray = np.arange(num)

    # set up torch objs
    tens: torch.tensor = torch.tensor(tens_arr, dtype=torch.float)
    conv: nn.Conv2d = nn.Conv2d(in_channels=o, out_channels=num, kernel_size=(m_k, n_k), stride=(m_k_stride, n_k_stride))
    conv.weight.data = torch.tensor(kern_arr, dtype=torch.float)
    conv.bias.data = torch.tensor(kern_bias, dtype=torch.float)
    pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=(m_p, n_p), stride=(m_p_stride, n_p_stride))

    # conv op
    out: torch.Tensor = conv(tens)
    out_pool: torch.Tensor = pool(out)

    # print options
    print_tens(tens=out.tolist())
    print("\n\n\n")
    print_tens(tens=out_pool.tolist())
    return None


if __name__ == "__main__":
    main()
