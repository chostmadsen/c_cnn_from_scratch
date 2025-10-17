import os
import numpy as np
from numpy.typing import NDArray
from tensorflow.keras.datasets import mnist
from helpers import write_tensor, write_label

out_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "c_cnn", "data")
data_pts: int = -1


def bin_mnist(path: str = out_dir, num: int = 1) -> None:
    r"""
    Reads mnist images and saves them as bin files in specified directory.

    :param path: directory location.
    :param num: number of items.
    """
    # load mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # combine mnist data (we love p-hacking)
    images: NDArray[np.float32] = np.float32(np.concatenate([x_train, x_test], axis=0).astype(np.float32) / 255.0)
    labels: NDArray[np.int32] = np.concatenate([y_train, y_test], axis=0).astype(np.int32)


    # make dirs
    os.makedirs(f"{path}/images", exist_ok=True)
    os.makedirs(f"{path}/labels", exist_ok=True)

    # save bins
    for i, (img, label) in enumerate(zip(images, labels)):
        if i == num: break
        print(f"\r[  {i}/{data_pts} pts  ]", end="")
        img_file: str = os.path.join(path, "images", f"img_{i}.bin")
        label_file: str = os.path.join(path, "labels", f"img_{i}.bin")
        write_tensor(array=img, file=img_file)
        write_label(label=int(label), file=label_file)
    print()
    return None


if __name__ == "__main__":
    bin_mnist(num=data_pts)
