import os
import numpy as np

root: str = os.path.dirname(__file__)

weights = [np.random.randn(3, 4).astype(np.float32), np.random.randn(4, 2).astype(np.float32)]
biases = [np.random.randn(4).astype(np.float32), np.random.randn(2).astype(np.float32)]

print(weights)
print(biases)

with open(os.path.join(root, "data", "params.bin"), "wb") as f:
    for w in weights + biases:
        f.write(w.tobytes())

