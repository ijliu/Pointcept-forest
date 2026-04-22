import numpy as np
from pathlib import Path


root = Path("data/boreal3d/train").glob("*/segment.npy")

all = {0:0,1:1,2:2,3:3}
for name in root:

    data = np.load(name)

    for i in range(4):
        idx = np.where(data == i)[0]
        all[i] = all[i] + len(idx)

print(all)
