import numpy as np

def make_2p5d_stacks(vol: np.ndarray, k=5, stride=1):
    Z, H, W = vol.shape
    half = k // 2
    idxs = list(range(half, Z-half, stride))
    stacks = []
    out_idxs = []
    for i in idxs:
        s = vol[i-half:i+half+1]
        stacks.append(s)
        out_idxs.append(i)
    return np.stack(stacks, axis=0), np.array(out_idxs)
