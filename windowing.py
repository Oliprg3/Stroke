import numpy as np

WINDOWS = {
    "brain": (40, 80),
    "subdural": (50, 130),
    "soft": (40, 380),
    "bone": (600, 2800),
}

def hu_to_window(img_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    img = np.clip(img_hu, low, high)
    img = (img - low) / (high - low)
    return img.astype(np.float32)

def stack_windows(volume_hu: np.ndarray, presets=("brain","subdural","soft")) -> np.ndarray:
    chans = [hu_to_window(volume_hu, *WINDOWS[p]) for p in presets]
    return np.stack(chans, axis=0)  
