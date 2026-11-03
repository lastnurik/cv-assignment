"""One RGB/crop/resize contract. Normalization stays inside the Keras model."""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

@dataclass(frozen=True)
class PreprocessConfig:
    crop_top_fraction: float = 0.2  # 96 rows at original 480px height
    width: int = 224
    height: int = 224
    version: str = 'crop_top_20pct_rgb_bilinear_v1'

    def __post_init__(self):
        if not 0 <= self.crop_top_fraction < 1:
            raise ValueError('crop_top_fraction must be in [0, 1).')
        if self.width <= 0 or self.height <= 0:
            raise ValueError('Image dimensions must be positive.')

    def to_dict(self):
        return asdict(self)

DEFAULT = PreprocessConfig()

def open_rgb(source):
    """Apply EXIF orientation exactly once and detach pixels from input handle."""
    with Image.open(source) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert('RGB')

def crop_rgb(image, config=DEFAULT):
    im = image.convert('RGB')
    top = int(im.height * config.crop_top_fraction)
    return im.crop((0, top, im.width, im.height)), (0, top, im.width, im.height)

def model_input(image, *, already_cropped=False, config=DEFAULT):
    im = image.convert('RGB')
    if not already_cropped:
        im, _ = crop_rgb(im, config)
    im = im.resize((config.width, config.height), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32)[None, ...]

def camera_rgb(frame_bgr, *, mirror=True):
    arr = np.asarray(frame_bgr)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError('Camera input must be uint8 H×W×3 BGR.')
    rgb = arr[..., ::-1]
    if mirror:
        rgb = rgb[:, ::-1]
    return Image.fromarray(np.ascontiguousarray(rgb))

def probabilities(output, classes=3):
    p = np.asarray(output, dtype=np.float64)
    if p.ndim == 2 and p.shape[0] == 1:
        p = p[0]
    if p.shape != (classes,) or not np.isfinite(p).all():
        raise ValueError('Expected one finite probability per class.')
    if (p < 0).any() or (p > 1).any() or abs(float(p.sum()) - 1) > 1e-5:
        raise ValueError('Model output must already be probabilities summing to one.')
    return p  # Deliberately no second softmax.
