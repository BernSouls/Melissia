"""Item A safeguard: Gabor bars at 12 known angles pick matching channel."""
import math
import numpy as np
import parameters as P
from perception.filters import GaborBank


def _bar_image(angle_idx, size=64):
    n_theta = P.GABOR_N_THETA
    theta = angle_idx * math.pi / n_theta
    img = np.zeros((size, size), dtype=np.float32)
    # draw a bar (line) at theta through center
    cx, cy = size // 2, size // 2
    length = size // 2
    for t in range(-length, length + 1):
        x = int(round(cx + t * math.cos(theta)))
        y = int(round(cy + t * math.sin(theta)))
        if 0 <= x < size and 0 <= y < size:
            img[y, x] = 1.0
    # thicken
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=1.0)
    return img


def test_gabor_12_angles():
    bank = GaborBank.build_default(scale=1.0)
    kernels = bank.kernels  # (n_theta, n_sf, 2, H, W)
    # use a mid spatial frequency
    isf = kernels.shape[1] // 2
    ip = 0
    for ai in range(P.GABOR_N_THETA):
        img = _bar_image(ai)
        # pad to kernel size
        H, W = kernels.shape[-2:]
        # resize image to kernel size? Instead convolve full and take max energy
        responses = []
        for ci in range(P.GABOR_N_THETA):
            k = kernels[ci, isf, ip]
            # convolve
            from scipy.signal import fftconvolve
            resp = fftconvolve(img, k, mode="same")
            responses.append(float(np.abs(resp).mean()))
        best = int(np.argmax(responses))
        assert best == ai, (ai, best, responses)
