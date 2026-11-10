import cv2
import numpy as np

class StressSimulator:
    """Physics-based degradation operators for industrial environments."""
    
    @staticmethod
    def _lux(img: np.ndarray, gain: float, gamma: float) -> np.ndarray:
        """Linear scaling (exposure / light level) followed by a gamma curve (sensor and display response)."""
        x = np.clip(img.astype(np.float32) / 255.0 * gain, 0.0, 1.0)
        return np.clip(np.power(x, gamma) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_lux_plus(img: np.ndarray, val: int) -> np.ndarray:
        """More light / over-exposure: gain 1 + val/50, gamma 1 / (1 + val/100) lifts the midtones. 0 = unchanged."""
        if val <= 0:
            return img
        return StressSimulator._lux(img, 1 + val / 50, 1 / (1 + val / 100))

    @staticmethod
    def apply_lux_minus(img: np.ndarray, val: int) -> np.ndarray:
        """Less light / under-exposure: gain 1 / (1 + val/50), gamma 1 + val/100 crushes the shadows. 0 = unchanged."""
        if val <= 0:
            return img
        return StressSimulator._lux(img, 1 / (1 + val / 50), 1 + val / 100)

    @staticmethod
    def apply_mblur(img: np.ndarray, val: int, angle: float = 0.0) -> np.ndarray:
        """Motion blur: directional line kernel of int(val/10)*2+1 pixels (same length as before), along `angle`
        degrees (0 = horizontal, e.g. a conveyor; 90 = vertical). 0 = unchanged."""
        length = int(val / 10) * 2 + 1
        if length <= 1:
            return img
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0
        if angle % 180:
            c = (length / 2 - 0.5, length / 2 - 0.5)
            k = cv2.warpAffine(k, cv2.getRotationMatrix2D(c, angle, 1.0), (length, length))
        k /= k.sum()
        return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)

    @staticmethod
    def apply_iso_noise(img: np.ndarray, val: int, seed: int = 0) -> np.ndarray:
        """Sensor noise: Gaussian, sigma = val grey levels. Seeded, so the same input gives the same image
        (pass a different seed per image or repeat for variation). 0 = unchanged."""
        if val <= 0:
            return img
        noise = np.random.default_rng(seed).normal(0, val, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_dpi(img: np.ndarray, val: int) -> np.ndarray:
        """Working distance / resolution loss: area-based downsampling to (100 - val)% of the pixels per side,
        then back to the original size. 0 = unchanged; at least 2 pixels per side remain."""
        scale = max(0.0, 1.0 - val / 100)
        if scale >= 1.0:
            return img
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def apply_defocus(img: np.ndarray, val: int) -> np.ndarray:
        """Focal drift: Gaussian blur with sigma = val / 10 pixels. 0 = unchanged."""
        sigma = val / 10
        if sigma <= 0:
            return img
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
