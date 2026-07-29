__all__ = [
    "LossyImageError",
    "SteganographyError",
    "adaptive_canny",
    "embed",
    "extract",
    "henon_indices",
    "henon_params",
    "shannon_entropy",
]
import hashlib
import logging
import typing as tp
from collections import Counter
from collections.abc import Buffer

import cv2
import mpmath as mp
import numpy as np

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

type ArrayBase[_ShapeT_co: tuple, _SCT: np.generic] = tp.Union[
    np.ndarray[_ShapeT_co, np.dtype[_SCT]], np.ndarray[tuple[int, ...], np.dtype[_SCT]]
]
type ArrayIndices[_Dim: int] = ArrayBase[tuple[_Dim], np.int64]
type Array3dIndex[_Dim: int] = TupleOf3[ArrayIndices[_Dim]]
type Array3d[_SCT: np.generic] = ArrayBase[tuple[int, int, tp.Literal[3]], _SCT]
type TupleOf3[_T] = tuple[_T, _T, _T]
type SupportsEntropy = tp.Sequence[tp.Hashable]
type GrayscaleArray = ArrayBase[tuple[int, int], np.uint8]


class LossyImageError(ValueError):
    pass


class SteganographyError(ValueError):
    pass


logger = logging.getLogger(__name__)


mp.mp.dps = 200
K = 48
S = 1 << K
MASK64 = (1 << 64) - 1


def _splitmix64(x: int, /) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & MASK64
    return (x ^ (x >> 31)) & MASK64


def _keyhash64(key: SupportsEntropy) -> int:
    if isinstance(key, Buffer):
        data = bytes(key)
    else:
        data = str(key).encode("utf-8", "surrogatepass")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def shannon_entropy(seq: SupportsEntropy, /) -> mp.mpf:
    counts = Counter(seq)
    n = mp.mpf(len(seq))
    ln2 = mp.log(2)
    h = mp.mpf("0")
    for c in counts.values():
        p = mp.mpf(c) / n
        h -= p * (mp.log(p) / ln2)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("shannon_entropy n=%d h=%s", int(n), h)
    return h


def _mp_to_fixed(x: mp.mpf, /) -> int:
    return int(mp.nint(x * S))


X0 = _mp_to_fixed(mp.mpf("0.123456789123"))
Y0 = _mp_to_fixed(mp.mpf("0.362436069531"))


def henon_params(key: SupportsEntropy) -> tuple[int, int]:
    ent = shannon_entropy(key)
    a = _mp_to_fixed((mp.mpf("56") - ent) / mp.mpf("40"))
    b = _mp_to_fixed((mp.mpf("24") + ent) / mp.mpf("80"))
    h = _keyhash64(key)
    a += ((h & 0xFFFFFFFF) - 0x80000000) >> 16
    b += (((h >> 32) & 0xFFFFFFFF) - 0x80000000) >> 16
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("henon_params a=%d b=%d", a, b)
    return a, b


def henon_indices(arr: Array3d, key: SupportsEntropy, count: int):
    if count < 0:
        raise ValueError("expected count to be non-negative number")

    def generate():
        a, b = henon_params(key)
        x, y = X0, Y0
        n = count
        max_steps = count * 10
        steps = 0
        visited = np.zeros(arr.size, dtype=bool)
        while n > 0 and steps < max_steps:
            steps += 1
            x = S + y - (a * (x**2)) // (S**2)
            y = (b * x) // S
            z = ((x & MASK64) ^ ((y & MASK64) << 1)) & MASK64
            idx = _splitmix64(z ^ _splitmix64(steps)) % arr.size
            if visited[idx]:
                continue
            visited[idx] = True
            yield idx
            n -= 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "henon_indices requested=%d generated=%d steps=%d",
                count,
                count - n,
                steps,
            )

    indices: ArrayIndices = np.fromiter(generate(), dtype=np.int64, count=count)
    d0, d1, d2 = _i_to_yxz(indices, *arr.shape[:2])
    return d0, d1, d2


def _i_to_yxz[_Dim: int](
    indices: ArrayIndices[_Dim], h: int, w: int
) -> Array3dIndex[_Dim]:
    plane = h * w
    z = indices // plane
    rem = indices - z * plane
    x = rem // h
    y = rem - x * h
    return y, x, z


def adaptive_canny(
    arr: GrayscaleArray,
    count: int,
    lo=(45, 85),
    hi=(135, 255),
    niter=10,
    tol: tp.Optional[float] = None,
) -> GrayscaleArray:
    if arr.dtype != np.uint8:
        raise ValueError("expected uint8")
    if tol is None:
        tol = 1.0 / arr.size
    lmin, lmax = (max(int(x), 0) & 0xFF for x in lo)
    hmin, hmax = (max(int(x), 0) & 0xFF for x in hi)
    target_density = count / arr.size
    target_edge_density = min(max(1.0 - target_density, 0.0), 1.0)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "adaptive_canny target_edge_density=%.6f count=%d size=%d",
            target_edge_density,
            count,
            arr.size,
        )
    filtered = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
    lo_t = 0.0
    hi_t = 1.0
    best_err = best_edges = None
    prev_lower = prev_upper = None
    reason = None
    for _ in range(niter):
        t = (lo_t + hi_t) * 0.5
        lower = int(round(lmin + t * (lmax - lmin)))
        upper = int(round(hmin + t * (hmax - hmin)))
        if lower > upper:
            lower, upper = upper, lower
        if best_edges is not None and (lower, upper) == (prev_lower, prev_upper):
            reason = "thresholds stable"
            break
        prev_lower, prev_upper = lower, upper
        edges = cv2.Canny(filtered, lower, upper)
        edge_density = cv2.countNonZero(edges) / arr.size
        err = abs(edge_density - target_edge_density)
        if best_err is None or err < best_err:
            best_err, best_edges = err, edges
        if err <= tol:
            reason = "tolerance met"
            break
        if edge_density > target_edge_density:
            lo_t = t
        else:
            hi_t = t
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "adaptive_canny result lower=%d upper=%d err=%.6f reason=%r",
            prev_lower,
            prev_upper,
            best_err if best_err is not None else -1.0,
            reason or "max iter reached",
        )
    return best_edges


DEFAULT_KEY = "SECRET_PASSWORD"
MAGIC = b"CES"
HEADER_BITS_SIZE = (len(MAGIC) + 4) * 8


def embed(
    img: Array3d[np.uint8],
    payload: ArrayBase[tuple[int], np.uint8],
    key: tp.Optional[SupportsEntropy] = None,
):
    if key is None:
        key = DEFAULT_KEY
    header = np.frombuffer(MAGIC + len(payload).to_bytes(4, "little"), dtype=np.uint8)
    header_bits = np.unpackbits(header)
    assert header_bits.size == HEADER_BITS_SIZE
    payload_bits = np.unpackbits(payload)
    logger.info("embed payload_bytes=%d", int(payload.size))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "embed header_bits=%d payload_bits=%d image_shape=%s",
            header_bits.size,
            payload_bits.size,
            img.shape,
        )
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    occupied = np.zeros(gray.shape, dtype=bool)
    for bits in [header_bits, payload_bits]:
        count = bits.size
        edges = adaptive_canny(gray, count) & ~occupied
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("embed edges_nonzero=%d", int(cv2.countNonZero(edges)))
        ys, xs = np.nonzero(edges)
        domain = np.empty((ys.size, 1, 3), dtype=np.uint8)
        try:
            d0, _, d2 = henon_indices(domain, key, count)
        except ValueError as e:
            if "iterator too short" in str(e):
                raise SteganographyError("payload too large for image") from e
            raise
        idx = ys[d0], xs[d0], d2
        img[idx] = (img[idx] & 0xFE) | bits
        occupied[idx[:2]] = True
    return img


def extract(
    cover_img: Array3d[np.uint8],
    carrier_img: Array3d[np.uint8],
    key: tp.Optional[SupportsEntropy] = None,
):
    if cover_img.shape != carrier_img.shape:
        raise ValueError(
            "shapes do not match: {.shape} and {.shape}".format(cover_img, carrier_img)
        )
    elif np.array_equal(cover_img, carrier_img):
        raise ValueError("cover image and carrier image are identical")
    if key is None:
        key = DEFAULT_KEY
    logger.info("extract cover_shape=%s", cover_img.shape)
    gray = cv2.cvtColor(cover_img, cv2.COLOR_BGR2GRAY)
    ignored = np.zeros(gray.shape, dtype=bool)

    def get_idx(count: int):
        edges = adaptive_canny(gray, count) & ~ignored
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("extract edges_nonzero=%d", int(cv2.countNonZero(edges)))
        ys, xs = np.nonzero(edges)
        domain = np.empty((ys.size, 1, 3), dtype=np.uint8)
        d0, _, d2 = henon_indices(domain, key, count)
        return ys[d0], xs[d0], d2

    header_idx = get_idx(HEADER_BITS_SIZE)
    header_bytes = np.packbits(carrier_img[header_idx] & 1).tobytes()
    if header_bytes.startswith(MAGIC):
        header_bytes = header_bytes.removeprefix(MAGIC)
    else:
        raise ValueError("bad password")
    ignored[header_idx[:2]] = True
    payload_len = int.from_bytes(header_bytes, "little")
    logger.info("extract payload_bytes=%d", payload_len)
    idx = get_idx(payload_len * 8)
    return np.packbits(carrier_img[idx] & 1)
