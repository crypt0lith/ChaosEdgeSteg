__all__ = [
    "LossyImageError",
    "SteganographyError",
    "adaptive_canny",
    "embed",
    "extract",
    "indices_3d",
]
import collections.abc as abc
import logging
import sys
import typing as tp
from functools import lru_cache
from hashlib import blake2b

import cv2
import mpmath as mp
import numpy as np

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

type ShapedNDArray[_ShapeT_co: tuple[int, ...], _SCT: np.generic] = np.ndarray[
    _ShapeT_co, np.dtype[_SCT]
]
type ArrayIndices[_Dim: int] = ShapedNDArray[tuple[_Dim], np.int64]
type TupleOf3[_T] = tuple[_T, _T, _T]
type Index3d[_Dim: int] = TupleOf3[ArrayIndices[_Dim]]
type Array3d[_SCT: np.generic] = ShapedNDArray[tuple[int, int, tp.Literal[3]], _SCT]
type GrayscaleArray = ShapedNDArray[tuple[int, int], np.uint8]


class LossyImageError(ValueError):
    pass


class SteganographyError(ValueError):
    pass


logger = logging.getLogger(__name__)


@lru_cache(maxsize=0x10)
def _logger_is_enabled(f, /):
    level = dict.get(
        {
            logger.critical: logging.CRITICAL,
            logger.error: logging.ERROR,
            logger.warning: logging.WARNING,
            logger.info: logging.INFO,
            logger.debug: logging.DEBUG,
        },
        f,
    )
    return level and logger.isEnabledFor(level)


def _attest_log[**P, R](
    f: abc.Callable[tp.Concatenate[str, P], R],
    /,
    msg: str,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R | None:
    if not _logger_is_enabled(f):
        return
    qualname = sys._getframe(1).f_code.co_qualname.replace(".<locals>.", ".")
    return f(f"{qualname}\t{msg}", *args, **kwargs)


mp.mp.dps = 200
K = 80
S = 1 << K
MASK64 = (1 << 64) - 1

DEFAULT_KEY = b"SECRET_PASSWORD"
MAGIC = b"CES"
HEADER_SIZE = len(MAGIC) + 4
PERSON = b"header", b"payload"


def _splitmix64(x: int, /) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & MASK64
    return (x ^ (x >> 31)) & MASK64


def _key_to_ic(key: abc.Buffer) -> TupleOf3[int]:
    """Convert a key into initial coords inside the folded-towel basin of
    attraction
    """
    h = blake2b(key, digest_size=24, person=b"key-ic").digest()
    # divide 192-bit hash into 64-bit hash word per axis
    wx, wy, wz = np.frombuffer(h, dtype="<u8").tolist()
    # x-,z-extents map to [0, 1) unit square
    # y-extent is [-0.1, 0.1)
    x0, z0 = ((v * S) >> 64 for v in [wx, wz])
    y0 = ((wy * (S // 5)) >> 64) - S // 10
    return x0, y0, z0


def indices_3d(arr: Array3d, key: abc.Buffer, count: int):
    """Return an array index 3-tuple for arr for count positions in chaotic
    pseudorandom order derived from key.

    The Rössler folded-towel map is used for the ordering, where the key hash
    is translated into initial coordinates for the hyperchaotic attractor.
    """
    if count < 0:
        raise ValueError("expected count to be non-negative number")

    # folded-towel coefficients
    A, B, C, D, E, F, G = (
        int(mp.nint(mp.mpf(v) * S))  # type: ignore
        for v in ["3.8", "0.05", "0.35", "0.1", "1.9", "3.78", "0.2"]
    )

    def step(xn, yn, zn, /):
        q = ((yn + C) * (S - 2 * zn)) // S
        xn1 = (A * ((xn * (S - xn)) // S)) // S - (B * q) // S
        yn1 = (D * (((q - S) * (S - (E * xn) // S)) // S)) // S
        zn1 = (F * ((zn * (S - zn)) // S)) // S + (G * yn) // S
        # when xn or zn are close to 0 or 1 (within ~2% edge of unit square),
        # the point produces an orbit which escapes to infinity. this makes
        # bigints explode and the consumer hangs. wrap it to settle on the
        # attractor instead.
        #
        # the probability of escape reaches near-zero after several steps, but
        # a 'safe' finite upper bound is not computable, so the guard runs here
        # instead of only during the burn-in phase.
        if xn1.bit_length() > K or zn1.bit_length() > K:
            xn1 %= S
            yn1 %= S
            zn1 %= S
        return xn1, yn1, zn1

    def fold(v, /):
        """xor-fold a 128-bit value to 64-bit, truncated"""
        return (v ^ (v >> 64)) & MASK64

    BURN_IN = 0xFF

    def generate():
        x, y, z = _key_to_ic(key)
        for _ in range(BURN_IN):
            x, y, z = step(x, y, z)
        n = count
        steps = 0
        max_steps = count * 10
        visited = np.zeros(arr.size, dtype=bool)
        while n > 0 and steps < max_steps:
            x, y, z = step(x, y, z)
            steps += 1
            w = fold(x) ^ fold(y) ^ fold(z)
            idx = _splitmix64(w ^ _splitmix64(steps)) % arr.size
            if visited[idx]:
                continue
            yield idx
            visited[idx] = True
            n -= 1
        _attest_log(
            logger.debug, "requested=%d generated=%d steps=%d", count, count - n, steps
        )

    indices: ArrayIndices = np.fromiter(generate(), dtype=np.int64, count=count)
    d0, d1, d2 = _i_to_yxz(indices, *arr.shape[:2])
    return d0, d1, d2


def _i_to_yxz[_Dim: int](indices: ArrayIndices[_Dim], h: int, w: int) -> Index3d[_Dim]:
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
    _attest_log(
        logger.debug,
        "target_edge_density=%.6f count=%d size=%d",
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
    _attest_log(
        logger.debug,
        "result lower=%d upper=%d err=%.6f reason=%r",
        prev_lower,
        prev_upper,
        best_err if best_err is not None else -1.0,
        reason or "max iter reached",
    )
    return best_edges


def _whiten(size: int, key: abc.Buffer, **kwargs):
    seed = np.frombuffer(blake2b(key, **kwargs).digest(), dtype=np.uint8)
    # we only whiten to prevent magic bytes from being used as a plaintext
    # oracle. 'repeating-key xor' is inert because linear ordering does not
    # survive downstream chaotic permutation.
    return np.resize(seed, size)


def embed(
    img: Array3d[np.uint8],
    payload: ShapedNDArray[tuple[int], np.uint8],
    key: tp.Optional[abc.Buffer] = None,
):
    if payload.size > img.size:
        raise SteganographyError("payload larger than cover image")
    if key is None:
        key = DEFAULT_KEY
    header = np.frombuffer(MAGIC + len(payload).to_bytes(4, "little"), dtype=np.uint8)
    assert header.size == HEADER_SIZE
    header_bits = np.unpackbits(header ^ _whiten(header.size, key, person=PERSON[0]))
    payload_bits = np.unpackbits(payload ^ _whiten(payload.size, key, person=PERSON[1]))
    _attest_log(logger.info, "payload_bytes=%d", int(payload.size))
    _attest_log(
        logger.debug,
        "header_bits=%d payload_bits=%d image_shape=%s",
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
            _attest_log(logger.debug, "edges_nonzero=%d", int(cv2.countNonZero(edges)))
        ys, xs = np.nonzero(edges)
        domain = np.empty((ys.size, 1, 3), dtype=np.uint8)
        try:
            d0, _, d2 = indices_3d(domain, key, count)
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
    key: tp.Optional[abc.Buffer] = None,
):
    if cover_img.shape != carrier_img.shape:
        raise ValueError(
            "shapes do not match: {.shape} and {.shape}".format(cover_img, carrier_img)
        )
    elif np.array_equal(cover_img, carrier_img):
        raise ValueError("cover image and carrier image are identical")
    if key is None:
        key = DEFAULT_KEY
    _attest_log(logger.info, "cover_shape=%s", cover_img.shape)
    gray = cv2.cvtColor(cover_img, cv2.COLOR_BGR2GRAY)
    ignored = np.zeros(gray.shape, dtype=bool)

    def get_idx(count: int):
        edges = adaptive_canny(gray, count) & ~ignored
        if logger.isEnabledFor(logging.DEBUG):
            _attest_log(logger.debug, "edges_nonzero=%d", int(cv2.countNonZero(edges)))
        ys, xs = np.nonzero(edges)
        domain = np.empty((ys.size, 1, 3), dtype=np.uint8)
        d0, _, d2 = indices_3d(domain, key, count)
        return ys[d0], xs[d0], d2

    header_idx = get_idx(HEADER_SIZE * 8)
    header = np.packbits(carrier_img[header_idx] & 1)
    header ^= _whiten(header.size, key, person=PERSON[0])
    header_bytes = header.tobytes()
    if header_bytes.startswith(MAGIC):
        header_bytes = header_bytes.removeprefix(MAGIC)
    else:
        raise ValueError("bad password")
    ignored[header_idx[:2]] = True
    payload_len = int.from_bytes(header_bytes, "little")
    _attest_log(logger.info, "payload_bytes=%d", payload_len)
    payload_idx = get_idx(payload_len * 8)
    payload = np.packbits(carrier_img[payload_idx] & 1)
    payload ^= _whiten(payload.size, key, person=PERSON[1])
    return payload
