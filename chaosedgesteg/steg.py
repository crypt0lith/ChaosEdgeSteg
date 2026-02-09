__all__ = ['adaptive_canny', 'embed', 'extract']

import logging
from typing import Optional

import cv2
import numpy as np

from . import SteganographyError, logger as _base_logger
from ._typing import Array3d, ArrayBase, GrayscaleArray, SupportsEntropy
from .henon import henon_indices

logger = _base_logger.getChild('steg')


def adaptive_canny(
    arr: GrayscaleArray,
    count: int,
    lo=(45, 85),
    hi=(135, 255),
    niter=10,
    tol: Optional[float] = None,
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


DEFAULT_KEY = 'SECRET_PASSWORD'


def embed(
    img: Array3d[np.uint8],
    payload: ArrayBase[tuple[int], np.uint8],
    key: Optional[SupportsEntropy] = None,
):
    if key is None:
        key = DEFAULT_KEY
    header = np.frombuffer(len(payload).to_bytes(4, 'little'), dtype=np.uint8)
    header_bits = np.unpackbits(header)
    assert header_bits.size == 32
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
            if 'iterator too short' in str(e):
                raise SteganographyError('payload too large for image') from e
            raise
        idx = ys[d0], xs[d0], d2
        img[idx] = (img[idx] & 0xFE) | bits
        occupied[idx[:2]] = True
    return img


def extract(
    cover_img: Array3d[np.uint8],
    carrier_img: Array3d[np.uint8],
    key: Optional[SupportsEntropy] = None,
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

    header_idx = get_idx(32)
    header_bytes = np.packbits(carrier_img[header_idx] & 1)
    ignored[header_idx[:2]] = True
    payload_len = int.from_bytes(header_bytes, 'little')
    logger.info("extract payload_bytes=%d", payload_len)
    idx = get_idx(payload_len * 8)
    return np.packbits(carrier_img[idx] & 1)
