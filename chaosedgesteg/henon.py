__all__ = ['shannon_entropy', 'henon_indices', 'henon_params']

from collections import Counter

import mpmath as mp
import numpy as np

from ._typing import Array3d, Array3dIndex, ArrayIndices, SupportsEntropy

mp.mp.dps = 200
K = 48
S = 1 << K
MASK64 = (1 << 64) - 1


def _splitmix64(x: int, /) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & MASK64
    return (x ^ (x >> 31)) & MASK64


def shannon_entropy(seq: SupportsEntropy, /) -> mp.mpf:
    counts = Counter(seq)
    n = mp.mpf(len(seq))
    ln2 = mp.log(2)
    h = mp.mpf('0')
    for c in counts.values():
        p = mp.mpf(c) / n
        h -= p * (mp.log(p) / ln2)
    return h


def _mp_to_fixed(x: mp.mpf, /) -> int:
    return int(mp.nint(x * S))


X0 = _mp_to_fixed(mp.mpf('0.123456789123'))
Y0 = _mp_to_fixed(mp.mpf('0.362436069531'))


def henon_params(key: SupportsEntropy) -> tuple[int, int]:
    ent = shannon_entropy(key)
    a = (mp.mpf('56') - ent) / mp.mpf('40')
    b = (mp.mpf('24') + ent) / mp.mpf('80')
    return _mp_to_fixed(a), _mp_to_fixed(b)


def henon_indices(arr: Array3d, key: SupportsEntropy, count: int):
    if count < 0:
        raise ValueError("expected count to be non-negative number")

    def generate():
        a, b = henon_params(key)
        x, y = X0, Y0
        n = count
        max_steps = count * 50
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

    indices: ArrayIndices = np.fromiter(generate(), dtype=np.int64, count=count)
    d0, d1, d2 = _i_to_yxz(indices, *arr.shape[:2])
    return d0, d1, d2


def _i_to_yxz[_Dim = int](
    indices: ArrayIndices[_Dim], h: int, w: int
) -> Array3dIndex[_Dim]:
    plane = h * w
    z = indices // plane
    rem = indices - z * plane
    x = rem // h
    y = rem - x * h
    return y, x, z
