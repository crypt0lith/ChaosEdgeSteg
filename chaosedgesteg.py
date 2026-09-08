#!/usr/bin/env python3
__all__ = [
    "Header",
    "LossyImageError",
    "Payload",
    "PayloadKind",
    "SteganographyError",
    "adaptive_canny",
    "dump_pyfile",
    "embed",
    "extract",
    "indices_3d",
    "loads_pycode",
    "loads_pyfile",
    "make_zipfile_arr",
]
import ast
import collections.abc as abc
import datetime
import enum
import functools as ft
import logging
import os
import signal
import struct
import sys
import typing as tp
import zipfile
from hashlib import blake2b
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import CodeType
from urllib.parse import urlparse

import cv2
import mpmath as mp
import numpy as np
from PIL import Image

BANNER = """\
\x1b[0;31m
\x1b[23C.    .\x1b[13C.\x1b[13C...
\x1b[13C,.....    ;     .\x1b[11C...\x1b[10C..;;;..   .....
\x1b[10C.;i         .s     .s.        ..;..       ,ji&OOoq;.   ......
        .jil\x1b[10CSS      s$.       .;;;.       g!'     %S.   ;;;;;;;.
      .ll$l\x1b[10C.$S      .s$.     .;;$;;.     ;$        $s  ;b@$$SSijc.
    ..7l$l\x1b[11C.$S       ;$S     .;$@$;.    .s$        S$. ;K       `s;
   ..77$S\x1b[11C.;$*       ;$$    .;$$*$$;.   .S$        S$; :B         '
 ..ps$$$S\x1b[11C.;$l       i$$    .$$* *$$.   ;S$        S$; ;;$
 ;sXSf$$s\x1b[11C.s$$      .$$$   .;$*   *$;.  ;S$        S$;  ;;$
 si`7f$$s\x1b[11CsS$$*...+*$$$$  .;$*******$;. .S$        S$.   ;;$.
 F  .>$$s\x1b[10CjyXF====##@@$$$  .S$##333##$S.  ;$s       S$;*.  ;*$    .
    .$$$S        ;iZ ?&ttfPPPPQ$$$ ..$$       $$.. .lSs      S$.zS. .!$$   .
    .s$$$s     .sZZ  ???       *$s .S$         $S.   $$S,   ;S$ sHl:.;x$$  :
    .ss$$Ss...;d$Z   ?li       *$. .S\x1b[11CS.    ;$@@sS$$; 'S$$$XX#$$.!
    ..ss@@@#GS$Z/    ;ll      .S.  sS\x1b[11CSs     .SSSSS.     \\$$$@@$;S.
     .ssS&&#$ff/     .;l      ;s   s.\x1b[12C;      s;  s       .\\X$$$Ss;
      ssSSSXF/`       ;!      s    ;.\x1b[12C;      ;   ;\x1b[10CxXSsZ,
      .sSS27.\x1b[10C;      *    ;\x1b[13C;      ;   ;\x1b[10C.l5Zz.
      .sSS4\x1b[12C;      ;    ;\x1b[13C;      ;   ;\x1b[11C.xXx;.
     ..sSi;\x1b[12C.      .    ;\x1b[13C;      ;   ;\x1b[11C.\\VS;.
     ..SSl\x1b[13C.      .    .\x1b[13C.      ;   .\x1b[12C.SS;.
     .;Sl;\x1b[13C.      .    .\x1b[13C.      .   .\x1b[12C.IS;.
    ..Sl..........\x1b[48C.........S;.
    .;Sl;\x1b[65C.lS.
    .SSl\x1b[67Cll.
    .Sl.\x1b[14C/ __/   / /\x1b[11C/ __// /\x1b[23C.l.
    .Si...........   / _/ / _  // _ `// -_) \\ \\ / __// -_)/ _ `/  ..........l.
    .l1\x1b[13C/___/ \\_,_/ \\_, / \\__//___/ \\__/ \\__/ \\_, /\x1b[12C.l.
    .l;\x1b[24C/___/\x1b[21C/___/\x1b[13C.i.
    .l;\x1b[68C.i.
     ;.\x1b[68C.i
     ..```````````\x1b[48C`````````..
     .\x1b[13CChaos-Based Edge Adaptive Steganography Tool\x1b[13C.

\x1b[18C\x1b[12A\x1b[1;37m................................................
\x1b[18C`##############################################`
\x1b[18C`##############################################`
\x1b[18C`###\x1b[5C###\x1b[3C###########\x1b[8C#############`
\x1b[18C`##\x1b[43C#`
\x1b[18C`#\x1b[43C##`
\x1b[18C`############\x1b[5C#####################\x1b[5C###`
\x1b[18C`##############################################`
\x1b[18C`##############################################`
\x1b[18C````````````````````````````````````\x1b[10C``\x1b[12D\x1b[30mcrypt0lith
\x1b[0m
"""

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


@ft.lru_cache(maxsize=0x10)
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
HEADER_NONCE_SIZE = blake2b.SALT_SIZE
Header = struct.Struct(f"<{len(MAGIC)}s{HEADER_NONCE_SIZE}sBI")

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
    wx, wy, wz = struct.unpack("<3Q", h)
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
    target=1.0,
    *,
    lo=(45, 85),
    hi=(135, 255),
    niter=10,
    tol: tp.Optional[float] = None,
) -> GrayscaleArray:
    arr = np.asarray(arr, dtype=np.uint8)
    target = min(max(target, 0.0), 1.0)
    bounds = np.asarray([lo, hi], dtype=np.uint8).astype(np.float32)
    if tol is None:
        tol = 1.0 / arr.size
    _attest_log(logger.debug, "target=%.6f size=%d", target, arr.size)
    filtered = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
    t_lo, t_hi = 0.0, 1.0
    best_err = best_edges = None
    prev_lower = prev_upper = None
    reason = None
    for _ in range(niter):
        t = (t_lo + t_hi) / 2
        lower, upper = bounds[:, 0] + t * (bounds[:, 1] - bounds[:, 0])
        if lower > upper:
            lower, upper = upper, lower
        if best_edges is not None and (lower, upper) == (prev_lower, prev_upper):
            reason = "thresholds stable"
            break
        edges = cv2.Canny(filtered, lower, upper)
        density = cv2.countNonZero(edges) / arr.size
        err = abs(density - target)
        if best_err is None or err < best_err:
            best_err, best_edges = err, edges
        prev_lower, prev_upper = lower, upper
        if err <= tol:
            reason = "tolerance met"
            break
        if density > target:
            t_lo = t
        else:
            t_hi = t
    _attest_log(
        logger.debug,
        "result lower=%d upper=%d err=%.6f reason=%r",
        prev_lower,
        prev_upper,
        best_err if best_err is not None else -1.0,
        reason or "max iter reached",
    )
    assert best_edges is not None
    return best_edges


def _whiten(size: int, key: abc.Buffer, /, **kwargs):
    seed = np.frombuffer(blake2b(key, **kwargs).digest(), dtype=np.uint8)
    # we only whiten to prevent magic bytes from being used as a plaintext
    # oracle. 'repeating-key xor' is inert because linear ordering does not
    # survive downstream chaotic permutation. treat as not exploitable
    # until proven otherwise.
    return np.resize(seed, size)


class PayloadKind(enum.IntEnum):
    RAW = enum.auto()
    PYCODE = enum.auto()
    PYFILE = enum.auto()
    ZIPFILE = enum.auto()


class Payload(tp.NamedTuple):
    data: ShapedNDArray[tuple[int], np.uint8]
    kind: PayloadKind


def embed(
    img: Array3d[np.uint8],
    payload: ShapedNDArray[tuple[int], np.uint8],
    kind: int,
    key: tp.Optional[abc.Buffer] = None,
):
    if payload.size > img.size:
        raise SteganographyError("payload larger than cover image")
    if key is None:
        key = DEFAULT_KEY
    nonce = os.urandom(HEADER_NONCE_SIZE)
    header = np.frombuffer(
        Header.pack(MAGIC, nonce, PayloadKind(kind), len(payload)), dtype=np.uint8
    )
    header = header ^ _whiten(header.size, key, person=PERSON[0])
    payload = payload ^ _whiten(payload.size, key, salt=nonce, person=PERSON[1])
    _attest_log(logger.info, "payload_bytes=%d", payload.size)
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    occupied = np.zeros(gray.shape, dtype=bool)
    for i, bits in enumerate(map(np.unpackbits, [header, payload])):
        count = bits.size
        target = 1.0
        if i == 0:
            target -= count / gray.size
        edges = adaptive_canny(gray, target) & ~occupied
        if logger.isEnabledFor(logging.DEBUG):
            _attest_log(logger.debug, "edges_nonzero=%d", cv2.countNonZero(edges))
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
        target = 1.0
        if not ignored.any():
            target -= count / gray.size
        edges = adaptive_canny(gray, target) & ~ignored
        if logger.isEnabledFor(logging.DEBUG):
            _attest_log(logger.debug, "edges_nonzero=%d", cv2.countNonZero(edges))
        ys, xs = np.nonzero(edges)
        domain = np.empty((ys.size, 1, 3), dtype=np.uint8)
        d0, _, d2 = indices_3d(domain, key, count)
        return ys[d0], xs[d0], d2

    header_idx = get_idx(Header.size * 8)
    header = np.packbits(carrier_img[header_idx] & 1)
    header ^= _whiten(header.size, key, person=PERSON[0])
    magic, nonce, kind, payload_len = Header.unpack(header.tobytes())
    if magic != MAGIC:
        raise ValueError("bad password")
    _attest_log(logger.info, "payload_bytes=%d", payload_len)
    ignored[header_idx[:2]] = True
    payload_idx = get_idx(payload_len * 8)
    payload = np.packbits(carrier_img[payload_idx] & 1)
    payload ^= _whiten(payload.size, key, salt=nonce, person=PERSON[1])
    return Payload(payload, PayloadKind(kind))


def _get_pycode_info(metadata: bytes):
    magic, bit_field, word = struct.unpack("<4sIQ", metadata)
    out = {"magic": magic, "bit_field": bit_field}
    if not (bit_field & 1):
        import datetime

        out["mtime"] = datetime.datetime.fromtimestamp(
            word & 0xFFFFFFFF, datetime.timezone.utc
        )
        out["src_size"] = word >> 32
    else:
        out["hash"] = word
    return out


def loads_pycode(buf: bytes) -> CodeType:
    import importlib.util
    import marshal

    info = _get_pycode_info(buf[:16])
    if info["magic"] != importlib.util.MAGIC_NUMBER:
        magic = int.from_bytes(info["magic"][:2], "little")
        raise ValueError(f"compiled on incompatible python version (magic: {magic})")
    return marshal.loads(buf[16:], allow_code=True)


def loads_pyfile(buf: bytes) -> CodeType:
    node = ast.parse(buf, mode="exec")
    node = ast.fix_missing_locations(node)
    return compile(node, filename="<unknown>", mode="exec")


def dump_pyfile(file: tp.BinaryIO) -> bytes:
    node = ast.parse(file.read(), mode="exec")
    node = ast.fix_missing_locations(node)
    return ast.unparse(node).encode()


def make_zipfile_arr(*paths: Path):
    with NamedTemporaryFile("w+b") as tmp:
        with zipfile.ZipFile(tmp, "w") as zf:
            for path in paths:
                if path.is_file():
                    zf.write(path, arcname=path.name)
                elif path.is_dir():
                    for child in path.rglob("*"):
                        if child.is_dir():
                            continue
                        zf.write(child, arcname=child.relative_to(path.parent))
                else:
                    from errno import ENOENT

                    raise FileNotFoundError(
                        ENOENT, "no such file or directory", os.fspath(path)
                    )
        tmp.seek(0)
        arr = np.fromfile(tmp, dtype=np.uint8)
    _attest_log(logger.debug, "payload size=%d", arr.size)
    return arr


def image_from_uri(uri: str):
    parsed = urlparse(uri)
    scheme = parsed.scheme
    fname = Path(parsed.path).name
    _attest_log(logger.info, "scheme=%s", scheme or "<none>")
    with NamedTemporaryFile("w+b") as tmp:
        if scheme == "file":
            path = Path.from_uri(uri)
            with path.open("rb") as f:
                while chunk := f.read(8192):
                    tmp.write(chunk)
        elif scheme.startswith("http") or not scheme:
            import requests

            with requests.get(uri, stream=True, timeout=10) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
        else:
            raise ValueError(f"unsupported uri scheme: {scheme!r}")
        with Image.open(tmp) as im:
            _attest_log(
                logger.debug, "format=%s mode=%s size=%s", im.format, im.mode, im.size
            )
            assert_lossless(im)
            im = im.copy()
    return im, fname


def open_image(path: str | os.PathLike[str] | tp.BinaryIO):
    with Image.open(path) as im:
        _attest_log(
            logger.debug, "format=%s mode=%s size=%s", im.format, im.mode, im.size
        )
        assert_lossless(im)
        fname = im.filename
        im = im.copy()
    return im, fname


def assert_lossless(im: Image.Image):
    fmt = (im.format or "").upper()
    _attest_log(logger.debug, "format=%s", fmt or "<none>")
    match fmt:
        case "PNG" | "BMP" | "GIF":
            return
        case "JPEG" | "JPG" | "MPO":
            raise LossyImageError(f"{fmt} uses lossy compression")
        case "WEBP":
            if im.info.get("lossless") is not True:
                raise LossyImageError(f"{fmt} does not use lossless compression")
        case "TIFF":
            if getattr(im, "tag_v2", {}).get(259) not in {1, 5, 8, 32773}:
                raise LossyImageError(f"{fmt} uses lossy or unknown compression")
        case _:
            raise ValueError(f"unsupported format: {fmt!r}")


def handle_cover_image(ns):
    path: str = ns.cover_img_path
    if ns.from_remote:
        handler, origin = image_from_uri, "remote"
    else:
        handler, origin = open_image, "local"
    _attest_log(logger.info, "[%s]\t%s", origin, path)
    im, fname = handler(path)
    with im.convert("RGB") as rgb:
        arr = np.array(rgb, dtype=np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _attest_log(logger.debug, "shape=%s dtype=%s", arr.shape, arr.dtype)
    return arr, fname


def get_ces_filename(suffix: str = ""):
    now = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")
    return f"{now}_{__name__}{suffix}"


def handle_password(ns):
    if hasattr(ns, "password"):
        password: bytes = ns.password
        _attest_log(logger.debug, "using password from argv")
        return password
    elif hasattr(ns, "password_file"):
        buf = bytearray()
        password_file: tp.BinaryIO = ns.password_file
        while chunk := password_file.read(4096):
            buf.extend(chunk)
        _attest_log(logger.debug, "using password from file")
        return bytes(buf)
    elif from_env := os.environ.get("CESPASSWORD"):
        _attest_log(logger.debug, "using password from env")
        return from_env.encode()
    else:
        _attest_log(logger.debug, "no password provided")


def handle_embed(ns):
    arr, fname = handle_cover_image(ns)
    assert isinstance(fname, str)
    suffix = Path(fname).suffix
    if hasattr(ns, "outfile"):
        outfile: Path = ns.outfile
        if outfile.is_dir():
            outfile /= Path(get_ces_filename(suffix))
        elif outfile.suffix != suffix:
            outfile = outfile.with_suffix(suffix)
    else:
        outfile = Path(get_ces_filename(suffix))
    if hasattr(ns, "from_raw"):
        kind = PayloadKind.RAW
        if ns.from_raw.seekable():
            payload = np.fromfile(ns.from_raw, dtype=np.uint8)
        else:
            payload = np.frombuffer(ns.from_raw.read(), dtype=np.uint8)
    elif hasattr(ns, "from_pycode"):
        kind = PayloadKind.PYCODE
        if ns.from_pycode.seekable():
            payload = np.fromfile(ns.from_pycode, dtype=np.uint8)
        else:
            payload = np.frombuffer(ns.from_pycode.read(), dtype=np.uint8)
    elif hasattr(ns, "from_pyfile"):
        kind = PayloadKind.PYFILE
        payload = np.frombuffer(dump_pyfile(ns.from_pyfile), dtype=np.uint8)
    elif hasattr(ns, "from_files"):
        kind = PayloadKind.ZIPFILE
        payload = make_zipfile_arr(*ns.from_files)
    else:
        raise RuntimeError("unreachable")
    _attest_log(logger.info, "size=%d outfile=%s", payload.size, outfile)
    steg_arr = embed(arr, payload, kind, key=handle_password(ns))
    cv2.imwrite(outfile, steg_arr)
    return outfile


def _zipinfo(infos: abc.Sequence[zipfile.ZipInfo]):
    import stat

    HOSTS = (
        "fat", "ami", "vms", "unx", "cms", "atr", "hpf", "mac", "zzz", "cpm",
        "ntf", "mvs", "vse", "acn", "vft", "ats", "bos", "tan", "440", "osx",
    )   # fmt: skip
    out = []
    size_u = size_c = 0
    for info in infos:
        if mode := info.external_attr >> 16:
            perms = stat.filemode(mode)
        else:
            dos = info.external_attr & 0xFF
            perms = ("d" if dos & 0x10 else "-") + (
                ("r" + ("-" if dos & 0x01 else "w") + "-") * 3
            )
        ver = "%d.%d" % (info.create_version // 10, info.create_version % 10)
        host = HOSTS[info.create_system] if info.create_system < len(HOSTS) else "???"
        attrs = "bt"[info.internal_attr & 1] + "-x"[bool(info.extra)]
        m = info.compress_type
        if m == zipfile.ZIP_DEFLATED:
            b1, b2 = ((info.flag_bits >> i) & 1 for i in [1, 2])
            meth = "def" + ["NX", "FS"][b2][b1]
        else:
            meth = {
                zipfile.ZIP_STORED: "stor",
                zipfile.ZIP_BZIP2: "bzp2",
                zipfile.ZIP_LZMA: "lzma",
            }.get(m, "u%03d" % m)
        if info.flag_bits:
            meth = meth[0].upper() + meth[1:]
        date = datetime.datetime(*info.date_time).strftime("%y-%b-%d %H:%M")
        line = "%-10s  %3s %s %8d %s %s %s %s"
        line %= perms, ver, host, info.file_size, attrs, meth, date, info.filename
        out.append(line)
        size_u += info.file_size
        size_c += info.compress_size
    ratio = 0.0 if size_u == 0 else (size_u - size_c) / size_u * 100
    out.append(
        f"{len(infos)} files, "
        f"{size_u} bytes uncompressed, "
        f"{size_c} bytes compressed: "
        f"{ratio:.1%}"
    )
    return out


class _ExtractFlag(enum.IntFlag):
    YES = enum.auto()
    EXEC = enum.auto()
    INSPECT = enum.auto()


def handle_extract(ns) -> tp.Optional[Path]:
    arr, _ = handle_cover_image(ns)
    steg_im, fname = open_image(ns.steg_img_path)
    flags = ft.reduce(lambda i, j: i | j, ns._flags, 0)

    def prompt(msg: str, /):
        if flags & _ExtractFlag.YES:
            return True
        while True:
            answer = input(f"{msg}? (y/N) ").strip().casefold()
            try:
                if answer in {"y", "yes"}:
                    return True
                if answer in {"n", "no"}:
                    return False
            finally:
                print("\x1b[A\r\x1b[2K", end="")
            print(
                "invalid answer%s." % ("" if len(answer) > 10 else f" {answer!r}"),
                "please enter 'yes' or 'no'",
                file=sys.stderr,
            )

    with steg_im.convert("RGB") as rgb:
        steg_arr = np.array(rgb, dtype=np.uint8)
        steg_arr = cv2.cvtColor(steg_arr, cv2.COLOR_RGB2BGR)
    payload = extract(arr, steg_arr, key=handle_password(ns))
    payload_buf = payload.data.tobytes()
    ask_exec = None
    match payload.kind:
        case PayloadKind.RAW:
            ext = ".bin"
        case PayloadKind.PYCODE:
            ext = ".pyc"
            ask_exec = loads_pycode
        case PayloadKind.PYFILE:
            ext = ".py"
            ask_exec = loads_pyfile
        case PayloadKind.ZIPFILE:
            ext = ".zip"
    if flags & _ExtractFlag.INSPECT:
        if payload.kind == PayloadKind.ZIPFILE:
            from io import BytesIO

            file = BytesIO(payload_buf)
            with zipfile.ZipFile(file) as zf:
                infos = zf.infolist()
                lines = [
                    f"Zip file size: {payload.data.size} bytes, "
                    f"number of entries: {len(infos)}"
                ]
                lines.extend(_zipinfo(infos))
            print(*lines, sep="\n")
        elif payload.kind == PayloadKind.RAW:
            import charset_normalizer

            guess = charset_normalizer.from_bytes(payload_buf).best()
            print(
                "Raw file",
                (f"{guess.encoding} text" if guess else "binary content"),
                sep=", ",
            )
        else:
            print(
                "Python",
                ("source code" if payload.kind == PayloadKind.PYFILE else "bytecode"),
            )
        if not prompt("Proceed"):
            return
    if ask_exec is not None and (
        (flags & _ExtractFlag.EXEC) or prompt("Execute embedded python code")
    ):
        return exec(ask_exec(payload_buf), {})
    if isinstance(fname, (bytes, bytearray)):
        fname = fname.decode()
    if hasattr(ns, "outfile"):
        outfile = ns.outfile
        if not isinstance(outfile, Path):
            outfile.write(payload_buf)
            _attest_log(logger.info, "bytes=%d; extracted to stdout", len(payload_buf))
            return
    else:
        outfile = Path.cwd()
    if outfile.is_dir():
        if (from_fname := outfile / (Path(fname).stem + ext)).exists():
            outfile /= get_ces_filename(ext)
        else:
            outfile = from_fname
    count = outfile.write_bytes(payload_buf)
    _attest_log(logger.info, "bytes=%d; extracted to %s", count, outfile)
    return outfile


def handle_base(ns):
    import logging

    levels = logging.WARNING, logging.INFO, logging.DEBUG
    verbosity = levels[min(ns.verbosity, len(levels) - 1)]

    class _PrefixFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            prefix = "[-]" if record.levelno >= logging.WARNING else "[*]"
            return f"{prefix} {record.getMessage()}"

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    handler = None
    debug_output: tp.Optional[Path] = ns.debug_output
    if debug_output is not None:
        handler = logging.FileHandler(debug_output, mode="a", encoding="utf-8")
    elif not ns.quiet:
        handler = logging.StreamHandler(sys.stderr)
    if handler is not None:
        handler.setLevel(verbosity)
        handler.setFormatter(_PrefixFormatter())
        logger.addHandler(handler)
    log_to_stderr = any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
        for h in logger.handlers
    )
    if not (ns.quiet or ns.no_banner):
        print(BANNER, file=sys.stderr)
    return log_to_stderr


def parse_args():
    import argparse

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="count",
        default=0,
        help="increase verbosity level",
    )
    base_parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        help="suppress stderr output",
    )
    base_parser.add_argument(
        "--no-banner",
        dest="no_banner",
        action="store_true",
        help="suppress banner output",
    )
    base_parser.add_argument(
        "-o",
        "--debug-output",
        dest="debug_output",
        type=Path,
        metavar="FILE",
        help="write logs to %(metavar)s",
    )

    cover_image_opts = base_parser.add_argument_group(title="cover image options")
    cover_image_opts.add_argument(
        dest="cover_img_path",
        metavar="IMG",
        help="""\
        path to cover image used for embed/extract.
        image must use a lossless format (eg., PNG, BMP)""",
    )
    cover_image_opts.add_argument(
        "-r",
        "--remote",
        dest="from_remote",
        action="store_true",
        help="interpret IMG as a URI to a remote image (default: %(default)s)",
    )

    password_opts = base_parser.add_argument_group(
        title="password options",
        description="specify key to use for chaotic coordinate mapping",
    )
    password_group = password_opts.add_mutually_exclusive_group()
    password_group.add_argument(
        "-p",
        "--password",
        dest="password",
        metavar="PASSWORD",
        type=str.encode,
        default=argparse.SUPPRESS,
        help="""\
        password string.
        this option is insecure and should be avoided,
        as it will be visible in process listings and stuff like that.
        use '--passwd-file' instead""",
    )
    password_group.add_argument(
        "-P",
        "--passwd-file",
        dest="password_file",
        metavar="FILE",
        type=argparse.FileType("rb"),
        default=argparse.SUPPRESS,
        help="read password from %(metavar)s",
    )

    parser = argparse.ArgumentParser(prog=__name__, allow_abbrev=False)

    cmd_subparsers = parser.add_subparsers(dest="cmd", required=True)

    embed_subparser = cmd_subparsers.add_parser("embed", parents=[base_parser])

    infile_group = embed_subparser.add_mutually_exclusive_group(required=True)
    infile_group.add_argument(
        "--raw",
        dest="from_raw",
        type=argparse.FileType("rb"),
        metavar="FILE",
        default=argparse.SUPPRESS,
    )
    infile_group.add_argument(
        "--py",
        dest="from_pyfile",
        type=argparse.FileType("rb"),
        metavar="PYFILE",
        default=argparse.SUPPRESS,
    )
    infile_group.add_argument(
        "--pyc",
        dest="from_pycode",
        type=argparse.FileType("rb"),
        metavar="PYCODE",
        default=argparse.SUPPRESS,
    )
    infile_group.add_argument(
        dest="from_files",
        type=Path,
        nargs="*",
        metavar="FILE",
        default=argparse.SUPPRESS,
    )

    embed_subparser.add_argument(
        "-O",
        "--outfile",
        dest="outfile",
        type=Path,
        metavar="FILE",
        default=argparse.SUPPRESS,
        help="write stego image to %(metavar)s",
    )

    extract_subparser = cmd_subparsers.add_parser("extract", parents=[base_parser])
    extract_subparser.set_defaults(_flags=[])
    extract_subparser.add_argument(dest="steg_img_path", metavar="STEG_IMG", type=Path)

    extract_subparser.add_argument(
        "-y", "--yes", dest="_flags", action="append_const", const=_ExtractFlag.YES
    )
    extract_subparser.add_argument(
        "--exec", dest="_flags", action="append_const", const=_ExtractFlag.EXEC
    )
    extract_subparser.add_argument(
        "--inspect", dest="_flags", action="append_const", const=_ExtractFlag.INSPECT
    )

    output_opts = extract_subparser.add_argument_group(
        title="output options",
        description="specify what to do with the extracted payload",
    )
    output_group = output_opts.add_mutually_exclusive_group()
    output_group.add_argument(
        "--stdout",
        dest="outfile",
        action="store_const",
        const=sys.stdout.buffer,
        default=argparse.SUPPRESS,
        help="""\
        write directly to stdout.
        warning: if payload is binary and stdout is a tty,
        this will mess up your terminal""",
    )
    output_group.add_argument(
        "-O",
        "--outfile",
        dest="outfile",
        type=lambda s: sys.stdout.buffer if s == "-" else Path(s),
        metavar="FILE",
        default=argparse.SUPPRESS,
        help="write extracted payload to %(metavar)s",
    )
    return parser.parse_args()


def main():
    ns = parse_args()
    log_to_stderr = handle_base(ns)
    match ns.cmd:
        case "embed":
            handler, target = handle_embed, "stego image"
        case "extract":
            handler, target = handle_extract, "payload"
        case _:
            raise RuntimeError("unreachable")
    try:
        outfile = handler(ns)
    except Exception:
        if not log_to_stderr:
            _attest_log(logger.exception, "error while handling %r", ns.cmd)
        raise
    except KeyboardInterrupt:
        if not ns.quiet:
            print("\nexiting...", file=sys.stderr)
        return 128 + signal.SIGINT
    else:
        if not (outfile is None or ns.quiet):
            print("[\x1b[32m*\x1b[0m]", f"{target} saved to {outfile}", file=sys.stderr)


if __name__ == "__main__":
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
