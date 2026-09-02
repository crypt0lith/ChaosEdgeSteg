#!/usr/bin/env python3
import collections.abc as abc
import datetime
import enum
import functools as ft
import os
import signal
import sys
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

from . import *
from . import __name__ as prog, _attest_log, logger


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


def open_image[AnyStr: (str, bytes)](path: AnyStr | os.PathLike[str] | BinaryIO):
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
    fname = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")
    fname += f"_{__package__}{suffix}"
    return fname


def handle_password(ns):
    if hasattr(ns, "password"):
        password: bytes = ns.password
        _attest_log(logger.debug, "using password from argv")
        return password
    elif hasattr(ns, "password_file"):
        buf = bytearray()
        password_file: BinaryIO = ns.password_file
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


def handle_extract(ns) -> Optional[Path]:
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
    debug_output: Optional[Path] = ns.debug_output
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
        from .banner import BANNER

        print(BANNER, file=sys.stderr)
    return log_to_stderr


def parse_args():
    import argparse

    from . import __version__

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

    parser = argparse.ArgumentParser(prog=prog, allow_abbrev=False)
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )

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
