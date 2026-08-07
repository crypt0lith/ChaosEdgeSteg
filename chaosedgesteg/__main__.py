#!/usr/bin/env python3
import datetime
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Optional
from urllib.parse import urlparse
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from . import LossyImageError, __name__ as prog, _attest_log, embed, extract, logger


def collect_zipfile_arr[_T: (Path, BinaryIO)](*paths: _T):
    with NamedTemporaryFile("w+b") as tmp:
        with ZipFile(tmp, "w") as zf:
            if len(paths) == 1 and not isinstance((fd := paths[0]), Path):
                zf.comment = b"0"
                with zf.open("0.bin", "w") as f:
                    while chunk := fd.read(4096):
                        f.write(chunk)
            else:
                for path in paths:
                    assert isinstance(path, Path)
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
    _attest_log(logger.debug, "payload size=%d", int(arr.size))
    return arr


def dump_zipfile_arr(arr: np.ndarray[tuple[int], np.dtype[np.uint8]]):
    with NamedTemporaryFile("w+b") as tmp:
        arr.tofile(tmp)
        tmp.seek(0)
        content = bytearray()
        is_zipfile = False
        with ZipFile(tmp, "r") as zf:
            if zf.comment == b"0":
                assert zf.namelist() == ["0.bin"]
                content.extend(zf.read("0.bin"))
            else:
                is_zipfile = True
        if is_zipfile:
            tmp.seek(0)
            while chunk := tmp.read(4096):
                content.extend(chunk)
    _attest_log(logger.debug, "payload size=%d is_zipfile=%s", len(content), is_zipfile)
    return is_zipfile, bytes(content)


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
    suffix = Path(fname).suffix
    if hasattr(ns, "outfile"):
        outfile: Path = ns.outfile
        if outfile.is_dir():
            outfile /= Path(get_ces_filename(suffix))
        elif outfile.suffix != suffix:
            outfile = outfile.with_suffix(suffix)
    else:
        outfile = Path(get_ces_filename(suffix))
    payload = collect_zipfile_arr(*ns.paths)
    _attest_log(logger.info, "size=%d outfile=%s", int(payload.size), outfile)
    steg_arr = embed(arr, payload, key=handle_password(ns))
    cv2.imwrite(outfile, steg_arr)
    return outfile


def handle_extract(ns) -> Optional[Path]:
    arr, _ = handle_cover_image(ns)
    steg_im, _ = open_image(ns.steg_img_path)
    with steg_im.convert("RGB") as rgb:
        steg_arr = np.array(rgb, dtype=np.uint8)
        steg_arr = cv2.cvtColor(steg_arr, cv2.COLOR_RGB2BGR)
    payload = extract(arr, steg_arr, key=handle_password(ns))
    is_zipfile, payload_buf = dump_zipfile_arr(payload)
    ext = ".zip" if is_zipfile else ".bin"
    if hasattr(ns, "outfile"):
        outfile: Path | BinaryIO = ns.outfile
        if isinstance(outfile, Path):
            if outfile.is_dir():
                outfile /= Path(get_ces_filename(ext))
        else:
            outfile.write(payload_buf)
            _attest_log(logger.info, "bytes=%d; extracted to stdout", len(payload_buf))
            return
    else:
        outfile = Path.cwd() / get_ces_filename(ext)
    outfile.write_bytes(payload_buf)
    _attest_log(logger.info, "bytes=%d; extracted to %s", len(payload_buf), outfile)
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
    embed_subparser.add_argument(
        dest="paths", type=Path, nargs="*", metavar="FILE", default=[sys.stdin.buffer]
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
    extract_subparser.add_argument(dest="steg_img_path", metavar="STEG_IMG", type=Path)

    extract_outfile_opts = extract_subparser.add_argument_group(
        title="output options",
        description="""\
        specify where to write extracted payload.
        by default, writes to %r,
        where <ext> is either %r or %r depending on the payload"""
        % (f"YYYYMMDDHHMMSS_{__package__}.<ext>", "bin", "zip"),
    )
    extract_outfile_group = extract_outfile_opts.add_mutually_exclusive_group()
    extract_outfile_group.add_argument(
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
    extract_outfile_group.add_argument(
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
    else:
        if not (outfile is None or ns.quiet):
            print("[\x1b[32m*\x1b[0m]", f"{target} saved to {outfile}", file=sys.stderr)


if __name__ == "__main__":
    import signal

    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
