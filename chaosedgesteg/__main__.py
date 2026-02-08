import argparse
import datetime
import os
import pathlib
import sys
import tempfile
import zipfile
from typing import BinaryIO
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image

from . import LossyImageError, __name__ as prog, __version__
from .steg import embed, extract


def collect_zipfile_arr(*paths: pathlib.Path | BinaryIO):
    with tempfile.NamedTemporaryFile('w+b') as tmp:
        with zipfile.ZipFile(tmp, 'w') as zf:
            if len(paths) == 1 and not isinstance(paths[0], pathlib.Path):
                zf.comment = b'0'
                [fd] = paths
                with tempfile.NamedTemporaryFile('w+b') as f:
                    while chunk := fd.read(4096):
                        f.write(chunk)
                    f.seek(0)
                    zf.write(f.name, arcname='0.bin')
            else:
                for path in map(pathlib.Path.expanduser, paths):  # noqa
                    if path.is_file():
                        zf.write(path, arcname=path.name)
                    elif path.is_dir():
                        for child in path.rglob('*'):
                            if child.is_dir():
                                continue
                            zf.write(child, arcname=child.relative_to(path.parent))
                    else:
                        from errno import ENOENT

                        raise FileNotFoundError(
                            ENOENT, 'no such file or directory', os.fspath(path)
                        )
        tmp.seek(0)
        arr = np.fromfile(tmp, dtype=np.uint8)
    return arr


def dump_zipfile_arr(arr: np.ndarray[tuple[int], np.dtype[np.uint8]]):
    with tempfile.NamedTemporaryFile('w+b') as tmp:
        arr.tofile(tmp)
        tmp.seek(0)
        content = bytearray()
        is_zipfile = False
        with zipfile.ZipFile(tmp, 'r') as zf:
            if zf.comment == b'0':
                assert zf.namelist() == ['0.bin']
                content.extend(zf.read('0.bin'))
            else:
                is_zipfile = True
        if is_zipfile:
            tmp.seek(0)
            while chunk := tmp.read(4096):
                content.extend(chunk)
    return is_zipfile, bytes(content)


def image_from_uri(uri: str):
    parsed = urlparse(uri)
    scheme = parsed.scheme
    fname = pathlib.Path(parsed.path).name
    with tempfile.NamedTemporaryFile('w+b') as tmp:
        if scheme == 'file':
            path = pathlib.Path.from_uri(uri)
            with path.open('rb') as f:
                while chunk := f.read(8192):
                    tmp.write(chunk)
        elif scheme.startswith('http'):
            with requests.get(uri, stream=True, timeout=10) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
        else:
            raise ValueError(
                f"unsupported uri scheme: {scheme!r}"
                if scheme
                else f"malformed uri: {uri!r}"
            )
        with Image.open(tmp) as im:
            assert_lossless(im)
            im = im.copy()
    return im, fname


def open_image[AnyStr: (str, bytes)](path: AnyStr | os.PathLike[AnyStr] | BinaryIO):
    with Image.open(path) as im:
        assert_lossless(im)
        fname = im.filename
        im = im.copy()
    return im, fname


def assert_lossless(im: Image.Image):
    fmt = (im.format or '').upper()
    if fmt in {'JPEG', 'JPG', 'MPO'}:
        raise LossyImageError(f"{fmt} uses lossy compression")
    elif fmt in {'PNG', 'BMP', 'GIF'}:
        pass
    elif fmt == 'WEBP' and im.info.get('lossless') is not True:
        raise LossyImageError(f"{fmt} does not use lossless compression")
    elif fmt == 'TIFF' and getattr(im, 'tag_v2', {}).get(259) not in {1, 5, 8, 32773}:
        raise LossyImageError(f"{fmt} uses lossy or unknown compression")
    else:
        raise ValueError(f"unsupported format: {fmt!r}")


def handle_cover_image(ns: argparse.Namespace):
    path: str = ns.cover_img_path
    if ns.from_remote:
        im, fname = image_from_uri(path)
    else:
        im, fname = open_image(path)
    if im.mode.lower() == 'rgb':
        arr = np.array(im, dtype=np.uint8)
    else:
        with im.convert('rgb') as rgb:
            arr = np.array(rgb, dtype=np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr, fname


def get_ces_filename(suffix: str = ''):
    fname = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")
    fname += f"_{prog}{suffix}"
    return fname


def handle_password(ns: argparse.Namespace):
    if hasattr(ns, 'password'):
        password: str = ns.password
        return password
    elif hasattr(ns, 'password_file'):
        buf = bytearray()
        password_file: BinaryIO = ns.password_file
        while chunk := password_file.read(4096):
            buf.extend(chunk)
        return bytes(buf)
    return


def handle_embed(ns: argparse.Namespace):
    arr, fname = handle_cover_image(ns)
    suffix = pathlib.Path(fname).suffix
    if hasattr(ns, 'outfile'):
        outfile: pathlib.Path = ns.outfile
        outfile = outfile.expanduser()
        if outfile.is_dir():
            outfile /= pathlib.Path(get_ces_filename(suffix))
        elif outfile.suffix != suffix:
            outfile = outfile.with_suffix(suffix)
    else:
        outfile = pathlib.Path(get_ces_filename(suffix))
    payload = collect_zipfile_arr(*ns.paths)
    steg_arr = embed(arr, payload, key=handle_password(ns))
    cv2.imwrite(outfile, steg_arr)


def handle_extract(ns: argparse.Namespace):
    arr, _ = handle_cover_image(ns)
    steg_im, _ = open_image(ns.steg_img_path.expanduser())
    if steg_im.mode.lower() == 'rgb':
        steg_arr = np.array(steg_im, dtype=np.uint8)
    else:
        with steg_im.convert('rgb') as rgb:
            steg_arr = np.array(rgb, dtype=np.uint8)
    steg_arr = cv2.cvtColor(steg_arr, cv2.COLOR_RGB2BGR)
    payload = extract(arr, steg_arr, key=handle_password(ns))
    is_zipfile, payload_buf = dump_zipfile_arr(payload)
    ext = '.zip' if is_zipfile else '.bin'
    if hasattr(ns, 'outfile'):
        outfile: pathlib.Path | BinaryIO = ns.outfile
        if isinstance(outfile, pathlib.Path):
            outfile = outfile.expanduser()
            if outfile.is_dir():
                outfile /= pathlib.Path(get_ces_filename(ext))
        else:
            outfile.write(payload_buf)
            return
    else:
        outfile = pathlib.Path.cwd() / get_ces_filename(ext)
    outfile.write_bytes(payload_buf)


def main():
    base_parser = argparse.ArgumentParser(add_help=False)

    cover_image_opts = base_parser.add_argument_group(title='cover image options')
    cover_image_opts.add_argument(
        dest='cover_img_path',
        metavar='IMG',
        help='. '.join(
            [
                'path to cover image used for embed/extract',
                'image must use a lossless format (eg., PNG, BMP)',
            ]
        ),
    )
    cover_image_opts.add_argument(
        '-r',
        '--remote',
        dest='from_remote',
        action='store_true',
        help='interpret IMG as a URI to a remote image (default: %(default)s)',
    )

    password_opts = base_parser.add_argument_group(
        title='password options',
        description='specify key to use for Hénon map parameter entropy',
    )
    password_group = password_opts.add_mutually_exclusive_group()
    password_group.add_argument(
        '-p',
        '--password',
        dest='password',
        metavar='PASSWORD',
        help='. '.join(
            [
                'plaintext string',
                'this option is insecure and should be avoided, '
                'as it will be visible in process listings and stuff like that',
                'use --passwd-file instead',
            ]
        ),
        default=argparse.SUPPRESS,
    )
    password_group.add_argument(
        '-P',
        '--passwd-file',
        dest='password_file',
        metavar='FILE',
        type=argparse.FileType('rb'),
        help='path to password file',
        default=argparse.SUPPRESS,
    )

    parser = argparse.ArgumentParser(prog=prog, allow_abbrev=False)
    parser.add_argument(
        '-V', '--version', action='version', version=f'%(prog)s {__version__}'
    )

    cmd_subparsers = parser.add_subparsers(dest='cmd', required=True)

    embed_subparser = cmd_subparsers.add_parser('embed', parents=[base_parser])
    embed_subparser.add_argument(
        dest='paths',
        type=pathlib.Path,
        nargs='*',
        metavar='FILE',
        default=[sys.stdin.buffer],
    )
    embed_subparser.add_argument(
        '-O',
        '--outfile',
        dest='outfile',
        type=pathlib.Path,
        metavar='FILE',
        default=argparse.SUPPRESS,
    )

    extract_subparser = cmd_subparsers.add_parser('extract', parents=[base_parser])
    extract_subparser.add_argument(
        dest='steg_img_path', metavar='STEG_IMG', type=pathlib.Path
    )
    extract_outfile_opts = extract_subparser.add_argument_group(
        title='output options',
        description='. '.join(
            [
                'specify where to write extracted payload',
                'by default, '
                'writes to %r, where <ext> is either %r or %r depending on the payload'
                % (f"%Y%m%d%H%M%S_{prog}.<ext>", 'bin', 'zip'),
            ]
        ),
    )
    extract_outfile_group = extract_outfile_opts.add_mutually_exclusive_group()
    extract_outfile_group.add_argument(
        '--stdout',
        dest='outfile',
        const=sys.stdout.buffer,
        help='. '.join(
            [
                'write directly to stdout',
                'warning: if payload is binary and stdout is a tty, '
                'this will mess up your terminal',
            ]
        ),
        action='store_const',
        default=argparse.SUPPRESS,
    )
    extract_outfile_group.add_argument(
        '-O',
        '--outfile',
        dest='outfile',
        type=pathlib.Path,
        metavar='FILE',
        help='write to the specified file',
        default=argparse.SUPPRESS,
    )

    ns = parser.parse_args()
    if ns.cmd == 'embed':
        return handle_embed(ns)
    elif ns.cmd == 'extract':
        return handle_extract(ns)


if __name__ == '__main__':
    sys.exit(main())
