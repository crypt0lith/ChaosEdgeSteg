#!/usr/bin/env python
# coding:UTF-8
import argparse
import math
import os
import re
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

import banner

banner.init(autoreset=True)


class SteganographyError(Exception):
    pass


class ChaosEdgeSteg:
    def __init__(self, key, image_path, verbose=False, debug=False):
        self.verbose = verbose
        self.debug = debug
        self.gray_img = None
        self.edge_map = None
        self.key = str(key[6:])
        self.image_path = image_path
        self.payload_length = int(key[0:4], 16)
        self.image = cv2.imread(image_path)
        self.a, self.b = self.generate_henon_parameters_from_key(key)
        self.henon_x, self.henon_y = self.generate_henon_map()
        self.edges = self.detect_edges(self.image)
        self.selected_edge_coordinates = self.map_chaotic_trajectory_to_edges()

    def print_message(self, message, is_debug=False):
        if self.debug or (self.verbose and not is_debug):
            print(message)

    def save_bitmap(self, image_path, image):
        if self.debug:
            cv2.imwrite(image_path, image)

    @staticmethod
    def data_to_bin(data):
        if isinstance(data, str):
            return ''.join(format(ord(i), '08b') for i in data)
        elif isinstance(data, bytes):
            return ''.join(format(byte, '08b') for byte in data)

    @staticmethod
    def bin_to_data(binary):
        return bytes([int(binary[i:i + 8], 2) for i in range(0, len(binary), 8)])

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
        self.print_message("Calculating adaptive thresholds...")

        # Constants
        min_lower_threshold = 85
        base_lower_threshold = 45
        min_upper_threshold = 255
        base_upper_threshold = 135

        # Apply bilateral filtering to reduce noise while preserving edges
        filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # Measure 'information density' of the payload
        normalized_payload_size = self.payload_length / (filtered_image.shape[0] * filtered_image.shape[1])
        self.print_message(f"Normalized Payload Size: {normalized_payload_size}", is_debug=True)

        # Measure 'edginess' of the image
        standard_edge_map = cv2.Canny(filtered_image, base_lower_threshold, base_upper_threshold)
        edge_density = cv2.countNonZero(standard_edge_map) / (filtered_image.shape[0] * filtered_image.shape[1])
        self.print_message(f"Edge density: {edge_density}", is_debug=True)

        # Compute the payload influence by multiplying normalized payload size with a constant factor
        amplify_factor = 100
        payload_influence = normalized_payload_size * amplify_factor
        self.print_message(f"Payload influence: {payload_influence}", is_debug=True)
        if 2 <= payload_influence < 2.5:
            print(
                "Warning: The payload size is approaching the limit for the given cover image. Processing might take "
                "longer than usual.")
        elif payload_influence >= 2.5:
            raise SteganographyError("The payload is too large for the selected cover image.")

        # Compute combined density
        combined_density = edge_density + payload_influence

        # Adjust the thresholds based on combined density
        threshold_scale = 1 - combined_density  # Scale the thresholds towards min values for high combined_density
        lower_threshold = int(base_lower_threshold + threshold_scale * (min_lower_threshold - base_lower_threshold))
        upper_threshold = int(base_upper_threshold + threshold_scale * (min_upper_threshold - base_upper_threshold))
        self.print_message(f"Thresholds after combined density adjustment: {lower_threshold, upper_threshold}",
                           is_debug=True)

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
        x, y = [0.1], [0.1]
        for i in range(self.payload_length * 8):
            next_x = y[-1] + 1 - self.a * x[-1] ** 2
            next_y = self.b * x[-1]
            x.append(next_x)
            y.append(next_y)
        return x, y

    def generate_henon_parameters_from_key(self, key):
        self.print_message("Generating Henon parameters from key...")

        entropy = self.calculate_key_entropy(key)
        self.print_message(f"Key entropy: {entropy}", is_debug=True)

        a = 1.4 - (0.2 * (entropy / 8))
        b = 0.3 + (0.1 * (entropy / 8))
        self.print_message(f"Parameter a: {a}", is_debug=True)
        self.print_message(f"Parameter b: {b}", is_debug=True)
        return a, b

    def map_chaotic_trajectory_to_edges(self):
        # Identify edge coordinates
        self.print_message("Detecting edges...")
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

        # Select edge coordinates and handle collisions
        print("Mapping chaotic trajectory to edge coordinates...")
        available_edge_mask = np.ones(len(edge_coordinates), dtype=bool)
        final_selected_edge_coordinates = []
        for i, index in tqdm(enumerate(normalized_indices[:, 0]), total=len(normalized_indices)):
            original_index = index  # Store the original index to check for full loop without available edge
            while not available_edge_mask[index]:
                index = (index + 1) % len(edge_coordinates)
                if index == original_index:  # We have looped around and no available edge is found
                    raise SteganographyError(
                        "Payload too large for the input image. All available edge coordinates have been exhausted.")
            final_selected_edge_coordinates.append(edge_coordinates[index])
            available_edge_mask[index] = False
        final_selected_edge_coordinates = np.array(final_selected_edge_coordinates)

        # Create a blank canvas to visualize the selected edge coordinates
        selected_edge_map = np.zeros_like(self.gray_img)
        selected_edge_map[final_selected_edge_coordinates[:, 0], final_selected_edge_coordinates[:, 1]] = 255
        self.save_bitmap('selected_edge_map.png', selected_edge_map)

        return final_selected_edge_coordinates

    def embed_payload(self, img_path, payload):
        print("Embedding payload...")
        binary_payload = self.data_to_bin(payload)
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

        print("Extracting payload...")
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
        extracted_payload_bin = ''.join(extracted_bits)
        payload = self.bin_to_data(extracted_payload_bin)
        return payload


def embed_action(args):
    # Check for mutual exclusivity of -p and -f
    if args.payload and args.payload_file:
        print("Error: Cannot use -p and -f simultaneously. Choose one method to provide the payload.")
        return
    elif args.payload:
        payload = args.payload
    elif args.payload_file:
        if not args.payload_file.endswith('.zip'):
            print("Error: Only ZIP archives are allowed with -f argument.")
            return
        with open(args.payload_file, 'rb') as file:
            payload = file.read()
    else:
        print("Error: Either -p or -f must be provided to specify the payload.")
        return

    # Adjust key by appending the hex length of the payload
    hex_length = f"{len(payload):04X}"
    adjusted_key = f"{hex_length}::{args.key}"
    print(f"Key with hex length appended: {adjusted_key}")

    # Instantiate the ChaosEdgeSteg object with the adjusted key and cover image path
    steg = ChaosEdgeSteg(adjusted_key, args.cover_image_path, args.verbose, args.debug)

    # Embed the payload
    stego_image = steg.embed_payload(args.cover_image_path, payload)

    # Determine the output image path
    if args.output_image_path:
        output_image_path = os.path.splitext(args.output_image_path)[0] + '.png'
    else:
        output_image_path = f'stego_{os.path.splitext(os.path.basename(args.cover_image_path))[0]}.png'

    cv2.imwrite(output_image_path, stego_image)
    print(f"Stego image saved to: {output_image_path}")


def extract_action(args):
    if not re.match(r"^[0-9A-Fa-f]+::", args.key):
        print("Invalid key format. Ensure the key has the correct hex length appendment (e.g., 'XXXX::key').")
        return

    # Instantiate the ChaosEdgeSteg object with the provided key and original cover image path
    steg = ChaosEdgeSteg(args.key, args.cover_image_path, args.verbose, args.debug)

    # Extract the payload from the stego image
    extracted_payload = steg.extract_payload(args.stego_image_path)

    # Check for ZIP file signature
    if extracted_payload[:4] == b'PK\x03\x04':
        # It's a ZIP payload
        # Determine the output file path
        if args.output_file:
            output_file_path = args.output_file
        else:
            # If no output file path provided, use a default name and save in the current directory
            output_file_path = 'extracted_payload.zip'
            print(f"Extracted ZIP archive saved to: {output_file_path}")

        # Save the payload as a binary file
        with open(output_file_path, 'wb') as file:
            file.write(extracted_payload)
    else:
        # It's a text payload
        extracted_text = extracted_payload.decode('utf-8', errors='replace')
        print(f"Extracted payload: {extracted_text}")


def main_cli():
    __header__ = banner.h()

    # Display the banner
    print(__header__)
    parser = argparse.ArgumentParser(description='ChaosEdgeSteg: A chaos-based edge adaptive steganography tool.')
    subparsers = parser.add_subparsers()

    # Embed action arguments
    embed_parser = subparsers.add_parser('embed', help='Embed payload into an image.')
    embed_parser.add_argument('-c', '--cover_image_path', required=True, help='Path to the cover image.')
    embed_parser.add_argument('-p', '--payload', type=str,
                              help='String payload to embed directly from the command line.')
    embed_parser.add_argument('-f', '--payload_file', type=str, help='Path to the payload file (ZIP archive).')
    embed_parser.add_argument('-k', '--key', required=True, help='Key to use for embedding.')
    embed_parser.add_argument('-o', '--output_image_path', default=None,
                              help='Path to save the output stego image. If not specified, defaults to '
                                   '"stego_<cover_image_name>".')
    embed_parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output.')
    embed_parser.add_argument('-vv', '--debug', action='store_true', default=False, help='Enable debug output and '
                                                                                         'save edge bitmaps.')
    embed_parser.set_defaults(func=embed_action)

    # Extract action arguments
    extract_parser = subparsers.add_parser('extract', help='Extract payload from a stego image.')
    extract_parser.add_argument('-c', '--cover_image_path', required=True,
                                help='Path to the original cover image used during embedding.')
    extract_parser.add_argument('-i', '--stego_image_path', required=True,
                                help='Path to the stego image from which to extract the payload.')
    extract_parser.add_argument('-k', '--key', required=True,
                                help='Key used during embedding, with the payload length appendment (e.g., '
                                     '"XXXX::key").')
    extract_parser.add_argument('-o', '--output_file', default=None,
                                help='Path to save the extracted message as a text file. If not specified, '
                                     'the message is printed to the console.')
    extract_parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Enable verbose output.')
    extract_parser.add_argument('-vv', '--debug', action='store_true', default=False, help='Enable debug output and '
                                                                                           'save edge bitmaps.')
    extract_parser.set_defaults(func=extract_action)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main_cli()
