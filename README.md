# ChaosEdgeSteg
Chaos-Based Edge Adaptive Steganography Tool

## Introduction
`ChaosEdgeSteg` is a unique steganography tool that leverages the unpredictable nature of chaotic systems, specifically the Henon map, combined with edge detection to embed information within images in a concealed manner.

### Mathematics & Logic of ChaosEdgeSteg

**Chaotic Systems:** At the heart of `ChaosEdgeSteg` is the Henon map, a type of discrete-time dynamical system. Such systems are characterized by their sensitivity to initial conditions, a property commonly referred to as the "butterfly effect". Mathematically, the Henon map is described by the following equations:

$$x_{n+1} = 1 - a x_n^2 + y_n$$

$$y_{n+1} = b x_n$$

Where \(a\) and \(b\) are constants. For `ChaosEdgeSteg`, we utilize typical values of \(a=1.4\) and \(b=0.3\) which produce chaotic behavior.

**Edge Detection:** `ChaosEdgeSteg` identifies edges in the cover image, which are regions with rapid intensity change, using the Canny edge detection method. These edges are less perceptible to the human eye when modified, making them ideal for embedding information.

**Embedding Mechanism:** The chaotic trajectory of the Henon map is mapped onto the detected edge coordinates. This trajectory determines where in the image the payload will be embedded. Due to the chaotic nature of the Henon map, this embedding is resistant to minor changes in the image and difficult to predict without knowledge of the initial conditions (i.e., the key).

## Features

- **Adaptive Embedding:** Embeds information based on image content, ensuring minimal perceptual distortion.
- **Chaos-Based Security:** Utilizes the Henon map to dictate the embedding positions, enhancing security.
- **Edge Prioritization:** Prioritizes edges for embedding, making the embedded information less perceptible.

## Installation

To use `ChaosEdgeSteg`, you can clone the repository:

```shell
git clone https://github.com/crypt0lith/ChaosEdgeSteg.git
cd ChaosEdgeSteg
```

After cloning the repository, install the required dependencies:

```shell
pip install -r requirements.txt
```

## Usage

### Embedding Payload

```bash
python ChaosEdgeSteg.py embed -c <cover_image_path> -f <payload_txt_file> -k 'secret_key' [-o <output_image_path>] [--save_edge_maps]
```

### Extracting Payload

```bash
python ChaosEdgeSteg.py extract -c <cover_image_path_from_embedding> -i <stego_image_path> -k 'key_with_hex_length' [-o <output_text_file>]
```

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy
