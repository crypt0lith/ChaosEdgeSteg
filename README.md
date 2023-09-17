# ChaosEdgeSteg

![chaosedgesteg_shellview](https://github.com/crypt0lith/ChaosEdgeSteg/assets/118923461/731142d7-be35-4b16-9eb8-76ff947b3348)

ChaosEdgeSteg is a unique steganography tool that leverages the unpredictable nature of chaotic systems, specifically the Hénon map, combined with edge detection to embed information within images in a concealed manner.

## Introduction


### ChaosEdgeSteg's Steganographic Mechanism


**Chaotic Systems:** The core mechanism of ChaosEdgeSteg revolves around the Hénon map, a type of discrete-time dynamical system. Such systems are characterized by their sensitivity to initial conditions, a property commonly referred to as the "butterfly effect". Mathematically, the Hénon map is described by the following equations:

$$x_{n+1} = 1 - a x_n^2 + y_n$$

$$y_{n+1} = b x_n$$

Where \(a\) and \(b\) are constants. For ChaosEdgeSteg, we utilize typical values of \(a=1.4\) and \(b=0.3\) which produce chaotic behavior.

**Edge Detection:** ChaosEdgeSteg identifies edges in the cover image, which are regions with rapid intensity change, using the Canny edge detection method. These edges are less perceptible to the human eye when modified, making them ideal for embedding information.

**Embedding Mechanism:** The chaotic trajectory of the Hénon map is mapped onto the detected edge coordinates. This trajectory determines where in the image the payload will be embedded. Due to the chaotic nature of the Hénon map, this embedding is highly sensitive to minor changes in the image and nearly impossible to predict without knowledge of the initial conditions (i.e., the key).

## Features

- **Chaos-Based Security:** Utilizes the Hénon map to dictate the embedding positions, enhancing security.

- **Edge Adaptive Embedding:** Prioritizes edges for embedding, making the embedded information less perceptible.

  - **Advanced Stealth:** Adaptively adjusts edge detection thresholds to reflect both image and payload size, ensuring optimum embedding conditions and minimizing risk of detection by modern steganalysis methods.

- **Defense in Depth:** Requires possession of the key, payload length, and original cover image in order to extract steganographic content. Keys are checked against an embedded SHA256 hash as an additional validation mechanism.

- **Payload Execution:** Embedded Python scripts can be executed as a fileless process when extracted. ChaosEdgeSteg spawns a temporary PowerShell instance, base64 encodes the script, and executes it. When the embedded script finishes execution, the spawned temporary shell is removed.

- **Embed a ZIP Archive:** If it's a `.zip` archive, it can be embedded with the `[-f]` option. Usually requires larger cover images.

- **Quiet Mode:** Suppresses dialogue messages, allowing output to be piped to other Unix tools in an obfuscated way.

## Installation

To use ChaosEdgeSteg, you can clone the repository:

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
python chaosedgesteg.py embed [-v/-vv/-q] -c <cover_image_path> -f <payload_file> -k 'secret_key' [-o <output_image_path>]
```

### Extracting Payload

```bash
python chaosedgesteg.py extract [-v/-vv/-q] -c <cover_image_path_from_embedding> -i <stego_image_path> -k '0000::secret_key' [-o <output_file>] [-x]
```
