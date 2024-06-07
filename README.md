ChaosEdgeSteg
---  

![chaosedgesteg_shellview](https://github.com/crypt0lith/ChaosEdgeSteg/assets/118923461/731142d7-be35-4b16-9eb8-76ff947b3348)

`ChaosEdgeSteg` is a unique steganography tool that leverages the unpredictable nature of chaotic systems, specifically the Hénon map, combined with edge detection to embed information within images in a concealed manner.

## Overview


### Steganographic Mechanism


**Chaotic Systems**: The core mechanism of `ChaosEdgeSteg` revolves around the [Hénon map](https://en.wikipedia.org/wiki/H%C3%A9non_map), a type of discrete-time dynamical system. Such systems are characterized by their sensitivity to initial conditions, a property commonly referred to as the "butterfly effect". Mathematically, the Hénon map is described by the following equations:

$$x_{n+1} = 1 - a x_n^2 + y_n$$

$$y_{n+1} = b x_n$$

Where \(a\) and \(b\) are constants. For `ChaosEdgeSteg`, we utilize typical values of \(a=1.4\) and \(b=0.3\) which produce chaotic behavior.

**Edge Detection**: `ChaosEdgeSteg` identifies edges in the cover image, which are regions with rapid intensity change, using the [Canny edge detection](https://en.wikipedia.org/wiki/Canny_edge_detector) method. These edges are less perceptible to the human eye when modified, making them ideal for embedding information.

**Embedding Mechanism**: The chaotic trajectory of the Hénon map is mapped onto the detected edge coordinates. This trajectory determines where in the image the payload will be embedded. Due to the chaotic nature of the Hénon map, this embedding is highly sensitive to minor changes in the image and nearly impossible to predict without knowledge of the initial conditions (i.e., the key).

## Features

- **Chaos-Based Steganography**: Utilizes the Hénon map to dictate the embedding positions.

- **Edge Adaptive Embedding**: Prioritizes edges for embedding, making the embedded information less perceptible.

	- **Contextual Thresholding**: Adaptively optimizes edge detection thresholds based on both image and payload size, further concealing the presence of the payload.

- **Enhanced Payload Security**: Requires possession of the key, payload length, and original cover image in order to extract steganographic content. Keys are checked against an embedded SHA256 hash as an additional validation mechanism.

- **Remote Extraction**: Provides options for payload extraction via remote hosts, enabling versatility in restrictive environments.

## Installation

To use `ChaosEdgeSteg`, you can clone the repository:

```bash  
git clone https://github.com/crypt0lith/ChaosEdgeSteg.git
cd ChaosEdgeSteg
```  

After cloning the repository, install the required dependencies:

```bash  
pip install -r requirements.txt
```  

## Usage

### Embedding Payload

```bash  
python -m chaosedgesteg embed [-v/-vv] -c <cover_image_path> -f <payload_file> -k 'secret_key' [-o <output_image_path>] [-q] [--save_key] [--save_bitmaps]
```  

### Extracting Payload

```bash  
python -m chaosedgesteg extract [-v/-vv] -c <cover_image_path> -i <stego_image_path> -k '0000::secret_key' [-o <output_file>] [-q] [-psx]
```

#### Remote Extraction

```bash  
python -m chaosedgesteg extract -c <cover_image_path> -iR stego_image.png <LHOST> <LPORT> -k '0000::secret_key' [-oR <output_file> <LHOST> <LPORT>] [--echo] [--obfuscate]  
```  

`-iR <stego_image.png> <LHOST> <LPORT>` instantiates an HTTP PUT server.

From a remote host, upload the stego image using `curl --upload-file`:

```bash  
curl --upload-file stego_image.png http://PUT_SERVER_IP:4444/
```  
  
  
- Use `-iR` with `--echo` to send the payload back to the remote host in plaintext:  
  
    ```bash  
    python -m chaosedgesteg extract -c <cover_image_path> -iR stego_image.png <LHOST> <LPORT> -k "0000::secret_key" --echo [--obfuscate]  
    ```  
- Use `-iR` with `--echo --obfuscate` **(recommended)** to base64 encode and XOR obfuscate the payload, then pipe the additional commands on the remote host:  
    ```bash  
    curl --upload-file stego_image.png http://PUT_SERVER_IP:4444/ | perl -pe 's/(.)/chr(ord($1) ^ 0xFF)/ge' | perl -pe 's/(.)/chr(ord($1) ^ 0x55)/ge' | base64 -d  
    ```  
  
Similar to `-iR`, `-oR <output_file> <LHOST> <LPORT>` can be used to instantiate an HTTP GET server.   
  
From a remote host, download the extracted file using the `wget` command:  
  
```bash  
wget http://GET_SERVER_IP:4884/extracted.zip
```