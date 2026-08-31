![image](https://raw.githubusercontent.com/crypt0lith/ChaosEdgeSteg/master/banner.png)

ChaosEdgeSteg is a chaos-based edge-adaptive steganography tool.
The steganographic technique utilizes Canny edge detection and the Rössler folded-towel map hyperchaotic attractor
to embed arbitrary files (in the format of a ZIP archive) into an image,
in a way that is very difficult to detect via steganalysis and near-impossible to extract without the original cover image (and optionally, password).

The embedding scheme comprises of 2 steps:

1.  Find the Canny edges of the cover image, adjusting thresholds adaptively to accommodate payload size.
    Edge regions are suitable for data hiding because the human visual system is less sensitive to distortions in edge regions.
2.  Embed the payload along edge pixel positions in a chaotic pseudorandom index order derived from the given password, or a default key if none was provided.

For large payloads, use a large cover image with many high-contrast regions, for example an [image of a tiger](https://learnopencv.com/edge-detection-using-opencv/#canny-edge).

## Usage

```bash
# embed files and directories in an image
chaosedgesteg embed /path/to/image.png secret.txt ./mydir/ -O /path/to/stegimage.png

# extract files from the steg image
chaosedgesteg extract /path/to/image.png /path/to/stegimage.png -O /path/to/extracted.zip

# use -r/--remote to use a remote image for the cover image
chaosedgesteg extract --remote http://example.com/image.png /path/to/stegimage.png -O /path/to/extracted.zip
```

## Installation

Install the package using your preferred package manager (`uv`, `pipx`, etc.):

```bash
uv tool install git+https://github.com/crypt0lith/ChaosEdgeSteg.git
```
