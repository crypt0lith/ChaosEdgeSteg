![image](https://raw.githubusercontent.com/crypt0lith/ChaosEdgeSteg/master/banner.png)
# ChaosEdgeSteg
## Installation

Install the package using `pip`:
```shell
pip install git+https://github.com/crypt0lith/ChaosEdgeSteg.git
```

## Basic usage

```shell
# embed files and directories in an image
chaosedgesteg embed /path/to/image.png secret.txt ./mydir/ -O /path/to/stegimage.png

# use -r/--remote to use a remote image for the cover image
chaosedgesteg embed --remote http://example.com/image.png ./file1 ./file2 -O /path/to/stegimage.png

# extract files from the steg image
chaosedgesteg extract /path/to/image.png /path/to/stegimage.png -O /path/to/extracted.zip
```