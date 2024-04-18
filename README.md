# STARCAM: <ins>S</ins>canning <ins>T</ins>opographic <ins>A</ins>ll-in-focus <ins>R</ins>econstruction with a <ins>C</ins>omputational <ins>A</ins>rray <ins>M</ins>icroscope
<center><img src="/media/starcam.jpg" alt="STARCAM" width="800"/></center>

STARCAM (<ins>S</ins>canning <ins>T</ins>opographic <ins>A</ins>ll-in-focus <ins>R</ins>econstruction with a <ins>C</ins>omputational <ins>A</ins>rray <ins>M</ins>icroscope) is a new computational 3D microscopy approach that enables scalable multi-gigapixel 3D topographic reconstruction over >110 cm<sup>2</sup> lateral fields of view (FOVs) and multi-mm axial ranges at micron-scale resolution. STARCAM is a direct extension of 3D-RAPID (https://github.com/kevinczhou/3D-RAPID), combining a parallelized 54-camera architecture and 3-axis sample scanning. From the resulting multi-terabyte-per-sample datasets, STARCAM reconstructs and stitches a 6-gigapixel, all-in-focus gigamosaic along with a coregistered 3D height map, using both parallax and sharpness information from the overlapped FOVs and z-stacks. Like 3D-RAPID, STARCAM trains a convolutional neural network (CNN) to map from the raw data to the 3D height maps via self supervision. This repository provides the Python code for performing these large-scale reconstructions.

For more details, see our accompanying paper [here](https://doi.org/10.1186/s40537-024-00901-0) (or our [arXiv preprint](https://arxiv.org/abs/2306.02634)). See also the repositories for [3D-RAPID](https://github.com/kevinczhou/3D-RAPID) and [smartphone photogrammetry](https://github.com/kevinczhou/mesoscopic-photogrammetry), which STARCAM extends.  

## Data
Due to the exceedingly large sizes of the datasets (up to 2.1 TB/sample), they are not publicly available at this time -- please contact us. For best performance, the data should be stored on a storage device with fast sustained read speeds (e.g., NVMe drives), since the data will be streamed as random patches during training.

## Setting up your compute environment
We used the same environment as for 3D-RAPID: https://github.com/kevinczhou/3D-RAPID?tab=readme-ov-file#setting-up-your-environment  
The patch and batch sizes were chosen to fit on a 24-GB GPU. Your CPU should ideally have at least 256 GB of RAM.

## Usage
You will only need to directly interact with the two Jupyter notebooks: `training.ipynb`, followed by `gigamosaic_inference.ipynb`.
