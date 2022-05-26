# Scaling Laws for Deep Learning Based Image Reconstruction

Code for reproducing the results from the paper __Scaling Laws For Deep Learning Based Image
Reconstruction__.

In particular
- Section 3: Empirical scaling laws for denoising
- Section 4: Empirical scaling laws for compressive sensing
- Section 5: Understanding scaling laws for denoising theoretically

The code for MRI reconstruction is based on the [fastMRI repository]( https://github.com/facebookresearch/fastMRI). The code for image denoising is based on [Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks](https://github.com/LabForComputationalVision/bias_free_denoising).

## Requirements
CUDA-enabled GPU is necessary to run the code. We tested this code using:
- Ubuntu 20.04
- CUDA 11.5
- Python 3.7.11
- PyTorch 1.10.0

## Installation
First, install PyTorch 1.10.0 with CUDA support following the instructions [here](https://pytorch.org/get-started/previous-versions/).
Then, to install the necessary packages run
```bash
pip install -r requirements.txt
```
## Datasets
### fastMRI
FastMRI is an open dataset, however you need to apply for access at https://fastmri.med.nyu.edu/. To run the experiments from our paper, you need to download the fastMRI brain dataset.

### ImageNet
ImageNet is an open dataset, and you can request access at https://image-net.org/download.php. To run the experiments from our paper, you need to download the ImageNet train set.

## Usage
The code to reproduce the numerical simnulations of the toy model is provided as a Jupyter Notebook.

The code to reproduce the experiments that build up to the empirical results in Sections 3 and 4 is given as Jupyter Notebooks as well as python scripts.
There, each experiment is defined by a training set size and a network size measured in number of channels in the first layer.

One or more experiments can be initialized at the same time and are executed in consecutive order. For example to obtain the results for two points in the scaling law for denoising run
```
python Empirical_SL_denoising.py \
--training True \
--val_testing True \
--test_testing True \
--exp_nums '001' '002' \
--train_sizes 300 3000 \
--channels 64 128 \
--path_to_ImageNet_train '/path/to/ImageNet/train/'
```
or equivalently for compressive sensing in the context of accelerated MRI
```
python Empirical_SL_CS.py \
--training True \
--testing True \
--exp_nums '001' '002' \
--train_sizes 50 100 \
--channels 64 128 \
--path_to_fastMRI_brain_dataset "brain_path: /path/to/fastMRI/brain/dataset/"
```
The results and checkpoints are stored in directories of the form `E001_t50_l4c64_bs1_lr001` indicating the training set size, the number of layers and channels, the batch size and learning rate.

After training and testing use `plot_scaling_law.py` to obtain a scaling law plot summarizing the results from all files of the form `E001...`,`E002...`,`...`.

## Acknowledgments and references
- [fastMRI repository]( https://github.com/facebookresearch/fastMRI)
- Zbontar et al. "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI". In: https://arxiv.org/abs/1811.08839* (2018).
- [bias free denoising](https://github.com/LabForComputationalVision/bias_free_denoising)
- Mohan et al. "Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks". In: *International Conference on Learning Representations.* (2020).
- Russakovsky et al. "ImageNet Large Scale Visual Recognition Challenge". In: *International Journal of Computer Vision* (2015).
