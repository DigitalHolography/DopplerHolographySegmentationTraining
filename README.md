# DopplerHolographySegmentationTraining

Deep learning framework for retinal vessel segmentation from Doppler holography images.

This repository contains the training and evaluation code used in the article **"Improving segmentation of retinal arteries and veins using cardiac signal in doppler holograms"** accepted at ISBI2026 ([IEEE](https://ieeexplore.ieee.org/document/11515426), [Arxiv](https://ieeexplore.ieee.org/document/11515426)). It provides implementations of several segmentation architectures, loss functions, metrics, and utilities for reproducible experiments on holographic Doppler imaging data.

The corresponding dataset is available on Hugging Face:
**https://huggingface.co/datasets/DigitalHolography/**

## Repository Structure

```text
.
├── models/                 # Segmentation model implementations
├── losses.py               # Loss functions
├── metrics.py              # Evaluation metrics
├── model_utils.py          # Utils for models loading and saving
├── data_utils.py           # Utils for dataset loading and data visualization
├── requirements.txt        # Python dependencies
└── model_benchmark.ipynb   # Training and evaluation notebook
```

## Installation

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Usage

The main entry point is the `model_benchmark.ipynb` notebook, which provides examples for:

* Loading the dataset
* Training segmentation models
* Evaluating performance metrics
* Comparing different input modalities

## Dataset

The repository is designed to work with the **Retinal Vessel Segmentation from Holographic Doppler Imaging** dataset, which contains:

* Power Doppler images (M0)
* Artery/vein segmentation masks
* Correlation maps
* Diasys images

These modalities allow the study of how temporal information derived from the cardiac cycle can improve retinal artery-vein segmentation.

The dataset loading is fully handled in the notebook.

## Citation

If you use this repository or the dataset in your work, please cite:

> M. Dubosc et al., "Improving Segmentation of Retinal Arteries and Veins Using Cardiac Signal in Doppler Holograms," 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI), London, United Kingdom, 2026, pp. 1-5, doi: 10.1109/ISBI61048.2026.11515426.
