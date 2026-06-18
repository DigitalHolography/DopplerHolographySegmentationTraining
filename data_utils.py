import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
# import model_utils
import os
# import cv2
from fastai.vision.all import *

class BinaryVesselDataset:
    def __init__(self, hf_split, input=["M0"], size=(512, 512)):
        self.data = []
        for sample in hf_split:
            x = np.array(sample[input[0]])                    # already preprocessed numpy
            y = np.array(np.array(sample["maskArtery"]).astype(bool) | np.array(sample["maskVein"]).astype(bool))
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = sample[0]  # PIL Image
        y = sample[1]  # PIL Image
        return PILImage.create(x.astype(np.uint8)), PILMask.create(y.astype(np.uint8))

class ArteryVeinDataset:
    def __init__(self, hf_split, input=["M0", "correlation", "diasys"], size=(512, 512)):
        self.data = []

        for sample in hf_split:
            x = np.zeros((size[0], size[1], len(input)), dtype=np.uint8)
            for i, col in enumerate(input):
                x[:, :, i] = np.array(sample[col].convert("L").resize(size, Image.BILINEAR))

            artery = np.array(sample["maskArtery"].convert("L").resize(size, Image.NEAREST))
            vein   = np.array(sample["maskVein"].convert("L").resize(size, Image.NEAREST))

            # encode as 0,1,2,3 (fastai-compatible)
            y = np.zeros((size[0], size[1]), dtype=np.uint8)
            y[artery > 0] = 1
            y[vein > 0] += 2

            self.data.append((x, y))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return PILImage.create(x), PILMask.create(y)
    
def multi2onehot_tensor(x:torch.Tensor, # Non one-hot encoded targs
        dim:int=2 # The axis to stack for encoding (class dimension)
    ) -> torch.Tensor:
        "Creates one binary mask per class"
        return torch.stack([torch.where((x==1) | (x==3), 1, 0), torch.where((x==2) | (x==3), 1, 0)], dim=dim)

def multi2onehot(x:np.ndarray, # Non one-hot encoded targs
        axis:int=2 # The axis to stack for encoding (class dimension)
    ) -> np.ndarray:
        "Creates one binary mask per class"
        return np.stack([np.where((x==1) | (x==3), 1, 0), np.where((x==2) | (x==3), 1, 0)], axis=axis)


def mask_to_rgb(mask):
    # Create an RGB image

    if len(mask.shape) == 2:
        one_hot_masks = multi2onehot(mask) # Convert to one-hot encoding
    else:
        one_hot_masks = mask.transpose(1, 2, 0)  # Change shape to [H, W, C]

    rgb_image = np.zeros((one_hot_masks.shape[0], one_hot_masks.shape[1], 3), dtype=np.uint8)  # Shape: [H, W, 3]

    # Map the first mask to the red channel and the second to the blue channel
    rgb_image[..., 0] = one_hot_masks[:,:,0] * 255  # Red channel
    rgb_image[..., 2] = one_hot_masks[:,:,1] * 255  # Blue channel

    return Image.fromarray(rgb_image)

def split_channels(inputs, channels):
    lists = [[] for _ in range(len(inputs))]
    for i in range(len(inputs)):
        for c in range(channels):
            lists[i].append(inputs[i][c,:,:])
    return lists

def show_masks(inputs, masks, masks_pred=None, multi=False, cmap='viridis', n=20):
    """Displays input images, ground truth masks, and optionally predicted masks in a grid format.
    Parameters:
    - inputs: List of input images (numpy arrays).
    - masks: List of ground truth masks (numpy arrays).
    - masks_pred: List of predicted masks (numpy arrays), optional.
    - multi: Boolean indicating if the masks are multi-class (True) or binary (False).
    - cmap: Colormap for displaying images.
    - n: Number of samples to display.
    """
    nb_rows = min(len(inputs), n)
    channels = 1
    # plot images and masks

    if cmap == 'gray':
        a = inputs[0]
        channels = 1 if len(a.shape) == 2 else a.shape[0]
        if channels != 1:
            inputs = split_channels(inputs, channels)
    
    nb_cols = channels + (1 if masks_pred is None else 2)
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(5*nb_cols, 5*nb_rows))
    
    for idx in range(nb_rows):
        for c in range(channels):
            axes[idx][c].imshow(Image.fromarray((np.squeeze(inputs[idx][c])*255).astype(np.uint8)), cmap=cmap)
            mask = np.squeeze(masks[idx])
            axes[idx][channels].imshow(mask_to_rgb(mask) if multi else mask, cmap="gray")
        if masks_pred is not None:
            axes[idx][channels+1].imshow(mask_to_rgb(np.squeeze(masks_pred[idx])) if multi else np.squeeze(masks_pred[idx][0]), cmap="gray")
        
    # add subtitles
    for c in range(channels):
        axes[0][c].set_title('Input')
    axes[0][channels].set_title('Ground truth masks')
    if masks_pred is not None:
        axes[0][channels + 1].set_title('Predicted masks')

    plt.show()

def combine_binary_masks(mask_red: np.ndarray, mask_blue: np.ndarray) -> Image.Image:
    """
    Combine two binary numpy arrays into an RGB image.
    - mask_red will appear in the red channel.
    - mask_blue will appear in the blue channel.
    """
    # Ensure both are binary and same shape
    assert mask_red.shape == mask_blue.shape, "Masks must have the same shape"
    
    # Normalize to 0–255 uint8
    red = (mask_red.astype(np.uint8)) * 255
    blue = (mask_blue.astype(np.uint8)) * 255
    
    # Create RGB channels (no green)
    rgb = np.stack([red, np.zeros_like(red), blue], axis=-1)
    
    # Convert to PIL image for easy display or saving
    return Image.fromarray(rgb)