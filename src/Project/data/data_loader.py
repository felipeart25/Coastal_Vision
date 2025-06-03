import os
import sys
import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import pandas as pd



path = r"/p/11207608-coclico/MSc_students/Daniel/Scripts/outputs/v2.0/daily/"
os.chdir(path)  # Change working directory
sys.path.append(path)  # Add to Python path if needed


def load_tiff_images(tiff_folder):
# 1. Load the TIFF images
# Assume your TIFF images are stored in a folder "tiff_folder" with names like "binary_mask0.tif"
    tiff_files = sorted(glob.glob(tiff_folder),
                        key=lambda x: int(x.split('binary_mask')[-1].split('.tif')[0]))

    images = []
    for file in tiff_files:
        with rasterio.open(file) as src:
            # Read as a numpy array (e.g., if single band, shape might be (height, width))
            img = src.read(1)  # Read the first band
            #transform = src.transform  # Get the affine transform
            #bounds = src.bounds
            #xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top
            #crs = src.crs
            images.append(img)


    # Convert list to numpy array with shape (num_days, height, width)
    images = np.array(images)
    images = np.expand_dims(images, axis=1)  # Now shape: (num_days, 1, height, width)
    return images

# 2. Load the wave data
def load_wave_data(csv_path):

    wave_df = pd.read_csv(csv_path)
    return wave_df[['Hs', 'Direction', 'Period']].values  # Shape: (num_days, 3)

def create_sequences(images, wave_data, input_length=10, target_length=10):

# 3. Define the sequence length: 20 days in total (10 input, 10 target)
    seq_length = input_length + target_length
    num_days = images.shape[0]
    img_input = []
    img_target = []
    wave_seqs = []

# 4. Create sequences using a sliding window

    for i in range(num_days - seq_length + 1):
        input_seq = images[i : i + input_length]       # Days i to i+9
        target_seq = images[i + input_length : i + seq_length]  # Days i+10 to i+19
        wave_seq = wave_data[i * 24 : (i + input_length) * 24]


        img_input.append(input_seq)
        img_target.append(target_seq)
        wave_seqs.append(wave_seq)

# Convert lists to tensors
    img_input = torch.tensor(np.array(img_input), dtype=torch.float32)
    img_target = torch.tensor(np.array(img_target), dtype=torch.float32)
    wave_seqs = torch.tensor(np.array(wave_seqs), dtype=torch.float32)

# 5. Create a PyTorch Dataset and DataLoaders
# Here we combine the input images, target images, and wave data
    return TensorDataset(img_input, img_target, wave_seqs)

def get_dataloaders(path, batch_size, input_length=10, target_length=10):

    tiff_path = os.path.join(path, 'binary_mask*.tif')
    cvs_path = os.path.join(path, 'time_series_data.csv')

    images = load_tiff_images(tiff_path)
    wave_data = load_wave_data(cvs_path)
    dataset = create_sequences(images, wave_data, input_length, target_length)

# Split dataset (for example: 70% train, 15% validation, 15% test)
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    indices = list(range(total))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create subsets using Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset

