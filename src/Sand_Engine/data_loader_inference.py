import os
import sys
import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import wandb


def extract_index(filename):
        base = os.path.basename(filename)
        number = ''.join(filter(str.isdigit, base))
        return int(number) if number else -1

def load_tiff_images(tiff_folder):
    # Load the TIFF images

    tiff_files = glob.glob(os.path.join(tiff_folder, "binary_mask*.tif"))

    tiff_files = sorted(tiff_files, key=extract_index)

    images = []
    for file in tiff_files:
        img = Image.open(file)
        img_array = np.array(img)  # Convert to NumPy array
        images.append(img_array)
    

    # Convert list to numpy array with shape (num_days, height, width)
    images = np.array(images)
    images = normalize_images(images)
    images = np.expand_dims(images, axis=1)  # Now shape: (num_days, 1, height, width)
    return images

def normalize_images(images):
    """Normalize each image to range [0, 1]"""
    images = images.astype(np.float32)
    max_val = np.max(images)
    if max_val > 0:
        images = images / max_val
    return images

def normalize_wave(wave_data):
    """
    Normalize each wave feature (Hs, Dir, Period) using standard score:
    z = (x - mean) / std
    Returns normalized data and stats for possible denormalization.
    """
    wave_data = wave_data.astype(np.float32)
    mean = wave_data.mean(axis=0)     # shape (3,)
    std = wave_data.std(axis=0) + 1e-8  # prevent division by zero
    normalized = (wave_data - mean) / std
    return normalized, mean, std

def load_wave_data(path):
    csv_path = os.path.join(path, 'time_series_data.csv')
    wave_df = pd.read_csv(csv_path)
    return wave_df[['Hs', 'Direction', 'Period']].values  # Shape: (num_days, 3)

def create_sequences(img_data, wave_data, step, input_length=10, target_length=10):
    seq_length = input_length + target_length
    num_days = img_data.shape[0]
    
    img_input, img_target, wave_input, wave_target = [], [], [], []

    for i in range(num_days - seq_length + 1):
        input_seq = img_data[i : i + input_length]
        target_seq = img_data[i + input_length : i + seq_length]
        
        wave_start = i * step
        wave_end = (i + input_length) * step
        wave_seq = wave_data[wave_start : wave_end]
        wave_future_seq = wave_data[wave_end : wave_end + (target_length * step)]

        img_input.append(input_seq)
        img_target.append(target_seq)
        wave_input.append(wave_seq)
        wave_target.append(wave_future_seq)

    return (
        torch.tensor(np.array(img_input), dtype=torch.float32),
        torch.tensor(np.array(img_target), dtype=torch.float32),
        torch.tensor(np.array(wave_input), dtype=torch.float32),
        torch.tensor(np.array(wave_target), dtype=torch.float32)
    )

def calculate_foreground_percentage(tensor):
    """
    Args:
        tensor (torch.Tensor): shape (N, T, 1, H, W), binary mask

    Returns:
        float: percentage of pixels with value 1 across the entire dataset
    """
    total_pixels = tensor.numel()
    foreground_pixels = (tensor == 1).sum().item()
    percentage = (foreground_pixels / total_pixels) * 100
    return percentage

def data_loaders(main_dir, batch_size, case=1, time="weekly"):
    output_freq = 2  # Frequency of output data in hours
    time_dict = {
        #'daily': 4,
        'weekly': 34,
        'bi-monthly': 72,
        'monthly': 144,
    }


    main_dir = r"/p/11207608-coclico/MSc_students/Daniel/Scripts/outputs/"
    cases = {
        1: ["v0_1", "v1_1", "v2_1", "v3_1", "v4_1", "v5_1", "v6_1", "v7_1", "v8_1"],
        2: ["v0_1", "v2_1", "v4_1", "v6_1", "v7_1"]
    }
    #case = 1
    #time = "weekly"
    step = time_dict[time] // output_freq
    scenario = cases.get(case, [])

    # Match folders that end with one of the version tags
    scenario_folders = [
        folder for folder in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, folder)) and any(folder.endswith(v) for v in scenario)
    ]

    print("Matching folders:", scenario_folders)
    data_loaders = {}

    for data_type in ["training", "validation", "testing"]:
        img_inputs = []
        img_targets = []
        wave_inputs = []
        wave_targets = []

        for folder in scenario_folders:
            dir_path = os.path.join(main_dir, folder, time, data_type)

            try:

                images = load_tiff_images(dir_path)
                waves = load_wave_data(dir_path)
                wave_data, wave_mean, wave_std = normalize_wave(waves)

                img_input, img_target, wave_input, wave_future = create_sequences(images, wave_data, step)

                img_inputs.append(img_input)
                img_targets.append(img_target)
                wave_inputs.append(wave_input)
                wave_targets.append(wave_future)

            except ValueError as e:
                print(f"Error processing folder {folder} in {data_type}: {e}")
                continue

        # Concatenate all tensors for this data_type
        img_input = torch.cat(img_inputs, dim=0)
        img_target = torch.cat(img_targets, dim=0)
        wave_input = torch.cat(wave_inputs, dim=0)
        wave_future = torch.cat(wave_targets, dim=0)

        print("Train input shape:", img_input.shape)
        print("Train target shape:", img_target.shape)
        print("Train wave input shape:", wave_input.shape)
        print("Train wave future shape:", wave_future.shape)

        


        # Create dataset and dataloader
        dataset = TensorDataset(img_input, img_target, wave_input, wave_future)
        shuffle = data_type == "training"
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        data_loaders[data_type] = loader

    

    train_loader = data_loaders["training"]
    val_loader = data_loaders["validation"]
    test_loader = data_loaders["testing"]

    train_inputs, _, _, _ = next(iter(train_loader))
    val_inputs, _, _, _ = next(iter(val_loader))
    test_inputs, _, _, _ = next(iter(test_loader))

    train_foreground = calculate_foreground_percentage(train_inputs)
    val_foreground = calculate_foreground_percentage(val_inputs)
    test_foreground = calculate_foreground_percentage(test_inputs)

    training_seqs = len(train_loader.dataset)
    validation_seqs = len(val_loader.dataset)
    testing_seqs = len(test_loader.dataset)

    summary = {
        "Case": case,
        "Time": time,
        "training_seqs": training_seqs,
        "validation_seqs": validation_seqs,
        "testing_seqs": testing_seqs,
        "train_foreground": f"{train_foreground:.2f}%",
        "val_foreground": f"{val_foreground:.2f}%",
        "test_foreground": f"{test_foreground:.2f}%",
    }

    summary_df = pd.DataFrame([summary])
    print("Summary DataFrame:")
    print(summary_df)

    # Log the summary to Weights & Biases
    fig, ax = plt.subplots(figsize=(12, 2))  # Adjust height depending on rows
    ax.axis('off')
    tbl = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    # Save the table as an image
    plt.tight_layout()
    plt.savefig("dataset_summary.png", dpi=300)
    plt.close()
    wandb.log({"Dataset Summary": wandb.Image("dataset_summary.png")})

    for inputs, targets, wave, wave_fut in train_loader:
        print("Train Inputs shape:", inputs.shape)  # Should be (B, T-10, 1, H, W)
        print("Train Targets shape:", targets.shape)  # Should be (B, 10, 1, H, W)
        print("Wave shape:", wave.shape)  # Should be (B, 480, 1)
        print("Wave shape:", wave_fut.shape)  # Should be (B, 240, 3)
        break

    for inputs, targets, wave, wave_fut in val_loader:
        print("Validation Inputs shape:", inputs.shape)  # Should be (B, T-10, 1, H, W)
        print("Validation Targets shape:", targets.shape)  # Should be (B, 10, 1, H, W)
        print("Wave shape:", wave.shape)  # Should be (B, 480, 1)
        print("Wave shape:", wave_fut.shape)  # Should be (B, 240, 3)
        break

    for inputs, targets, wave, wave_fut in test_loader:
        print("Test Inputs shape:", inputs.shape)  # Should be (B, T-10, 1, H, W)
        print("Test Targets shape:", targets.shape)  # Should be (B, 10, 1, H, W)
        print("Wave shape:", wave.shape)  # Should be (B, 480, 1)
        print("Wave shape:", wave_fut.shape)  # Should be (B, 240, 3)
        break

    return train_loader, val_loader, test_loader