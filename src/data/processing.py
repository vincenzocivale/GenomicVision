import numpy as np
from typing import Dict, Optional, Any
import os
import torch
from tqdm import tqdm



from src.data.dataset import load_hf_dataset

# Define the standard mapping from nucleotide to an integer index.
NUCLEOTIDE_MAP: Dict[str, int] = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def one_hot_encode(sequence: str, nucleotide_map: Dict[str, int] = NUCLEOTIDE_MAP, include_n: bool = False) -> np.ndarray:
    """
    Performs one-hot encoding on a DNA sequence.

    Args:
        sequence (str): The input DNA sequence (e.g., "ATGC").
        nucleotide_map (Dict[str, int]): Mapping from nucleotide to index.
        include_n (bool): If True, adds a 5th channel for 'N' (unknown) bases.

    Returns:
        np.ndarray: A 2D NumPy array of shape (sequence_length, num_channels),
                    where num_channels is 4 (or 5 if include_n is True).
    """
    num_channels = len(nucleotide_map) + 1 if include_n else len(nucleotide_map)
    encoded_sequence = np.zeros((len(sequence), num_channels), dtype=np.uint8)
    
    for i, nucleotide in enumerate(sequence.upper()):
        index = nucleotide_map.get(nucleotide)
        if index is not None:
            encoded_sequence[i, index] = 1
        elif include_n:
            # If the nucleotide is not in the map (e.g., 'N'), set the last channel.
            encoded_sequence[i, -1] = 1
            
    return encoded_sequence

def sequence_to_image(sequence_1d: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Maps a 1D one-hot encoded sequence to a 2D image format using a simple raster scan.

    The sequence is padded with zeros if it's shorter than the image area.
    The final image will have shape (num_channels, height, width).

    Args:
        sequence_1d (np.ndarray): A 1D one-hot encoded sequence from one_hot_encode.
                                  Shape: (sequence_length, num_channels).
        width (int): The desired width of the output image.
        height (int): The desired height of the output image.

    Returns:
        np.ndarray: A 3D NumPy array representing the image, with channels first.
                    Shape: (num_channels, height, width).
    """
    seq_len, num_channels = sequence_1d.shape
    image_size = width * height
    
    # Pad the sequence with zeros if it's shorter than the image area.
    if seq_len < image_size:
        padding_needed = image_size - seq_len
        padding = np.zeros((padding_needed, num_channels), dtype=np.uint8)
        sequence_1d = np.concatenate([sequence_1d, padding])
    elif seq_len > image_size:
        # Truncate the sequence if it's longer.
        sequence_1d = sequence_1d[:image_size, :]
        
    # Reshape the flat sequence into a 2D grid and transpose for (C, H, W) format.
    # 1. Reshape to (H, W, C)
    image_hwc = sequence_1d.reshape(height, width, num_channels)
    # 2. Transpose to (C, H, W) which is the standard format for PyTorch.
    image_chw = image_hwc.transpose(2, 0, 1)
    
    return image_chw

def genome2image_dataset(
    hf_dataset_name: str,
    hf_subset_name: str,
    split: str,
    save_dir: str,
    image_width: int,
    image_height: int
) -> None:
    """
    Loads a dataset from Hugging Face, converts each sequence to an image tensor,
    and saves each sample locally.

    Args:
        hf_dataset_name (str): Name of the Hugging Face dataset.
        hf_subset_name (str): Name of the subset.
        split (str): Data split to process ('train', 'validation', 'test').
        save_dir (str): The directory where processed tensors will be saved.
        image_width (int): Width of the output image.
        image_height (int): Height of the output image.
    """
    # 1. Create the output directory
    output_path = os.path.join(save_dir, hf_subset_name, split)
    os.makedirs(output_path, exist_ok=True)

    # 2. Load the raw sequence dataset
    raw_dataset = load_hf_dataset(hf_dataset_name, hf_subset_name, split)

    # 3. Loop, process, and save each sample
    for i, sample in enumerate(tqdm(raw_dataset, desc=f"Processing {split} split")):
        sequence_str = sample['sequence']
        label = sample['label']
        
        # Convert sequence to image tensor
        encoded_sequence = one_hot_encode(sequence_str)
        image_np = sequence_to_image(encoded_sequence, image_width, image_height)
        image_tensor = torch.from_numpy(image_np).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Create a dictionary to save
        processed_sample: Dict[str, Any] = {
            "image": image_tensor,
            "label": label_tensor
        }
        
        # Save the processed sample as a PyTorch tensor file
        file_path = os.path.join(output_path, f"sample_{i}.pt")
        torch.save(processed_sample, file_path)

    print(f"Preprocessing complete. Saved {len(raw_dataset)} files to {output_path}")
