from datasets import load_dataset
import numpy as np
import torch

def load_hf_dataset(dataset_name, download=False):
    """
    Load a dataset from Hugging Face Datasets library.

    Args:
        dataset_name (str): The name of the dataset to load.
        download (bool): If True, download the dataset if not already present.

    Returns:
        Dataset: The loaded dataset object.
    """
    cache_dir = "/home/vcivale/GenomicVision/data/raw"
    if download:
        dataset = load_dataset(dataset_name, download_mode="force_redownload", cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset

# class ImageCollator:
#     """Collator per convertire immagini PIL in tensori."""
    
#     def __call__(self, features):
#         images = []
#         labels = []
        
#         for feature in features:
#             # Converti immagine PIL in tensor
#             img = feature['image']
#             if hasattr(img, 'convert'):  # PIL Image
#                 img = np.array(img)
            
#             # Converti da HWC a CHW se necessario
#             if img.ndim == 3 and img.shape[-1] == 4:
#                 img = np.transpose(img, (2, 0, 1))
            
#             images.append(torch.from_numpy(img).float() / 255.0)
#             labels.append(feature['label'])
        
#         return {
#             'pixel_values': torch.stack(images),
#             'labels': torch.tensor(labels, dtype=torch.long)
#         }