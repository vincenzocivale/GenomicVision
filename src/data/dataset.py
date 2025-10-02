from datasets import load_dataset

def load_hf_dataset(dataset_name, download=False):
    """
    Load a dataset from Hugging Face Datasets library.

    Args:
        dataset_name (str): The name of the dataset to load.
        download (bool): If True, download the dataset if not already present.

    Returns:
        Dataset: The loaded dataset object.
    """
    cache_dir = "/data2/home/vcivale/GenomicVision/data/raw"
    if download:
        dataset = load_dataset(dataset_name, download_mode="force_redownload", cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset