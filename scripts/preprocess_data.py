import os
import torch
import shutil
from tqdm import tqdm
import logging
from typing import Dict, Any, Generator

from src.data.processing import one_hot_encode, sequence_to_image
from src.data.dataset import load_hf_dataset

from datasets import Dataset as HFDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def process_and_save_intermediate(
    raw_dataset: HFDataset,
    temp_dir: str,
    image_width: int,
    image_height: int
) -> None:
    """Processes each sample and saves it as an intermediate .pt file."""
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Processing {len(raw_dataset)} samples and saving to temp dir: {temp_dir}")
    
    for i, sample in enumerate(tqdm(raw_dataset, desc="Creating intermediate files")):
        sequence_str = sample['seq']
        label = sample['label']
        
        encoded_sequence = one_hot_encode(sequence_str)
        image_np = sequence_to_image(encoded_sequence, image_width, image_height)
        image_tensor = torch.from_numpy(image_np).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        processed_sample: Dict[str, Any] = {"image": image_tensor, "label": label_tensor}
        file_path = os.path.join(temp_dir, f"sample_{i}.pt")
        torch.save(processed_sample, file_path)

def create_hf_dataset_from_intermediate(
    temp_dir: str,
    hf_save_path: str
) -> None:
    """Consolidates intermediate .pt files into a final Hugging Face dataset."""
    
    def data_generator() -> Generator[Dict[str, Any], None, None]:
        """A generator that loads each .pt file and yields its content."""
        file_names = sorted(os.listdir(temp_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
        for fname in file_names:
            if fname.endswith(".pt"):
                file_path = os.path.join(temp_dir, fname)
                yield torch.load(file_path)

    logger.info(f"Creating Hugging Face dataset from files in {temp_dir}...")
    # Create dataset from the generator
    hf_dataset = HFDataset.from_generator(data_generator)
    
    logger.info(f"Saving consolidated dataset to {hf_save_path}...")
    # Save the final dataset to disk
    hf_dataset.save_to_disk(hf_save_path)
    logger.info("Final Hugging Face dataset saved successfully.")


if __name__ == "__main__":
    # --- Configuration ---
    HF_DATASET_NAME = "katarinagresova/Genomic_Benchmarks_human_nontata_promoters"
    IMG_WIDTH = 16
    IMG_HEIGHT = 16
    
    # Define final and temporary directories
    FINAL_HF_DIR = "/home/vcivale/GenomicVision/data/interim"
    TEMP_PT_DIR = "/home/vcivale/GenomicVision/data/temp"
    CLEANUP_TEMP_FILES = True

    raw_dataset = load_hf_dataset(HF_DATASET_NAME)

    for data_split in raw_dataset.keys():
        logger.info(f"--- Processing split: {data_split} ---")
        
        # Define paths for this split
        temp_split_dir = os.path.join(TEMP_PT_DIR, data_split)
        final_split_dir = os.path.join(FINAL_HF_DIR, data_split)
        
        
        # 2. Process and save intermediate files
        process_and_save_intermediate(
            raw_dataset=raw_dataset[data_split],
            temp_dir=temp_split_dir,
            image_width=IMG_WIDTH,
            image_height=IMG_HEIGHT
        )
        
        # 3. Consolidate into a final HF dataset
        create_hf_dataset_from_intermediate(
            temp_dir=temp_split_dir,
            hf_save_path=final_split_dir
        )
        
        # 4. (Optional) Clean up intermediate files
        if CLEANUP_TEMP_FILES:
            logger.info(f"Cleaning up temporary directory: {temp_split_dir}")
            shutil.rmtree(temp_split_dir)

    print("\nAll splits have been processed and saved as Hugging Face datasets.")