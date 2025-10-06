import os
import torch
import shutil
from tqdm import tqdm
import logging
from typing import Dict, Any, Generator

from src.data.processing import one_hot_encode, sequence_to_image
from src.data.dataset import load_hf_dataset

from datasets import DatasetDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    # --- Configuration ---
    HF_DATASET_NAME = "katarinagresova/Genomic_Benchmarks_human_nontata_promoters"
    IMG_WIDTH = 16
    IMG_HEIGHT = 16
    
    # Define final and temporary directories
    FINAL_HF_DIR = "/home/vcivale/GenomicVision/data/interim"

    raw_dataset = load_hf_dataset(HF_DATASET_NAME)
    processed_splits = DatasetDict()

    for data_split in raw_dataset.keys():
        logger.info(f"--- Processing split: {data_split} ---")

        # 2. Definisci una funzione di mappatura per la conversione
        def convert_to_image(sample):
            sequence_str = sample['seq']
            encoded_sequence = one_hot_encode(sequence_str)
            image_np = sequence_to_image(encoded_sequence, IMG_WIDTH, IMG_HEIGHT)
            # Restituiamo un dizionario con la nuova colonna 'image'
            return {"image": torch.from_numpy(image_np).float()}

        # 3. Applica la conversione a tutto lo split in modo efficiente
        # .map() è molto più veloce di un loop for
        logger.info(f"Applying image conversion to {data_split} split...")
        processed_dataset = raw_dataset[data_split].map(
            convert_to_image,
            batched=False, 
            remove_columns=["seq"]
        )
        
        # 4. Aggiungi lo split processato al nostro DatasetDict
        processed_splits[data_split] = processed_dataset


    logger.info(f"Saving the complete DatasetDict to {FINAL_HF_DIR}...")
    processed_splits.save_to_disk(FINAL_HF_DIR)