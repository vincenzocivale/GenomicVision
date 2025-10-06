"""
Script di training semplificato per modelli di visione su sequenze genomiche.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

# from src.data.dataset import ImageCollator
from src.model.vision_models import VisionModelFactory, VisionModelType



# Dataset
DATASET_PATH = "/home/vcivale/GenomicVision/data/interim"  
NUM_CLASSES = 2  

# Modello
MODEL_TYPE = "efficientnet_b0"  
PRETRAINED = False
INPUT_CHANNELS = 4

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
OUTPUT_DIR = "/equilibrium/datasets/TCGA-histological-data/genomic_vision//results"

# W&B
WANDB_PROJECT = "genomic-vision"
WANDB_RUN_NAME = f"{MODEL_TYPE}_pretrained_{PRETRAINED}"




def compute_metrics(pred: EvalPrediction):
    """Calcola metriche."""
    preds = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }



def main():
    # Init W&B
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
    
    print("Caricamento dataset...")
    dataset = load_from_disk(DATASET_PATH)

    
    # Split validation dal train (10%)
    if 'validation' not in dataset.keys():
        train_val = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset['train'] = train_val['train']
        dataset['validation'] = train_val['test']

    dataset = dataset.rename_column("image", "pixel_values")
    dataset.set_format(type='torch', columns=['pixel_values', 'label'])

    
    # Crea modello
    print(f"Creazione modello {MODEL_TYPE}...")
    model = VisionModelFactory.create(
        model_type=VisionModelType(MODEL_TYPE),
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        pretrained=PRETRAINED
    )
    
    
    # Parametri
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri: {total_params:,}")
    
    # Training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=500,
        
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
        dataloader_drop_last=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        # data_collator=ImageCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Training
    print("Training...")
    trainer.train()
    
    # Evaluation su test
    print("Valutazione test set...")
    test_metrics = trainer.evaluate(dataset['test'], metric_key_prefix="test")
    
    print("\nRisultati finali:")
    print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Test F1: {test_metrics['test_f1']:.4f}")
    
    # Salva modello
    trainer.save_model()
    wandb.finish()


if __name__ == "__main__":
    main()