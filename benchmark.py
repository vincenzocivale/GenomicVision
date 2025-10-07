# run_sweep.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments, Trainer, AutoImageProcessor, DefaultDataCollator
from sklearn.metrics import accuracy_score, f1_score


from src.seq2image import GenomicImageGenerator
from src.model import inizialize_model

# ==============================================================================
# --- CONFIGURAZIONE PRINCIPALE DELLO SWEEP ---
# ==============================================================================

BASE_CONFIG = {
    "project_name": "GenomicVisionSweep_Refactored",
    "dataset_name": "katarinagresova/Genomic_Benchmarks_human_nontata_promoters",
    "sequence_col": "seq",
    "label_col": "label",
    "image_size": 16,
    "training": {
        "output_dir": "./results",
        "epochs": 50,
        "batch_size": 1024,
        "learning_rate": 0.005,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "save_total_limit": 1,
    }
}

MODELS_TO_TEST = [
    "timm/resnet18.a1_in1k",
    "timm/efficientnet_b0.ra_in1k",
    "google/vit-base-patch16-224",
]

DATASET_METHODS = ["cgr", "hilbert", "gaf", "3_channel"]

# ==============================================================================
# --- FUNZIONI DI SUPPORTO ---
# ==============================================================================

def compute_metrics(p):
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

def run_single_experiment(config, model_name, dataset_name, pil_dataset):
    """Esegue un singolo esperimento di addestramento."""
    
    run_name = f"{model_name.split('/')[-1]}_{dataset_name}"
    
    run = wandb.init(
        project=config["project_name"],
        config=config,
        name=run_name,
        reinit=True
    )
    wandb.config.update({"model_name": model_name, "dataset_type": dataset_name})

    processor = AutoImageProcessor.from_pretrained(model_name)
    
    def transform(examples):
        # Assicura che l'input sia sempre a 3 canali per compatibilità.
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = processor(images, return_tensors="pt")["pixel_values"]
        return examples

    processed_datasets = pil_dataset.map(transform, batched=True, remove_columns=['image'])
    
    labels = sorted(set(processed_datasets['train']['label']))
    
    model, image_processor = inizialize_model(
        model_name=model_name,
        num_labels=len(labels),
        pretrained=True
    )

    training_args = TrainingArguments(
        output_dir=f"{config['training']['output_dir']}/{run.name}",
        num_train_epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        weight_decay=config['training']['weight_decay'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        run_name=run.name
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets['train'],
        eval_dataset=processed_datasets['test'],
        compute_metrics=compute_metrics,
        data_collator=DefaultDataCollator(),
    )

    try:
        trainer.train()
    finally:
        wandb.finish()

# ==============================================================================
# --- ESECUZIONE DELLO SWEEP ---
# ==============================================================================

if __name__ == "__main__":
    
    # FASE 1: Genera tutti i dataset di immagini una sola volta.
    raw_dataset = load_dataset(BASE_CONFIG["dataset_name"])
    
    generator = GenomicImageGenerator(
        image_size=BASE_CONFIG["image_size"],
        sequence_col=BASE_CONFIG["sequence_col"],
        label_col=BASE_CONFIG["label_col"]
    )

    pil_datasets = {}
    for method in DATASET_METHODS:
        if method == "3_channel":
            dataset = DatasetDict()
            dataset['train'] = generator.generate_3_channel_dataset(raw_dataset["train"]) # type: ignore
            dataset['test'] = generator.generate_3_channel_dataset(raw_dataset["test"]) # type: ignore
            pil_datasets[method] = dataset
        else:
            dataset = DatasetDict()
            dataset['train'] = generator.generate_single_channel_dataset(raw_dataset["train"], method=method)
            dataset['test'] = generator.generate_single_channel_dataset(raw_dataset["test"], method=method)
            pil_datasets[method] = dataset
    
    # FASE 2: Esegui il ciclo di esperimenti.
    for dataset_name, dataset_obj in pil_datasets.items():
        for model_name in MODELS_TO_TEST:
            run_single_experiment(
                config=BASE_CONFIG,
                model_name=model_name,
                dataset_name=dataset_name,
                pil_dataset=dataset_obj
            )

    print("\n🎉 Sweep completato! Controlla i risultati su Weights & Biases.")