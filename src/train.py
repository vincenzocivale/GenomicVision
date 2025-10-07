import wandb
import numpy as np
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score


def inizialize_training(model, training_args, dataset):

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
    )

    return trainer

def compute_metrics(p):
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}



