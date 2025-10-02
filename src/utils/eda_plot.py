import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
from collections import Counter
from typing import List

def plot_class_distribution(dataset: Dataset, title: str = "Class Distribution") -> None:
    """
    Analyzes and plots the distribution of labels in the dataset.

    Args:
        dataset (Dataset): The Hugging Face dataset, expected to have a 'label' column.
        title (str): The title for the plot.
    """
    labels = dataset['label']
    label_counts = Counter(labels)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()))
    
    plt.title(title, fontsize=16)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(ticks=list(label_counts.keys()), labels=[f"Class {k}" for k in label_counts.keys()])
    plt.show()

def plot_sequence_length_distribution(dataset: Dataset, title: str = "Sequence Length Distribution") -> None:
    """
    Calculates and plots the distribution of sequence lengths.

    Args:
        dataset (Dataset): The Hugging Face dataset, expected to have a 'sequence' column.
        title (str): The title for the plot.
    """
    lengths = [len(seq) for seq in dataset['seq']]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, kde=False, bins=30)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.show()

def plot_gc_content_distribution(dataset: Dataset, title: str = "GC Content Distribution by Class") -> None:
    """
    Calculates and plots the distribution of GC content for each class.

    Args:
        dataset (Dataset): The dataset with 'sequence' and 'label' columns.
        title (str): The title for the plot.
    """
    
    def calculate_gc_content(sequence: str) -> float:
        """Helper function to calculate GC content of a DNA sequence."""
        if not sequence:
            return 0.0
        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        return (gc_count / len(sequence)) * 100

    df = dataset.to_pandas()
    df['gc_content'] = df['seq'].apply(calculate_gc_content)
    
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x='gc_content', hue='label', kde=True, bins=40, palette="viridis")
    
    plt.title(title, fontsize=16)
    plt.xlabel("GC Content (%)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(title='Class')
    plt.show()