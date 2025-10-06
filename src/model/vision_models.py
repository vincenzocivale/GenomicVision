import torch
import timm
from typing import Optional
import torch


def get_vision_model(
    model_name: str,
    num_classes: int,
    in_channels: int = 4,
    pretrained: bool = True
) -> torch.nn.Module:
    """
    Istanzia un modello di computer vision da `timm` adattandolo per un numero
    personalizzato di canali di input e classi di output.

    La libreria `timm` gestisce automaticamente l'adattamento del primo layer convoluzionale
    per accettare un numero di canali diverso da 3, anche quando si usano pesi pre-addestrati.

    Args:
        model_name (str): Nome del modello da caricare (es. 'resnet34', 'efficientnet_b0').
        num_classes (int): Numero di classi finali per il classificatore.
        in_channels (int, optional): Numero di canali dell'immagine di input. Default: 4.
        pretrained (bool, optional): Se True, carica i pesi pre-addestrati su ImageNet.
                                    `timm` adatterà il primo layer per i canali extra. Default: True.

    Returns:
        torch.nn.Module: Il modello PyTorch pronto per l'addestramento.
    """

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=num_classes
    )

    return model