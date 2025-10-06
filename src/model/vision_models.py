from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models
from enum import Enum


class VisionModelType(Enum):
    """Enumerazione dei tipi di modelli disponibili."""
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    VIT_B_16 = "vit_b_16"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    DENSENET121 = "densenet121"
    CUSTOM_CNN = "custom_cnn"


class BaseVisionModel(nn.Module, ABC):
    """Classe base astratta per tutti i modelli di visione."""
    
    def __init__(
        self, 
        num_classes: int,
        input_channels: int = 4,
        pretrained: bool = False,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Numero di classi per la classificazione
            input_channels: Numero di canali in input (4 per le basi genomiche)
            pretrained: Se usare pesi pre-addestrati (dove disponibile)
            freeze_backbone: Se congelare i pesi del backbone
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model = None
        
    @abstractmethod
    def build(self) -> nn.Module:
        """Costruisce e restituisce il modello PyTorch."""
        pass
    
    def forward(self, pixel_values, labels=None):
        """Forward pass compatibile con HF Trainer."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")
        
        logits = self.model(pixel_values)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {'loss': loss, 'logits': logits}
    
    def _adapt_first_conv(self, model: nn.Module, first_conv_name: str) -> nn.Module:
        """
        Adatta il primo layer convoluzionale per accettare input_channels canali.
        
        Args:
            model: Il modello da modificare
            first_conv_name: Nome del primo layer convoluzionale
        """
        if self.input_channels != 3:
            # Ottieni il primo conv layer
            first_conv = dict(model.named_modules())[first_conv_name]
            
            # Crea nuovo conv layer con il numero corretto di canali
            new_conv = nn.Conv2d(
                self.input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # Se pretrained, inizializza con media dei pesi RGB
            if self.pretrained and self.input_channels == 4:
                with torch.no_grad():
                    # Media dei pesi RGB per i primi 3 canali
                    new_conv.weight[:, :3, :, :] = first_conv.weight
                    # Replica la media per il quarto canale
                    new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
            
            # Sostituisci il layer
            parent_module = model
            name_parts = first_conv_name.split('.')
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], new_conv)
        
        return model
    
    def _freeze_layers(self, model: nn.Module, classifier_name: str) -> nn.Module:
        """Congela tutti i layer tranne il classificatore finale."""
        if self.freeze_backbone:
            for name, param in model.named_parameters():
                if classifier_name not in name:
                    param.requires_grad = False
        return model


class ResNetModel(BaseVisionModel):
    """Wrapper per architetture ResNet."""
    
    def __init__(self, variant: str = "resnet18", **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        
    def build(self) -> nn.Module:
        # Seleziona la variante
        if self.variant == "resnet18":
            self.model = models.resnet18(weights="DEFAULT" if self.pretrained else None)
        elif self.variant == "resnet34":
            self.model = models.resnet34(weights="DEFAULT" if self.pretrained else None)
        elif self.variant == "resnet50":
            self.model = models.resnet50(weights="DEFAULT" if self.pretrained else None)
        else:
            raise ValueError(f"Variante ResNet non supportata: {self.variant}")
        
        # Adatta primo conv layer
        self.model = self._adapt_first_conv(self.model, "conv1")
        
        # Modifica classificatore finale
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        
        # Congela backbone se richiesto
        self.model = self._freeze_layers(self.model, "fc")
        
        return self


class EfficientNetModel(BaseVisionModel):
    """Wrapper per architetture EfficientNet."""
    
    def __init__(self, variant: str = "b0", **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        
    def build(self) -> nn.Module:
        if self.variant == "b0":
            self.model = models.efficientnet_b0(weights="DEFAULT" if self.pretrained else None)
        elif self.variant == "b1":
            self.model = models.efficientnet_b1(weights="DEFAULT" if self.pretrained else None)
        else:
            raise ValueError(f"Variante EfficientNet non supportata: {self.variant}")
        
        # Adatta primo conv layer
        self.model = self._adapt_first_conv(self.model, "features.0.0")
        
        # Modifica classificatore
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        self.model = self._freeze_layers(self.model, "classifier")
        
        return self


class ViTModel(BaseVisionModel):
    """Wrapper per Vision Transformer."""
    
    def build(self) -> nn.Module:
        self.model = models.vit_b_16(weights="DEFAULT" if self.pretrained else None)
        
        # Adatta patch embedding per 4 canali
        self.model = self._adapt_first_conv(self.model, "conv_proj")
        
        # Modifica head
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, self.num_classes)
        
        self.model = self._freeze_layers(self.model, "heads")
        
        return self


class MobileNetModel(BaseVisionModel):
    """Wrapper per MobileNetV3."""
    
    def __init__(self, variant: str = "small", **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        
    def build(self) -> nn.Module:
        if self.variant == "small":
            model = models.mobilenet_v3_small(weights="DEFAULT" if self.pretrained else None)
        elif self.variant == "large":
            model = models.mobilenet_v3_large(weights="DEFAULT" if self.pretrained else None)
        else:
            raise ValueError(f"Variante MobileNet non supportata: {self.variant}")
        
        model = self._adapt_first_conv(model, "features.0.0")
        
        # Modifica classificatore
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, self.num_classes)
        
        model = self._freeze_layers(model, "classifier")
        
        return model


class DenseNetModel(BaseVisionModel):
    """Wrapper per DenseNet."""
    
    def __init__(self, variant: str = "121", **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        
    def build(self) -> nn.Module:
        if self.variant == "121":
            model = models.densenet121(weights="DEFAULT" if self.pretrained else None)
        else:
            raise ValueError(f"Variante DenseNet non supportata: {self.variant}")
        
        model = self._adapt_first_conv(model, "features.conv0")
        
        # Modifica classificatore
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, self.num_classes)
        
        model = self._freeze_layers(model, "classifier")
        
        return model


class CustomCNNModel(BaseVisionModel):
    """CNN personalizzata per sequenze genomiche 16x16x4."""
    
    def build(self) -> nn.Module:
        return nn.Sequential(
            # Block 1
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 2x2
            
            # Classifier
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )


class VisionModelFactory:
    """Factory per la creazione di modelli di visione."""
    
    _builders = {
        VisionModelType.RESNET18: lambda **kwargs: ResNetModel(variant="resnet18", **kwargs),
        VisionModelType.RESNET34: lambda **kwargs: ResNetModel(variant="resnet34", **kwargs),
        VisionModelType.RESNET50: lambda **kwargs: ResNetModel(variant="resnet50", **kwargs),
        VisionModelType.EFFICIENTNET_B0: lambda **kwargs: EfficientNetModel(variant="b0", **kwargs),
        VisionModelType.EFFICIENTNET_B1: lambda **kwargs: EfficientNetModel(variant="b1", **kwargs),
        VisionModelType.VIT_B_16: lambda **kwargs: ViTModel(**kwargs),
        VisionModelType.MOBILENET_V3_SMALL: lambda **kwargs: MobileNetModel(variant="small", **kwargs),
        VisionModelType.MOBILENET_V3_LARGE: lambda **kwargs: MobileNetModel(variant="large", **kwargs),
        VisionModelType.DENSENET121: lambda **kwargs: DenseNetModel(variant="121", **kwargs),
        VisionModelType.CUSTOM_CNN: lambda **kwargs: CustomCNNModel(**kwargs),
    }
    
    @classmethod
    def create(
        cls,
        model_type: VisionModelType,
        num_classes: int,
        input_channels: int = 4,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> nn.Module:
        """
        Crea e inizializza un modello di visione.
        
        Args:
            model_type: Tipo di modello da creare
            num_classes: Numero di classi per la classificazione
            input_channels: Numero di canali in input (default: 4)
            pretrained: Se usare pesi pre-addestrati
            freeze_backbone: Se congelare il backbone
            device: Device su cui caricare il modello
            
        Returns:
            Modello PyTorch pronto per il training/inference
            
        Example:
            >>> factory = VisionModelFactory()
            >>> model = factory.create(
            ...     model_type=VisionModelType.RESNET18,
            ...     num_classes=10,
            ...     pretrained=True
            ... )
        """
        if model_type not in cls._builders:
            raise ValueError(f"Tipo di modello non supportato: {model_type}")
        
        builder = cls._builders[model_type](
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        model = builder.build()
        model = model.to(device)
        
        return model
    
    @classmethod
    def get_available_models(cls) -> list:
        """Restituisce la lista dei modelli disponibili."""
        return [model_type.value for model_type in VisionModelType]
    
    @classmethod
    def get_model_info(cls, model_type: VisionModelType) -> Dict[str, Any]:
        """Restituisce informazioni sul modello."""
        info = {
            VisionModelType.RESNET18: {
                "parameters": "~11M",
                "description": "ResNet-18, architettura residuale leggera"
            },
            VisionModelType.RESNET50: {
                "parameters": "~25M",
                "description": "ResNet-50, architettura residuale profonda"
            },
            VisionModelType.EFFICIENTNET_B0: {
                "parameters": "~5M",
                "description": "EfficientNet-B0, ottimizzato per efficienza"
            },
            VisionModelType.VIT_B_16: {
                "parameters": "~86M",
                "description": "Vision Transformer, architettura basata su attention"
            },
            VisionModelType.MOBILENET_V3_SMALL: {
                "parameters": "~2.5M",
                "description": "MobileNetV3 Small, molto leggero"
            },
            VisionModelType.CUSTOM_CNN: {
                "parameters": "~0.5M",
                "description": "CNN personalizzata per sequenze genomiche"
            },
        }
        return info.get(model_type, {"parameters": "N/A", "description": "N/A"})