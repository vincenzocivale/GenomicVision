from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoConfig
from torch import nn

def inizialize_model(model_name: str, num_labels: int, pretrained: bool):

    if pretrained:
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        
    else:
        config = AutoConfig.from_pretrained(model_name)
        
        # 2. Aggiorna la configurazione con i dettagli del nostro task
        config.num_labels = num_labels
        
        # 3. Crea il modello dall'architettura definita nella config, con pesi casuali
        model = AutoModelForImageClassification.from_config(config)

    image_processor = AutoImageProcessor.from_pretrained(model_name)

    return model, image_processor