import torch
import timm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from datasets import DatasetDict
from typing import Dict, Any



class DatasetProcessor:
    def __init__(self, model_name: str, in_channels: int = 4, image_column_name: str = "image"):
        self.model_name = model_name
        self.in_channels = in_channels
        self.image_column_name = image_column_name

        config = timm.data.resolve_model_data_config({'model_name': model_name})
        self.input_size = config['input_size']
        mean = config['mean']
        std = config['std']

        self.mean = list(mean) + [0.5] * (in_channels - len(mean))
        self.std = list(std) + [0.5] * (in_channels - len(std))

        self._create_transforms()


    def _create_transforms(self):
        # Since we're working with tensor data directly (not PIL images), 
        # we use transforms that work with tensors
        common_transforms = [T.Normalize(mean=self.mean, std=self.std)]
        
        self.train_transforms = T.Compose([
            T.RandomResizedCrop(self.input_size[-1], scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            *common_transforms
        ])
        
        self.val_transforms = T.Compose([
            T.Resize(self.input_size[-1], interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(self.input_size[-1]),
            *common_transforms
        ])

    def process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        # Capture the image column name and transforms for use in nested functions
        image_column_name = self.image_column_name
        train_transforms = self.train_transforms
        
        def _preprocess(examples):
            # The image data is already in numeric format as a list of 4 channels (matrices)
            # Shape: [batch_size, channels=4, height=16, width=16]
            image_batch = examples[image_column_name]
            
            # Convert list format to tensor format and apply transforms
            transformed_images = []
            for img_data in image_batch:
                # Convert from list of lists to tensor: [4, 16, 16]
                img_tensor = torch.tensor(img_data, dtype=torch.float32)
                
                # Apply transforms (which expect tensor input)
                # Note: transforms like RandomResizedCrop work on tensors
                transformed_img = train_transforms(img_tensor)
                transformed_images.append(transformed_img)
            
            examples['pixel_values'] = transformed_images
            return examples

        # Apply transform to the dataset
        dataset.set_transform(_preprocess, output_all_columns=True)

        return dataset