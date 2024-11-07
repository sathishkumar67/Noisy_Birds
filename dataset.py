from __future__ import annotations

# Essential Libraries
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# Data Handling and Loading
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class Birddataset(Dataset):
    def __init__(self, image_dir: str, allowed_classes: List, dataset_type: str = None, do_transform: bool = True):
        # Initialize paths and transformation
        self.image_dir = image_dir
        self.allowed_classes = allowed_classes
        self.dataset_type = dataset_type
        self.do_transform = do_transform

        self.train_samples = []
        self.test_samples = []

        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

        # Scan directories only once
        with ThreadPoolExecutor() as executor:
            futures = []
            for class_name in os.listdir(image_dir):
                class_path = os.path.join(image_dir, class_name)
                if os.path.isdir(class_path) and (class_name in allowed_classes or class_name == "unlabeled"):
                    # Use thread pool for parallel file processing
                    futures.append(executor.submit(self.get_class_samples, class_path, class_name))

            for future in futures:
                class_samples = future.result()

                # Handle 'unlabeled' case separately
                if class_samples[0][1] == "unlabeled":
                    self.train_samples.extend(class_samples)
                else:
                    # Split train and test samples
                    random.seed(42)
                    random.shuffle(class_samples)
                    self.train_samples.extend(class_samples[:-3])
                    self.test_samples.extend(class_samples[-3:])

    
    def get_class_samples(self, class_dir: str, class_name: str) -> List[Tuple[str, str]]:
        # Lazy file reading with os.scandir, which is faster and memory efficient
        return [(os.path.join(class_dir, img_entry.name), class_name) 
                for img_entry in os.scandir(class_dir) if img_entry.is_file()]


    def __len__(self) -> int:
        """
        Returns:
            int: The total number of image in the designated dataset type.
        """
        # Return the length of the dataset (number of images) depending on the dataset type
        if self.dataset_type == "train":
            return len(self.train_samples)
        else:
            return len(self.test_samples)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, class_name = self.train_samples[index]
        
        if self.dataset_type == "train":
            image = Image.open(img_path)
        else:
            image = Image.open(img_path)
        
        if self.do_transform:
            image = self.transform(image)
        
        return image, class_name