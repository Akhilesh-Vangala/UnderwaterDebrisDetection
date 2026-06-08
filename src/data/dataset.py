import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import pandas as pd


class UnderwaterGarbageDataset(Dataset):
    def __init__(self, data_dir: str, annotations_file: Optional[str] = None,
                 transform: Optional[Callable] = None, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        if annotations_file:
            self.annotations = pd.read_csv(annotations_file)
        else:
            self.annotations = self._generate_synthetic_annotations()
        
        self.classes = [
            'plastic_bag', 'bottle', 'can', 'wrapper', 
            'rope', 'fishing_gear', 'tire', 'other'
        ]
        self.num_classes = len(self.classes)
    
    def _generate_synthetic_annotations(self):
        np.random.seed(42)
        n_samples = 9200 if self.split == 'train' else 1000
        
        data = []
        for i in range(n_samples):
            class_idx = np.random.randint(0, self.num_classes)
            data.append({
                'image_path': f'image_{i:05d}.jpg',
                'class_id': class_idx,
                'class_name': self.classes[class_idx]
            })
        
        return pd.DataFrame(data)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_path = self.data_dir / row['image_path']
        
        if image_path.exists():
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = self._generate_synthetic_image()
        
        label = row['class_id']
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def _generate_synthetic_image(self):
        h, w = 224, 224
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        blue_tint = np.array([100, 150, 200])
        image = (image * 0.3 + blue_tint * 0.7).astype(np.uint8)
        
        noise = np.random.normal(0, 10, (h, w, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
