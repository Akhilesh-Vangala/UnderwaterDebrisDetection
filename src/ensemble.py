import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict
from torch.utils.data import DataLoader


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None,
                 voting: str = 'soft'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.voting = voting
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(output)
        
        outputs = torch.stack(outputs, dim=0)
        
        if self.voting == 'soft':
            probs = torch.softmax(outputs, dim=-1)
            weighted_probs = torch.sum(
                probs * torch.tensor(self.weights, device=x.device).view(-1, 1, 1),
                dim=0
            )
            return torch.log(weighted_probs + 1e-8)
        else:
            preds = torch.argmax(outputs, dim=-1)
            weighted_preds = torch.sum(
                preds.float() * torch.tensor(self.weights, device=x.device).view(-1, 1),
                dim=0
            )
            return weighted_preds.unsqueeze(-1).repeat(1, outputs.shape[-1])


class StackingEnsemble(nn.Module):
    def __init__(self, base_models: List[nn.Module], meta_model: nn.Module):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_outputs = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                base_outputs.append(output)
        
        stacked_features = torch.cat(base_outputs, dim=1)
        final_output = self.meta_model(stacked_features)
        return final_output


class AdaptiveEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], num_classes: int = 8):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.gating_network = nn.Sequential(
            nn.Linear(num_classes * len(models), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(models)),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(output)
        
        stacked_outputs = torch.cat(outputs, dim=1)
        weights = self.gating_network(stacked_outputs)
        
        weighted_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            weight = weights[:, i].unsqueeze(1)
            weighted_output += weight * output
        
        return weighted_output


class EnsembleTrainer:
    def __init__(self, models: List[nn.Module], device: str = 'cuda'):
        self.models = models
        self.device = device
    
    def train_ensemble(self, train_loader: DataLoader, val_loader: DataLoader,
                      epochs: int = 10, learning_rate: float = 0.001) -> Dict:
        ensemble = EnsembleModel(self.models)
        ensemble = ensemble.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ensemble.parameters(), lr=learning_rate)
        
        best_val_acc = 0.0
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            ensemble.train()
            train_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = ensemble(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            ensemble.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = ensemble(images)
                    _, preds = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (preds == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            train_loss /= len(train_loader)
            
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        return {'ensemble': ensemble, 'history': history, 'best_val_acc': best_val_acc}
