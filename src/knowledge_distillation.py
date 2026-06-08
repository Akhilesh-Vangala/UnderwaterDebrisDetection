import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        student_loss = self.ce_loss(student_logits, labels)
        
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        return total_loss


class FeatureDistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor,
                student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feature_loss = self.mse_loss(student_features, teacher_features)
        ce_loss = nn.functional.cross_entropy(student_logits, labels)
        total_loss = self.alpha * feature_loss + (1 - self.alpha) * ce_loss
        return total_loss


class AttentionTransferLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_attention: torch.Tensor, teacher_attention: torch.Tensor,
                student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        attention_loss = self.mse_loss(student_attention, teacher_attention)
        ce_loss = nn.functional.cross_entropy(student_logits, labels)
        total_loss = self.alpha * attention_loss + (1 - self.alpha) * ce_loss
        return total_loss


class KnowledgeDistillationTrainer:
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 device: str = 'cuda', temperature: float = 4.0, alpha: float = 0.7):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device
        self.teacher_model.eval()
        self.criterion = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> torch.Tensor:
        images, labels = images.to(self.device), labels.to(self.device)
        
        with torch.no_grad():
            teacher_logits = self.teacher_model(images)
        
        student_logits = self.student_model(images)
        loss = self.criterion(student_logits, teacher_logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
