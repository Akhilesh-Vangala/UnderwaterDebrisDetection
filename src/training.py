import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
import math


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class CosineAnnealingWarmRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0, last_epoch: int = -1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = epoch
        else:
            self.T_cur = epoch
        
        if epoch >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult
        
        super(CosineAnnealingWarmRestarts, self).step(epoch)


class Lookahead:
    def __init__(self, optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = {param: param.data.clone() for param in optimizer.param_groups[0]['params']}
    
    def step(self):
        self.optimizer.step()
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p in self.slow_weights:
                        self.slow_weights[p] += self.alpha * (p.data - self.slow_weights[p])
                        p.data.copy_(self.slow_weights[p])
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class ModelTrainer:
    def __init__(self, device: str = None, model_dir: str = 'outputs/models', seed: int = 42):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using device: {self.device}")
    
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, learning_rate: float = 0.001, weight_decay: float = 1e-4,
              use_mixed_precision: bool = True, use_lookahead: bool = True,
              use_focal_loss: bool = False, use_label_smoothing: bool = True,
              label_smoothing: float = 0.1, focal_gamma: float = 2.0,
              class_weights: Optional[List[float]] = None, model_name: str = 'cswin',
              warmup_epochs: int = 5, gradient_accumulation_steps: int = 1) -> Dict:
        
        model = model.to(self.device)
        
        if use_focal_loss:
            criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        elif use_label_smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(self.device) if class_weights else None)
        
        base_optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                                    betas=(0.9, 0.999), eps=1e-8)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5) if use_lookahead else base_optimizer
        
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer.optimizer if use_lookahead else optimizer,
                                                      start_factor=0.1, total_iters=warmup_epochs)
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer.optimizer if use_lookahead else optimizer,
            T_0=epochs - warmup_epochs, T_mult=2, eta_min=1e-6
        )
        
        scaler = GradScaler() if use_mixed_precision and self.device == 'cuda' else None
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        
        total_steps = len(train_loader) * epochs
        current_step = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if use_mixed_precision and scaler is not None:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels) / gradient_accumulation_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels) / gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                if use_lookahead and (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                
                train_loss += loss.item() * gradient_accumulation_steps
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                current_step += 1
                if epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()
                
                current_lr = optimizer.optimizer.param_groups[0]['lr'] if use_lookahead else optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'acc': f'{100 * train_correct / train_total:.2f}%',
                    'lr': f'{current_lr:.6f}'
                })
            
            if epoch >= warmup_epochs:
                main_scheduler.step()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            current_lr = optimizer.optimizer.param_groups[0]['lr'] if use_lookahead else optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': main_scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss
                }, self.model_dir / f'{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f'\nBest validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}')
        
        checkpoint = torch.load(self.model_dir / f'{model_name}_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return {
            'model': model,
            'history': history,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'checkpoint': checkpoint
        }
    
    def validate(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc
    
    def train_with_curriculum(self, model: nn.Module, train_loaders: List[DataLoader],
                              val_loader: DataLoader, epochs_per_stage: List[int],
                              learning_rates: List[float], **kwargs) -> Dict:
        model = model.to(self.device)
        all_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for stage, (train_loader, epochs, lr) in enumerate(zip(train_loaders, epochs_per_stage, learning_rates)):
            print(f"\nCurriculum Stage {stage + 1}/{len(train_loaders)}")
            result = self.train(model, train_loader, val_loader, epochs=epochs,
                              learning_rate=lr, model_name=f'cswin_stage_{stage}', **kwargs)
            
            for key in all_history:
                all_history[key].extend(result['history'][key])
        
        return {'model': model, 'history': all_history}
