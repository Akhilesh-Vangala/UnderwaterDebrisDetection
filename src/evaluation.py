import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report, cohen_kappa_score, matthews_corrcoef
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    def __init__(self, num_classes: int = 8, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [
            'plastic_bag', 'bottle', 'can', 'wrapper',
            'rope', 'fishing_gear', 'tire', 'other'
        ]
    
    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader, 
                 device: str = 'cuda', use_tta: bool = False,
                 tta_transforms: Optional[List] = None) -> Dict:
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='Evaluating'):
                images = images.to(device)
                
                if use_tta and tta_transforms:
                    tta_preds = []
                    for transform in tta_transforms:
                        aug_images = transform(images)
                        outputs = model(aug_images)
                        tta_preds.append(torch.softmax(outputs, dim=1))
                    outputs = torch.stack(tta_preds).mean(dim=0)
                    probs = outputs
                else:
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_logits.extend(outputs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_logits = np.array(all_logits)
        
        accuracy = (all_preds == all_labels).mean()
        top5_accuracy = self._calculate_topk_accuracy(all_logits, all_labels, k=5)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        ap_per_class = []
        for i in range(self.num_classes):
            y_true_class = (all_labels == i).astype(int)
            y_score_class = all_probs[:, i]
            if y_true_class.sum() > 0:
                ap = average_precision_score(y_true_class, y_score_class, average='macro')
            else:
                ap = 0.0
            ap_per_class.append(ap)
        
        mean_ap = np.mean(ap_per_class)
        
        confusion_mat = confusion_matrix(all_labels, all_preds)
        
        kappa = cohen_kappa_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        per_class_metrics = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AP': ap_per_class,
            'Support': support
        })
        
        return {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'mean_ap': mean_ap,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'kappa': kappa,
            'mcc': mcc,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_mat,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'logits': all_logits
        }
    
    def _calculate_topk_accuracy(self, logits: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
        topk_preds = np.argsort(logits, axis=1)[:, -k:]
        correct = 0
        for i, label in enumerate(labels):
            if label in topk_preds[i]:
                correct += 1
        return correct / len(labels)
    
    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        comparison = []
        for model_name, result in results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Top-5 Accuracy': result['top5_accuracy'],
                'mAP': result['mean_ap'],
                'Macro F1': result['macro_f1'],
                'Weighted F1': result['weighted_f1'],
                'Kappa': result['kappa'],
                'MCC': result['mcc']
            })
        
        df = pd.DataFrame(comparison)
        return df.sort_values('mAP', ascending=False)
    
    def plot_confusion_matrix(self, confusion_mat: np.ndarray, save_path: Optional[str] = None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_performance(self, per_class_metrics: pd.DataFrame, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].bar(per_class_metrics['Class'], per_class_metrics['Precision'])
        axes[0, 0].set_title('Precision per Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(per_class_metrics['Class'], per_class_metrics['Recall'])
        axes[0, 1].set_title('Recall per Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(per_class_metrics['Class'], per_class_metrics['F1-Score'])
        axes[1, 0].set_title('F1-Score per Class')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(per_class_metrics['Class'], per_class_metrics['AP'])
        axes[1, 1].set_title('Average Precision per Class')
        axes[1, 1].set_ylabel('AP')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_results(self, results: Dict):
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"\nOverall Accuracy: {results['accuracy']*100:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_accuracy']*100:.2f}%")
        print(f"Mean Average Precision (mAP): {results['mean_ap']*100:.2f}%")
        print(f"\nMacro Metrics:")
        print(f"  Precision: {results['macro_precision']*100:.2f}%")
        print(f"  Recall: {results['macro_recall']*100:.2f}%")
        print(f"  F1-Score: {results['macro_f1']*100:.2f}%")
        print(f"\nWeighted Metrics:")
        print(f"  Precision: {results['weighted_precision']*100:.2f}%")
        print(f"  Recall: {results['weighted_recall']*100:.2f}%")
        print(f"  F1-Score: {results['weighted_f1']*100:.2f}%")
        print(f"\nAgreement Metrics:")
        print(f"  Cohen's Kappa: {results['kappa']:.4f}")
        print(f"  Matthews Correlation Coefficient: {results['mcc']:.4f}")
        print(f"\nPer-Class Performance:")
        print(results['per_class_metrics'].to_string(index=False))
        print(f"\n{'='*70}")
    
    def generate_classification_report(self, results: Dict) -> str:
        return classification_report(
            results['labels'],
            results['predictions'],
            target_names=self.class_names,
            digits=4
        )
