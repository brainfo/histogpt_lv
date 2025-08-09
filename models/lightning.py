import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC


@dataclass
class MILConfiguration:
    """Configuration for Multiple Instance Learning training"""
    warmup_steps: int
    total_steps: int
    betas: Tuple[float, float] = (0.9, 0.95)
    start_lr: float = 0.0
    max_lr: float = 5e-5
    final_lr: float = 1e-6
    weight_decay: float = 0.05
    num_classes: int = 2


class MultipleInstanceLearning(pl.LightningModule):
    """PyTorch Lightning module for Multiple Instance Learning"""
    
    def __init__(self, config: MILConfiguration, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        
        # TorchMetrics for validation - optimized for imbalanced binary classification
        from torchmetrics import MetricCollection
        
        # Primary metrics for imbalanced classification
        self.val_metrics_primary = MetricCollection({
            'auroc': AUROC(task="binary"),  # PRIMARY: Best for imbalanced classes
            'f1_macro': F1Score(task="binary", average='macro'),  # PRIMARY: Balanced F1
            'balanced_acc': Accuracy(task="binary", average='macro'),  # PRIMARY: Balanced accuracy
        })
        
        # Secondary monitoring metrics
        self.val_metrics_secondary = MetricCollection({
            'acc': Accuracy(task="binary"),  # Standard accuracy (can be misleading)
            'precision_macro': Precision(task="binary", average='macro'),
            'recall_macro': Recall(task="binary", average='macro'),
            'f1_weighted': F1Score(task="binary", average='weighted'),
        })
        
        # Per-class metrics for detailed analysis
        self.val_metrics_per_class = MetricCollection({
            'precision_per_class': Precision(task="binary", average=None),
            'recall_per_class': Recall(task="binary", average=None),
            'f1_per_class': F1Score(task="binary", average=None),
        })
        
        # Test metrics (same structure)
        self.test_metrics_primary = MetricCollection({
            'auroc': AUROC(task="binary"),
            'f1_macro': F1Score(task="binary", average='macro'),
            'balanced_acc': Accuracy(task="binary", average='macro'),
        })
        
        self.test_metrics_secondary = MetricCollection({
            'acc': Accuracy(task="binary"),
            'precision_macro': Precision(task="binary", average='macro'),
            'recall_macro': Recall(task="binary", average='macro'),
            'f1_weighted': F1Score(task="binary", average='weighted'),
        })
        
        self.test_metrics_per_class = MetricCollection({
            'precision_per_class': Precision(task="binary", average=None),
            'recall_per_class': Recall(task="binary", average=None),
            'f1_per_class': F1Score(task="binary", average=None),
        })
        
        # Store predictions for metrics calculation
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x, pos=None):
        return self.model(x, pos)
    
    def training_step(self, batch, batch_idx):
        features, coords, labels = batch
        print(f"[TRAIN] Batch {batch_idx}: features.shape={features.shape if hasattr(features, 'shape') else type(features)}")
        print(f"[TRAIN] Batch {batch_idx}: coords.shape={coords.shape if hasattr(coords, 'shape') else type(coords)}")
        print(f"[TRAIN] Batch {batch_idx}: labels.shape={labels.shape}, labels.dtype={labels.dtype}, labels={labels}")
        
        logits = self.forward(features, coords)
        print(f"[TRAIN] Batch {batch_idx}: logits.shape={logits.shape}, logits.dtype={logits.dtype}")
        print(f"[TRAIN] Batch {batch_idx}: logits={logits}")
        
        loss = self.criterion(logits, labels)
        print(f"[TRAIN] Batch {batch_idx}: loss={loss}")
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics (batch_size=1 for MIL - one slide per batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, coords, labels = batch
        print(f"[VAL] Batch {batch_idx}: features.shape={features.shape if hasattr(features, 'shape') else type(features)}")
        print(f"[VAL] Batch {batch_idx}: coords.shape={coords.shape if hasattr(coords, 'shape') else type(coords)}")
        print(f"[VAL] Batch {batch_idx}: labels.shape={labels.shape}, labels.dtype={labels.dtype}, labels={labels}")
        
        logits = self.forward(features, coords)
        print(f"[VAL] Batch {batch_idx}: logits.shape={logits.shape}, logits.dtype={logits.dtype}")
        print(f"[VAL] Batch {batch_idx}: logits={logits}")
        
        loss = self.criterion(logits, labels)
        print(f"[VAL] Batch {batch_idx}: loss={loss}")
        
        # Store outputs for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'logits': logits,
            'labels': labels
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        # Aggregate outputs
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # Calculate metrics using TorchMetrics
        probs = F.softmax(all_logits, dim=1)
        preds = torch.argmax(all_logits, dim=1)
        
        # Primary metrics (most important for imbalanced classification)
        primary_results = self.val_metrics_primary(preds, all_labels)
        # AUROC needs probabilities for positive class
        primary_results['auroc'] = self.val_metrics_primary['auroc'](probs[:, 1], all_labels)
        
        # Secondary metrics
        secondary_results = self.val_metrics_secondary(preds, all_labels)
        
        # Per-class metrics for detailed analysis
        per_class_results = self.val_metrics_per_class(preds, all_labels)
        
        # Log primary metrics with progress bar (MONITOR THESE)
        batch_size = len(self.validation_step_outputs)  # Number of slides in validation
        self.log('val_loss', avg_loss, prog_bar=True, batch_size=batch_size)
        self.log('val_auroc', primary_results['auroc'], prog_bar=True, batch_size=batch_size)  # BEST for imbalanced
        self.log('val_f1_macro', primary_results['f1_macro'], prog_bar=True, batch_size=batch_size)  # BALANCED F1
        self.log('val_balanced_acc', primary_results['balanced_acc'], prog_bar=True, batch_size=batch_size)  # BALANCED ACC
        
        # Log secondary metrics (without progress bar)
        self.log_dict({f'val_{k}': v for k, v in secondary_results.items()}, batch_size=batch_size)
        
        # Log per-class metrics with class names (handle 0-dim tensors)
        precision_per_class = per_class_results['precision_per_class']
        recall_per_class = per_class_results['recall_per_class']
        f1_per_class = per_class_results['f1_per_class']
        
        # Handle both scalar (0-dim) and vector cases
        if precision_per_class.dim() == 0:
            # Only one class present, use the scalar value for both
            self.log('val_precision_bcc', precision_per_class, batch_size=batch_size)
            self.log('val_precision_scc', precision_per_class, batch_size=batch_size)
        else:
            self.log('val_precision_bcc', precision_per_class[0], batch_size=batch_size)  # Basal cell
            self.log('val_precision_scc', precision_per_class[1], batch_size=batch_size)  # Squamous cell
            
        if recall_per_class.dim() == 0:
            self.log('val_recall_bcc', recall_per_class, batch_size=batch_size)
            self.log('val_recall_scc', recall_per_class, batch_size=batch_size)
        else:
            self.log('val_recall_bcc', recall_per_class[0], batch_size=batch_size)
            self.log('val_recall_scc', recall_per_class[1], batch_size=batch_size)
            
        if f1_per_class.dim() == 0:
            self.log('val_f1_bcc', f1_per_class, batch_size=batch_size)
            self.log('val_f1_scc', f1_per_class, batch_size=batch_size)
        else:
            self.log('val_f1_bcc', f1_per_class[0], batch_size=batch_size)
            self.log('val_f1_scc', f1_per_class[1], batch_size=batch_size)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        features, coords, labels = batch
        logits = self.forward(features, coords)
        loss = self.criterion(logits, labels)
        
        # Store outputs for epoch-end metrics
        self.test_step_outputs.append({
            'loss': loss,
            'logits': logits,
            'labels': labels
        })
        
        return loss
    
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
            
        # Aggregate outputs
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        # Calculate metrics using TorchMetrics
        probs = F.softmax(all_logits, dim=1)
        preds = torch.argmax(all_logits, dim=1)
        
        # Primary metrics
        primary_results = self.test_metrics_primary(preds, all_labels)
        primary_results['auroc'] = self.test_metrics_primary['auroc'](probs[:, 1], all_labels)
        
        # Secondary metrics
        secondary_results = self.test_metrics_secondary(preds, all_labels)
        
        # Per-class metrics
        per_class_results = self.test_metrics_per_class(preds, all_labels)
        
        # Build comprehensive metrics dict
        metrics = {
            'test_loss': avg_loss.item(),
            # Primary metrics (most important)
            'test_auroc': primary_results['auroc'].item(),
            'test_f1_macro': primary_results['f1_macro'].item(),
            'test_balanced_acc': primary_results['balanced_acc'].item(),
            # Secondary metrics
            'test_acc': secondary_results['acc'].item(),
            'test_precision_macro': secondary_results['precision_macro'].item(),
            'test_recall_macro': secondary_results['recall_macro'].item(),
            'test_f1_weighted': secondary_results['f1_weighted'].item(),
            # Per-class metrics (handle 0-dim tensors)
            'test_precision_bcc': per_class_results['precision_per_class'][0].item() if per_class_results['precision_per_class'].dim() > 0 else per_class_results['precision_per_class'].item(),
            'test_precision_scc': per_class_results['precision_per_class'][1].item() if per_class_results['precision_per_class'].dim() > 0 else per_class_results['precision_per_class'].item(),
            'test_recall_bcc': per_class_results['recall_per_class'][0].item() if per_class_results['recall_per_class'].dim() > 0 else per_class_results['recall_per_class'].item(),
            'test_recall_scc': per_class_results['recall_per_class'][1].item() if per_class_results['recall_per_class'].dim() > 0 else per_class_results['recall_per_class'].item(),
            'test_f1_bcc': per_class_results['f1_per_class'][0].item() if per_class_results['f1_per_class'].dim() > 0 else per_class_results['f1_per_class'].item(),
            'test_f1_scc': per_class_results['f1_per_class'][1].item() if per_class_results['f1_per_class'].dim() > 0 else per_class_results['f1_per_class'].item(),
        }
        
        # Log all metrics
        test_batch_size = len(self.test_step_outputs)  # Number of slides in test
        for key, value in metrics.items():
            self.log(key, value, batch_size=test_batch_size)
        
        # Clear outputs
        self.test_step_outputs.clear()
        
        return metrics
    
    def configure_optimizers(self):
        """Configure optimizer with cosine annealing schedule"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.max_lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine annealing with warmup
        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                # Linear warmup
                return current_step / self.config.warmup_steps
            else:
                # Cosine annealing
                progress = (current_step - self.config.warmup_steps) / (
                    self.config.total_steps - self.config.warmup_steps
                )
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }