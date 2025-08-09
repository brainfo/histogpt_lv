#!/usr/bin/env python3
"""
Training script for MIL cancer classifier with pretrained HistoGPT weights
STRICTLY OFFLINE - No runtime downloads
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datasets.datasets import MILDataset, mil_collate_fn
from utils.pretrained_loader import load_pretrained_histogpt_weights
from ..models.aggregator import CancerClassifier, OriginalAggregator
from ..models.perceiver import FlashPerceiver
from ..models.lightning import MultipleInstanceLearning, MILConfiguration
    
import logging
import argparse
from pathlib import Path
import json

def setup_offline_environment():
    """Setup environment for offline training"""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info("Offline environment configured")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mil_training.log')
    ]
)
logger = logging.getLogger(__name__)


def verify_offline_requirements(histogpt_weights_path: str, biogpt_path: str) -> bool:
    """Verify all required files exist for offline training"""
    required_files = [
        (histogpt_weights_path, "HistoGPT weights"),
        (biogpt_path, "BioGPT model directory"),
        (os.path.join(biogpt_path, "config.json"), "BioGPT config"),
        (os.path.join(biogpt_path, "pytorch_model.bin"), "BioGPT model"),
        (os.path.join(biogpt_path, "tokenizer_config.json"), "BioGPT tokenizer")
    ]
    
    all_exist = True
    for file_path, description in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Missing {description}: {file_path}")
            all_exist = False
        else:
            logger.info(f"Found {description}: {file_path}")
    
    return all_exist

def create_model_with_pretrained_weights(histogpt_weights_path: str, biogpt_path: str):
    """Create model and load pretrained HistoGPT weights (offline only)"""
    d_input = 1024
    d_model = 1536
    
    # Create classifier for cancer detection
    classifier = CancerClassifier(d_input=d_input, d_model=d_model, num_cls=2)
    
    # Load pretrained weights with offline BioGPT path
    success = load_pretrained_histogpt_weights(
        classifier, histogpt_weights_path, biogpt_path
    )
    if not success:
        logger.warning("Failed to load pretrained weights, training from scratch")
    return classifier


def mil_collate_fn_adapted(batch):
    """Adapted collate function for the new model interface"""
    feats, coords, labels = mil_collate_fn(batch)
    
    # The Aggregator expects lists of tensors
    # Convert single tensors to list format: (N, 1024) -> [tensor(N, 1024)]
    if isinstance(feats, torch.Tensor):
        feats = [feats]
        coords = [coords]
    
    return feats, coords, labels


def train_fold(fold_idx: int, train_indices: list, valid_indices: list, 
               dataset: MILDataset, histogpt_weights_path: str, 
               biogpt_path: str, config: dict) -> dict:
    """Train a single fold (offline only)"""
    print(f"\n=== Training Fold {fold_idx + 1}/{config['k_folds']} ===")
    
    # Create data loaders
    trainset = Subset(dataset, train_indices)
    validset = Subset(dataset, valid_indices)
    
    trainloader = DataLoader(
        trainset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=mil_collate_fn_adapted,
    )
    validloader = DataLoader(
        validset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=mil_collate_fn_adapted,
    )
    
    # Create model with pretrained weights (offline)
    model = create_model_with_pretrained_weights(histogpt_weights_path, biogpt_path)
    
    # Create Lightning module
    lightning_module = MultipleInstanceLearning(
        MILConfiguration(
            warmup_steps=len(trainloader) * config['warmup_epochs'],
            total_steps=len(trainloader) * config['max_epochs'],
            betas=config['betas'],
            start_lr=config['start_lr'],
            max_lr=config['max_lr'],
            final_lr=config['final_lr'],
            weight_decay=config['weight_decay'],
            num_classes=config['num_classes'],
        ),
        model,
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auroc',
        mode='max',
        save_top_k=1,
        save_last=True,
        filename=f'fold_{fold_idx + 1}_best_{{epoch}}_{{val_auroc:.3f}}',
        dirpath=f'./checkpoints/fold_{fold_idx + 1}',
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_auroc',
        patience=10,
        mode='max',
        min_delta=0.001,
        verbose=True
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir='./logs',
        name=f'fold_{fold_idx + 1}',
        version=None
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        precision=config['precision'],
        gradient_clip_val=config['gradient_clip_val'],
        max_epochs=config['max_epochs'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
        enable_model_summary=False,
        enable_progress_bar=True,
    )
    
    # Train and validate
    trainer.fit(lightning_module, trainloader, validloader)
    
    # Load best model for final evaluation
    best_model = MultipleInstanceLearning.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        config=lightning_module.config,
        model=model
    )
    
    # Final validation on best model
    results = trainer.validate(best_model, validloader)
    
    return results[0]


def run_cross_validation(feat_path: str, histogpt_weights_path: str, 
                        biogpt_path: str, config: dict) -> dict:
    """Run k-fold cross-validation (offline only)"""
    logger.info(f"Starting offline cross-validation with {config['k_folds']} folds")
    
    # Initialize dataset
    dataset = MILDataset(feat_path, max_patches_per_slide=config['max_patches_per_slide'])
    
    # Get labels for stratification
    labels = np.array(dataset.label_list)
    
    # Print dataset info
    logger.info(f"Dataset: {len(dataset)} samples")
    logger.info(f"Class distribution: {np.bincount(labels)}")
    
    # Verify we have enough samples for k-fold CV
    min_class_count = np.min(np.bincount(labels))
    if min_class_count < config['k_folds']:
        logger.warning(f"Minimum class count ({min_class_count}) < k_folds ({config['k_folds']})")
        logger.warning("Consider reducing k_folds or using stratified sampling")
    
    # Stratified K-Fold split
    skf = StratifiedKFold(n_splits=config['k_folds'], shuffle=True, random_state=config['seed'])
    folds = list(skf.split(np.arange(len(dataset)), labels))
    
    # Print fold info
    for fold_idx, (train_indices, valid_indices) in enumerate(folds):
        print(f"Fold {fold_idx + 1}: Train={len(train_indices)}, Valid={len(valid_indices)}")
        print(f"  Train class distribution: {np.bincount(labels[train_indices])}")
        print(f"  Valid class distribution: {np.bincount(labels[valid_indices])}")
    
    # Run cross-validation
    cv_results = []
    for fold_idx, (train_indices, valid_indices) in enumerate(folds):
        fold_results = train_fold(
            fold_idx, train_indices, valid_indices, dataset, 
            histogpt_weights_path, biogpt_path, config
        )
        cv_results.append(fold_results)
        logger.info(f"Fold {fold_idx + 1} Results: {fold_results}")
    
    # Calculate summary statistics
    summary = {}
    if cv_results:
        metrics = list(cv_results[0].keys())
        for metric in metrics:
            values = [fold[metric] for fold in cv_results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
    
    return {
        'cv_results': cv_results,
        'summary': summary,
        'dataset_info': {
            'total_samples': len(dataset),
            'class_distribution': np.bincount(labels).tolist()
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train MIL cancer classifier (OFFLINE ONLY)')
    parser.add_argument('--feat_path', type=str, required=True,
                        help='Path to directory containing .h5 feature files')
    parser.add_argument('--histogpt_weights', type=str, required=True,
                        help='Path to pretrained HistoGPT weights')
    parser.add_argument('--biogpt_path', type=str, default='../microsoft_biogpt-large',
                        help='Path to local BioGPT model directory')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs per fold')
    parser.add_argument('--max_lr', type=float, default=5e-5,
                        help='Maximum learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--max_patches_per_slide', type=int, default=1000,
                        help='Maximum patches per slide')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--check_setup', action='store_true',
                        help='Check offline setup before training')
    
    args = parser.parse_args()
    
    # Setup offline environment
    setup_offline_environment()
    
    # Check setup if requested
    if args.check_setup:
        logger.info("Checking offline setup...")
        if not verify_offline_requirements(args.histogpt_weights, args.biogpt_path):
            logger.error("Offline setup check failed")
            sys.exit(1)
        logger.info("Offline setup check passed")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = {
        'k_folds': args.k_folds,
        'seed': args.seed,
        'max_patches_per_slide': args.max_patches_per_slide,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'num_classes': 2,
        'max_epochs': args.max_epochs,
        'warmup_epochs': 5,
        'betas': (0.9, 0.95),
        'start_lr': 0.0,
        'max_lr': args.max_lr,
        'final_lr': 1e-6,
        'weight_decay': 0.05,
        'accelerator': 'gpu',
        'devices': 1,
        'precision': 'bf16-mixed',
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 16,
    }
    
    # Verify offline requirements
    if not verify_offline_requirements(args.histogpt_weights, args.biogpt_path):
        logger.error("Required files missing for offline training")
        logger.error("Run with --check_setup to see what's missing")
        sys.exit(1)
    
    # Run cross-validation
    results = run_cross_validation(
        args.feat_path, args.histogpt_weights, args.biogpt_path, config
    )
    
    # Print results
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    print(f"Dataset: {results['dataset_info']['total_samples']} samples")
    print(f"Class distribution: {results['dataset_info']['class_distribution']}")
    
    print(f"\nFold Results:")
    for i, fold_result in enumerate(results['cv_results']):
        print(f"Fold {i+1}: {fold_result}")
    
    print(f"\nSummary Statistics:")
    for metric, value in results['summary'].items():
        print(f"{metric}: {value:.4f}")
    
    # Save results with timestamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'cv_results_{timestamp}.json'
    
    # Add metadata to results
    results['metadata'] = {
        'timestamp': timestamp,
        'args': vars(args),
        'config': config
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    print(f"\nResults saved to {results_file}")
    
    # Save best model from cross-validation
    print("\n" + "="*60)
    print("BEST MODEL SELECTION")
    print("="*60)
    
    # Find best fold based on validation AUROC
    best_fold_idx = 0
    best_auroc = 0
    for i, fold_result in enumerate(results['cv_results']):
        if 'val_auroc' in fold_result and fold_result['val_auroc'] > best_auroc:
            best_auroc = fold_result['val_auroc']
            best_fold_idx = i
    
    print(f"Best fold: {best_fold_idx + 1} (AUROC: {best_auroc:.4f})")
    
    # Copy best model to final location
    import shutil
    best_checkpoint_dir = f'./checkpoints/fold_{best_fold_idx + 1}'
    final_model_dir = output_dir / 'best_model'
    
    if os.path.exists(best_checkpoint_dir):
        if final_model_dir.exists():
            shutil.rmtree(final_model_dir)
        shutil.copytree(best_checkpoint_dir, final_model_dir)
        print(f"Best model saved to {final_model_dir}")
        logger.info(f"Best model from fold {best_fold_idx + 1} saved to {final_model_dir}")
    else:
        print(f"Warning: Best model checkpoint not found at {best_checkpoint_dir}")
        logger.warning(f"Best model checkpoint not found at {best_checkpoint_dir}")
    
    # Also save a summary file
    summary_file = output_dir / f'summary_{timestamp}.txt'
    with open(summary_file, 'w') as f:
        f.write("MIL Cancer Classifier Training Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: {results['dataset_info']['total_samples']} samples\n")
        f.write(f"Class distribution: {results['dataset_info']['class_distribution']}\n\n")
        f.write("Cross-Validation Results:\n")
        for metric, value in results['summary'].items():
            f.write(f"{metric}: {value:.4f}\n")
    
    logger.info(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)