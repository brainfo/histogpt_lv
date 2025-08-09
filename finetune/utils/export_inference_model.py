#!/usr/bin/env python3
"""
Export a self-contained inference model from training checkpoint
Combines the base model architecture, pretrained weights, and fine-tuned checkpoint
into a single optimized model for inference.
"""

import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from ..models.lightning import MultipleInstanceLearning, MILConfiguration
from ..models.aggregator import CancerClassifier
from pretrained_loader import load_pretrained_histogpt_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceModel(torch.nn.Module):
    """Self-contained inference model with optimized configuration"""
    
    def __init__(self, base_model, class_names=None):
        super().__init__()
        self.model = base_model
        self.class_names = class_names or ['basal cell carcinoma', 'squamous cell carcinoma']
        
        # Optimize for inference
        self._prepare_for_inference()
    
    def _prepare_for_inference(self):
        """Prepare model for inference by disabling training features"""
        def disable_training_features(module):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False
            module.training = False
            for child in module.children():
                disable_training_features(child)
        
        disable_training_features(self.model)
        self.model.eval()
    
    def forward(self, feats_list, coords_list):
        """Forward pass optimized for inference"""
        with torch.no_grad():
            return self.model.forward(feats_list, coords_list)
    
    def predict(self, feats, coords, device='auto'):
        """High-level prediction interface"""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Move to device and ensure proper dtype
        feats = feats.to(device, dtype=torch.bfloat16)
        coords = coords.to(device)
        self.model = self.model.to(device, dtype=torch.bfloat16)
        
        # Convert to list format expected by model
        feats_list = [feats[0]] if feats.dim() == 3 else [feats]
        coords_list = [coords[0]] if coords.dim() == 3 else [coords]
        
        # Forward pass with autocast for mixed precision
        with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
            logits = self.forward(feats_list, coords_list)
        
        # Calculate probabilities and prediction
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probs,
            'predicted_class': pred_class,
            'predicted_diagnosis': self.class_names[pred_class.item()],
            'confidence': float(probs[0, pred_class].item())
        }

def export_inference_model(checkpoint_path, histogpt_weights_path, biogpt_path, output_path):
    """Export a self-contained inference model"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Create model with same architecture as training
    d_input = 1024
    d_model = 1536
    
    # Create classifier
    classifier = CancerClassifier(d_input=d_input, d_model=d_model, num_cls=2)
    
    # Load pretrained weights
    success = load_pretrained_histogpt_weights(
        classifier, histogpt_weights_path, biogpt_path
    )
    if not success:
        logger.warning("Failed to load pretrained weights")
    
    # Create Lightning module configuration (dummy values for inference)
    config = MILConfiguration(
        warmup_steps=100,
        total_steps=1000,
        num_classes=2
    )
    
    # Load from checkpoint
    lightning_module = MultipleInstanceLearning.load_from_checkpoint(
        checkpoint_path,
        config=config,
        model=classifier
    )
    
    # Extract the underlying model
    base_model = lightning_module.model
    
    # Create inference wrapper
    inference_model = InferenceModel(
        base_model=base_model,
        class_names=['basal cell carcinoma', 'squamous cell carcinoma']
    )
    
    # Save the self-contained model
    logger.info(f"Saving inference model to: {output_path}")
    torch.save({
        'model_state_dict': inference_model.state_dict(),
        'model_class': 'InferenceModel',
        'class_names': inference_model.class_names,
        'architecture_config': {
            'd_input': d_input,
            'd_model': d_model,
            'num_classes': 2
        },
        'export_info': {
            'source_checkpoint': str(checkpoint_path),
            'histogpt_weights': str(histogpt_weights_path),
            'biogpt_path': str(biogpt_path)
        }
    }, output_path)
    
    logger.info("Export completed successfully")
    return inference_model

def main():
    parser = argparse.ArgumentParser(description='Export self-contained inference model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best model checkpoint (.ckpt file)')
    parser.add_argument('--histogpt_weights', type=str, required=True,
                        help='Path to pretrained HistoGPT weights')
    parser.add_argument('--biogpt_path', type=str, default='../microsoft_biogpt-large',
                        help='Path to local BioGPT model directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for exported model (.pth file)')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.histogpt_weights):
        logger.error(f"HistoGPT weights not found: {args.histogpt_weights}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        export_inference_model(
            args.checkpoint,
            args.histogpt_weights,
            args.biogpt_path,
            args.output
        )
        print(f"Successfully exported inference model to: {args.output}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()