#!/usr/bin/env python3
"""
Single inference example using the exported inference model
Loads a self-contained inference model and runs inference on a single H5 file
"""

import os
import sys
import torch
import h5py
import numpy as np
import argparse
import logging
import json
import time
from pathlib import Path
from models.aggregator import CancerClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_offline_environment():
    """Setup environment for offline inference"""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

def load_h5_data(h5_path: str, max_patches: int = 1000):
    """Load features and coordinates from H5 file"""
    logger.info(f"Loading data from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as file:
        feats = file['features'][()]
        coords = file['coordinates'][()]
        
        # Apply patch limiting if needed
        if len(feats) > max_patches:
            logger.info(f"Limiting patches from {len(feats)} to {max_patches}")
            indices = np.random.choice(len(feats), max_patches, replace=False)
            indices = np.sort(indices)
            feats = feats[indices]
            coords = coords[indices]
        
        # Convert to tensors and add batch dimension
        feats = torch.tensor(feats, dtype=torch.bfloat16).unsqueeze(0)  # (1, N, 1024)
        coords = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
        
        logger.info(f"Loaded {feats.shape[1]} patches with feature dim {feats.shape[2]}")
        return feats, coords

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

def load_inference_model(model_path: str):
    """Load the exported inference model"""
    logger.info(f"Loading inference model from: {model_path}")
    
    # Load the saved model data
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract architecture config
    config = checkpoint['architecture_config']
    
    # Recreate the base model architecture
    base_model = CancerClassifier(
        d_input=config['d_input'],
        d_model=config['d_model'],
        num_cls=config['num_classes']
    )
    
    # Create inference wrapper
    inference_model = InferenceModel(
        base_model=base_model,
        class_names=checkpoint['class_names']
    )
    
    # Load the saved weights
    inference_model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Inference model loaded successfully")
    return inference_model

def extract_diagnosis_from_filename(filename: str) -> str:
    """Extract true diagnosis from filename for comparison"""
    disease_patterns = {
        'sbbc': 'basal cell carcinoma',
        'ibcc': 'basal cell carcinoma', 
        'pek': 'squamous cell carcinoma'
    }
    
    filename = filename.lower()
    
    # Direct pattern matching
    for file_code, diagnosis in disease_patterns.items():
        if file_code.lower() in filename:
            return diagnosis
    
    # Split by underscore and dash, then check each part
    parts = filename.replace('-', '_').split('_')
    for part in parts:
        for file_code, diagnosis in disease_patterns.items():
            if file_code.lower() in part:
                return diagnosis
    
    return "unknown"

def run_inference(model, feats, coords, device):
    """Run inference using the inference model's predict method"""
    logger.info("Running inference...")
    
    # Use the model's built-in predict method
    result = model.predict(feats, coords, device)
    
    return result['logits'], result['probabilities'], result['predicted_class']

def main():
    parser = argparse.ArgumentParser(description='Run inference with exported inference model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to exported inference model (.pth file)')
    parser.add_argument('--h5_file', type=str, required=True,
                        help='Path to H5 file for inference')
    parser.add_argument('--max_patches', type=int, default=1000,
                        help='Maximum patches to use for inference')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: auto-generated based on input filename)')
    parser.add_argument('--output_format', type=str, choices=['json', 'txt'], default='json',
                        help='Output format: json or txt (default: json)')
    
    args = parser.parse_args()
    
    # Setup offline environment
    setup_offline_environment()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Verify files exist
    if not os.path.exists(args.model):
        logger.error(f"Inference model not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.h5_file):
        logger.error(f"H5 file not found: {args.h5_file}")
        sys.exit(1)
    
    try:
        # Load data
        feats, coords = load_h5_data(args.h5_file, args.max_patches)
        
        # Load model
        model = load_inference_model(args.model)
        
        # Run inference
        logits, probs, pred_class = run_inference(model, feats, coords, device)
        
        # Extract true diagnosis from filename
        filename = Path(args.h5_file).name
        true_diagnosis = extract_diagnosis_from_filename(filename)
        
        # Class mapping
        class_names = ['basal cell carcinoma', 'squamous cell carcinoma']
        pred_diagnosis = class_names[pred_class.item()]
        
        # Check if prediction is correct
        is_correct = None
        if true_diagnosis != "unknown":
            true_class = 0 if "basal" in true_diagnosis else 1
            is_correct = (pred_class.item() == true_class)
        
        # Prepare results
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        results = {
            "timestamp": timestamp,
            "input_file": filename,
            "input_path": str(args.h5_file),
            "model_path": str(args.model),
            "number_of_patches": int(feats.shape[1]),
            "max_patches_limit": args.max_patches,
            "true_diagnosis": true_diagnosis,
            "predicted_diagnosis": pred_diagnosis,
            "predicted_class": int(pred_class.item()),
            "prediction_confidence": float(probs[0, pred_class].item()),
            "class_probabilities": {
                "basal_cell_carcinoma": float(probs[0, 0].item()),
                "squamous_cell_carcinoma": float(probs[0, 1].item())
            },
            "raw_logits": [float(x) for x in logits[0].tolist()],
            "prediction_correct": is_correct,
            "device_used": str(device)
        }
        
        # Generate output filename if not provided
        if args.output_file is None:
            input_stem = Path(args.h5_file).stem
            timestamp_short = time.strftime("%Y%m%d_%H%M%S")
            if args.output_format == 'json':
                output_file = f"inference_{input_stem}_{timestamp_short}.json"
            else:
                output_file = f"inference_{input_stem}_{timestamp_short}.txt"
        else:
            output_file = args.output_file
        
        # Write results to file
        output_path = Path(output_file)
        
        if args.output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            # Text format
            with open(output_path, 'w') as f:
                f.write("INFERENCE RESULTS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Timestamp: {results['timestamp']}\n")
                f.write(f"File: {results['input_file']}\n")
                f.write(f"Input path: {results['input_path']}\n")
                f.write(f"Model: {results['model_path']}\n")
                f.write(f"Number of patches: {results['number_of_patches']}\n")
                f.write(f"Max patches limit: {results['max_patches_limit']}\n")
                f.write(f"True diagnosis: {results['true_diagnosis']}\n")
                f.write(f"Predicted diagnosis: {results['predicted_diagnosis']}\n")
                f.write(f"Predicted class: {results['predicted_class']}\n")
                f.write(f"Prediction confidence: {results['prediction_confidence']:.4f}\n")
                f.write(f"Class probabilities:\n")
                f.write(f"  Basal cell carcinoma: {results['class_probabilities']['basal_cell_carcinoma']:.4f}\n")
                f.write(f"  Squamous cell carcinoma: {results['class_probabilities']['squamous_cell_carcinoma']:.4f}\n")
                f.write(f"Raw logits: {results['raw_logits']}\n")
                f.write(f"Prediction correct: {results['prediction_correct']}\n")
                f.write(f"Device used: {results['device_used']}\n")
                f.write("=" * 60 + "\n")
        
        logger.info(f"Results written to: {output_path}")
        print(f"Inference completed. Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()