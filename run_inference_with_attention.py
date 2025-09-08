"""
Inference with Attention Visualization
Combines the working inference from run_best_model.py with attention visualization
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
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
from models.aggregator import CancerClassifier
from utils.attention_visualizer import (
    AttentionExtractor,
    GradientAttentionVisualizer
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_offline_environment():
    """Setup environment for offline inference"""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

def load_h5_data(h5_path: str, max_patches: int = 1000):
    """Load features and coordinates from H5 file (same as run_best_model.py)"""
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
    """Self-contained inference model with optimized configuration (same as run_best_model.py)"""
    
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
    """Load the exported inference model (same as run_best_model.py)"""
    logger.info(f"Loading inference model from: {model_path}")
    
    # Load the saved model data
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract architecture config
    config = checkpoint['architecture_config']
    
    # Recreate the base model architecture
    base_model = CancerClassifier(
        d_input=config['d_input'],
        d_model=config['d_model'],
        num_cls=config['num_classes'],
        use_flash_attn=False  # Use False for compatibility
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

def create_attention_visualizations(model, feats, coords, args, results):
    """Create attention visualizations using the loaded model"""
    logger.info("Creating attention visualizations...")
    
    # Extract attention weights using the base model
    # (Everything is already on the same device from main() - simple!)
    extractor = AttentionExtractor(model.model)
    
    # Use the real coordinates (not dummy ones!)
    attention_data = extractor.extract_attention(feats, coords)
    attention_weights = attention_data['attention_weights'][0].cpu().float().numpy()  # Convert to float32 before numpy
    combined_attention = attention_data['combined_attention'][0].cpu().float().numpy()  # Convert to float32 before numpy
    
    logger.info(f"Attention weights - Mean: {attention_weights.mean():.4f}, "
               f"Max: {attention_weights.max():.4f}, Min: {attention_weights.min():.4f}")
    
    # Initialize visualizer
    visualizer = GradientAttentionVisualizer(patch_size=args.patch_size)
    
    # Create spatial heatmap
    slide_dims = (args.slide_width, args.slide_height)
    
    # Create heatmaps for both attention types
    attention_heatmap = visualizer.create_attention_heatmap(
        attention_weights, coords[0].float().numpy()[:, :2], slide_dims, downsample_factor=args.downsample_factor
    )
    
    combined_heatmap = visualizer.create_attention_heatmap(
        combined_attention, coords[0].float().numpy()[:, :2], slide_dims, downsample_factor=args.downsample_factor
    )
    
    # Load or create background image
    if args.slide_image and args.slide_image.exists():
        logger.info(f"Loading slide image from {args.slide_image}")
        background_image = load_slide_image(args.slide_image)
    else:
        logger.info("Creating simple background from coordinates...")
        background_image = create_coord_background(coords[0].float().numpy()[:, :2])
    
    # Create attention overlays
    attention_overlay = visualizer.overlay_attention_on_image(
        background_image, attention_heatmap, alpha=0.6, colormap='jet'
    )
    
    combined_overlay = visualizer.overlay_attention_on_image(
        background_image, combined_heatmap, alpha=0.6, colormap='plasma'
    )
    
    # Save visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename_stem = Path(args.h5_file).stem
    
    # Save comparison plots
    fig1 = visualizer.plot_attention_comparison(
        background_image, attention_overlay, attention_weights,
        title=f"Attention Analysis - {results['predicted_diagnosis']} (conf: {results['prediction_confidence']:.3f})",
        save_path=output_dir / f"{filename_stem}_attention_comparison.png"
    )
    
    fig2 = visualizer.plot_attention_comparison(
        background_image, combined_overlay, combined_attention,
        title=f"Combined Attention - {results['predicted_diagnosis']} (conf: {results['prediction_confidence']:.3f})",
        save_path=output_dir / f"{filename_stem}_combined_attention.png"
    )
    
    # Save attention statistics
    stats_file = output_dir / f"{filename_stem}_attention_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Attention Statistics for {filename_stem}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Prediction: {results['predicted_diagnosis']}\n")
        f.write(f"Confidence: {results['prediction_confidence']:.4f}\n")
        f.write(f"Number of patches: {len(attention_weights)}\n")
        f.write(f"Attention weights:\n")
        f.write(f"  Mean: {attention_weights.mean():.6f}\n")
        f.write(f"  Std:  {attention_weights.std():.6f}\n")
        f.write(f"  Min:  {attention_weights.min():.6f}\n")
        f.write(f"  Max:  {attention_weights.max():.6f}\n")
        f.write(f"\nTop 10 most attended patches (coordinates):\n")
        top_indices = np.argsort(attention_weights)[::-1][:10]
        coords_np = coords[0].float().numpy()
        for i, idx in enumerate(top_indices):
            f.write(f"  {i+1}. Patch at ({coords_np[idx][0]:.1f}, {coords_np[idx][1]:.1f}) "
                   f"- Attention: {attention_weights[idx]:.6f}\n")
    
    extractor.cleanup()
    logger.info(f"Attention visualizations saved to: {output_dir}")
    
    return {
        'attention_weights': attention_weights,
        'combined_attention': combined_attention,
        'output_dir': output_dir
    }

def load_slide_image(image_path: Path) -> np.ndarray:
    """Load slide thumbnail image"""
    image = Image.open(image_path)
    image = np.array(image.convert('RGB'))
    return image

def create_coord_background(coordinates: np.ndarray, grid_size: tuple = (1024, 768)) -> np.ndarray:
    """Create simple background from coordinates"""
    # Create coordinate grid
    min_x, min_y = coordinates.min(axis=0)
    max_x, max_y = coordinates.max(axis=0)
    
    grid = np.ones((grid_size[1], grid_size[0], 3), dtype=np.uint8) * 240
    
    # Normalize coordinates to grid
    if max_x > min_x and max_y > min_y:
        norm_coords = (coordinates - [min_x, min_y]) / [max_x - min_x, max_y - min_y]
        norm_coords = (norm_coords * [grid_size[0] - 1, grid_size[1] - 1]).astype(int)
        
        # Mark patch positions
        for x, y in norm_coords:
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                grid[y, x] = [100, 100, 100]  # Dark gray for patch positions
    
    return grid

def run_inference(model, feats, coords, device):
    """Run inference using the inference model's predict method"""
    logger.info("Running inference...")
    
    # Use the model's built-in predict method
    result = model.predict(feats, coords, device)
    
    return result['logits'], result['probabilities'], result['predicted_class']

def main():
    parser = argparse.ArgumentParser(description='Run inference with attention visualization')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to exported inference model (.pth file)')
    parser.add_argument('--h5_file', type=str, required=True,
                        help='Path to H5 file for inference')
    parser.add_argument('--slide_image', type=str, default=None,
                        help='Optional path to slide thumbnail image')
    parser.add_argument('--max_patches', type=int, default=1000,
                        help='Maximum patches to use for inference')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default='./attention_outputs',
                        help='Output directory for visualizations and results')
    parser.add_argument('--slide_width', type=int, default=20000,
                        help='Original slide width in pixels')
    parser.add_argument('--slide_height', type=int, default=15000,
                        help='Original slide height in pixels')
    parser.add_argument('--patch_size', type=int, default=512,
                        help='Size of patches used for feature extraction')
    parser.add_argument('--downsample_factor', type=int, default=32,
                        help='Downsample factor for attention heatmap (higher = smaller output)')
    parser.add_argument('--skip_attention', action='store_true',
                        help='Skip attention visualization (inference only)')
    
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
        # Load data (same as run_best_model.py)
        feats, coords = load_h5_data(args.h5_file, args.max_patches)
        
        # Load model (same as run_best_model.py)
        model = load_inference_model(args.model)
        
        # OPTION 1: Move everything to target device upfront (simple & reliable)
        feats = feats.to(device)
        coords = coords.to(device)
        model = model.to(device)
        
        # Run inference (same as run_best_model.py)
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
        
        # Create attention visualizations if not skipped
        if not args.skip_attention:
            if args.slide_image:
                args.slide_image = Path(args.slide_image)
            attention_results = create_attention_visualizations(model, feats, coords, args, results)
            results['attention_analysis'] = {
                'attention_mean': float(attention_results['attention_weights'].mean()),
                'attention_max': float(attention_results['attention_weights'].max()),
                'attention_std': float(attention_results['attention_weights'].std())
            }
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_stem = Path(args.h5_file).stem
        timestamp_short = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"inference_with_attention_{input_stem}_{timestamp_short}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results written to: {output_file}")
        print(f"\nInference completed successfully!")
        print(f"Prediction: {pred_diagnosis} (confidence: {results['prediction_confidence']:.3f})")
        print(f"Results saved to: {output_file}")
        if not args.skip_attention:
            print(f"Attention visualizations saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

