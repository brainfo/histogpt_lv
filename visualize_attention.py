"""
Example script for creating gradient attention visualizations.

Usage:
    python visualize_attention.py \
        --model_path /path/to/trained_model.pth \
        --h5_path /path/to/slide_features.h5 \
        --slide_image /path/to/slide_thumbnail.png \
        --output_dir ./attention_outputs \
        --slide_name "sample_slide"
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import h5py
from PIL import Image
import cv2
import sys
import os

from models.aggregator import CancerClassifier
from utils.attention_visualizer import (
    AttentionExtractor, 
    GradientAttentionVisualizer,
    load_slide_data
)

# Import the working InferenceModel class and load function
import logging
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
        with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
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
        num_cls=config['num_classes'],
        use_flash_attn=True
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


def load_slide_image(image_path: Path) -> np.ndarray:
    """Load and prepare slide thumbnail image."""
    if image_path.exists():
        image = Image.open(image_path)
        image = np.array(image.convert('RGB'))
        return image
    else:
        print(f"Warning: Slide image not found at {image_path}")
        return None


def create_patch_grid_from_h5(h5_path: Path, coordinates: np.ndarray, 
                             grid_size: tuple = (256, 256)) -> np.ndarray:
    """
    Create a grid visualization from patches stored in H5 file.
    This is a fallback when individual patch images are not available.
    """
    # Create a simple grid based on coordinates
    min_x, min_y = coordinates.min(axis=0)
    max_x, max_y = coordinates.max(axis=0)
    
    # Create coordinate grid
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


def main():
    parser = argparse.ArgumentParser(description="Create gradient attention visualizations")
    parser.add_argument("--model_path", type=Path, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--h5_path", type=Path, required=True,
                       help="Path to H5 file with features and coordinates")
    parser.add_argument("--slide_image", type=Path, 
                       help="Optional path to slide thumbnail image")
    parser.add_argument("--output_dir", type=Path, default="./attention_outputs",
                       help="Output directory for visualizations")
    parser.add_argument("--slide_name", type=str, default="slide",
                       help="Name for output files")
    parser.add_argument("--slide_width", type=int, default=20000,
                       help="Original slide width in pixels")
    parser.add_argument("--slide_height", type=int, default=15000,
                       help="Original slide height in pixels")
    parser.add_argument("--patch_size", type=int, default=512,
                       help="Size of patches used for feature extraction")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model_path}")
    model = load_inference_model(str(args.model_path))
    
    print(f"Loading slide data from {args.h5_path}")
    features, coordinates = load_slide_data(args.h5_path)
    
    print(f"Loaded {len(features)} patches with features of shape {features.shape}")
    print(f"Coordinate range: X({coordinates[:, 0].min()}-{coordinates[:, 0].max()}), "
          f"Y({coordinates[:, 1].min()}-{coordinates[:, 1].max()})")
    
    # Extract attention weights
    print("Extracting attention weights...")
    extractor = AttentionExtractor(model.model)  # Use the base model from InferenceModel wrapper
    
    # Prepare input tensors
    x = torch.tensor(features).unsqueeze(0).float()  # Add batch dimension
    pos = torch.zeros((1, features.shape[0], 2)).float()  # Dummy positions
    
    attention_data = extractor.extract_attention(x, pos)
    attention_weights = attention_data['attention_weights'][0].numpy()  # Float16 is numpy-compatible
    combined_attention = attention_data['combined_attention'][0].numpy()  # Float16 is numpy-compatible
    
    print(f"Attention weights - Mean: {attention_weights.mean():.4f}, "
          f"Max: {attention_weights.max():.4f}, Min: {attention_weights.min():.4f}")
    
    # Initialize visualizer
    visualizer = GradientAttentionVisualizer(patch_size=args.patch_size)
    
    # Create spatial heatmap
    print("Creating spatial attention heatmap...")
    slide_dims = (args.slide_width, args.slide_height)
    
    # Create heatmaps for both attention types
    attention_heatmap = visualizer.create_attention_heatmap(
        attention_weights, coordinates, slide_dims, downsample_factor=32
    )
    
    combined_heatmap = visualizer.create_attention_heatmap(
        combined_attention, coordinates, slide_dims, downsample_factor=32
    )
    
    # Load or create background image
    if args.slide_image and args.slide_image.exists():
        print(f"Loading slide image from {args.slide_image}")
        background_image = load_slide_image(args.slide_image)
    else:
        print("Creating background image from patch coordinates...")
        background_image = create_patch_grid_from_h5(args.h5_path, coordinates, 
                                                   grid_size=(1024, 768))
    
    # Create attention overlays
    print("Creating attention overlays...")
    
    # Standard attention overlay
    attention_overlay = visualizer.overlay_attention_on_image(
        background_image, attention_heatmap, alpha=0.6, colormap='jet'
    )
    
    # Combined attention overlay (with gates)
    combined_overlay = visualizer.overlay_attention_on_image(
        background_image, combined_heatmap, alpha=0.6, colormap='plasma'
    )
    
    # Create comparison plots
    print("Creating visualization plots...")
    
    # Standard attention comparison
    fig1 = visualizer.plot_attention_comparison(
        background_image, attention_overlay, attention_weights,
        title=f"Gradient Attention Analysis - {args.slide_name}",
        save_path=args.output_dir / f"{args.slide_name}_attention_comparison.png"
    )
    
    # Combined attention comparison
    fig2 = visualizer.plot_attention_comparison(
        background_image, combined_overlay, combined_attention,
        title=f"Combined Attention (with Gates) - {args.slide_name}",
        save_path=args.output_dir / f"{args.slide_name}_combined_attention.png"
    )
    
    # Save individual heatmaps
    print("Saving individual heatmaps...")
    
    # Save attention heatmap
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_heatmap, cmap='jet')
    plt.colorbar(label='Attention Weight')
    plt.title(f'Spatial Attention Heatmap - {args.slide_name}')
    plt.axis('off')
    plt.savefig(args.output_dir / f"{args.slide_name}_heatmap_only.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save combined heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(combined_heatmap, cmap='plasma')
    plt.colorbar(label='Combined Attention Weight')
    plt.title(f'Combined Attention Heatmap - {args.slide_name}')
    plt.axis('off')
    plt.savefig(args.output_dir / f"{args.slide_name}_combined_heatmap_only.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create top patches visualization if we have enough data
    if len(coordinates) > 16:
        print("Creating top patches visualization...")
        
        # Create dummy patches for visualization (in real use, load actual patch images)
        dummy_patches = []
        for i in range(min(64, len(coordinates))):
            # Create a simple colored patch based on attention weight
            patch = np.ones((64, 64, 3), dtype=np.uint8)
            intensity = int(255 * attention_weights[i] / attention_weights.max())
            patch[:, :, 0] = intensity  # Red channel based on attention
            patch[:, :, 1] = 128  # Fixed green
            patch[:, :, 2] = 255 - intensity  # Blue inverse of attention
            dummy_patches.append(patch)
        
        fig3 = visualizer.create_patch_attention_grid(
            dummy_patches, attention_weights, coordinates,
            grid_size=(8, 8),
            save_path=args.output_dir / f"{args.slide_name}_top_patches.png"
        )
    
    # Save attention statistics
    stats_file = args.output_dir / f"{args.slide_name}_attention_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Attention Statistics for {args.slide_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of patches: {len(attention_weights)}\n")
        f.write(f"Attention weights:\n")
        f.write(f"  Mean: {attention_weights.mean():.6f}\n")
        f.write(f"  Std:  {attention_weights.std():.6f}\n")
        f.write(f"  Min:  {attention_weights.min():.6f}\n")
        f.write(f"  Max:  {attention_weights.max():.6f}\n")
        f.write(f"Combined attention (with gates):\n")
        f.write(f"  Mean: {combined_attention.mean():.6f}\n")
        f.write(f"  Std:  {combined_attention.std():.6f}\n")
        f.write(f"  Min:  {combined_attention.min():.6f}\n")
        f.write(f"  Max:  {combined_attention.max():.6f}\n")
        f.write(f"\nTop 10 most attended patches (coordinates):\n")
        top_indices = np.argsort(attention_weights)[::-1][:10]
        for i, idx in enumerate(top_indices):
            f.write(f"  {i+1}. Patch at ({coordinates[idx][0]}, {coordinates[idx][1]}) "
                   f"- Attention: {attention_weights[idx]:.6f}\n")
    
    # Cleanup
    extractor.cleanup()
    
    print(f"\nVisualization complete! Files saved to: {args.output_dir}")
    print("Generated files:")
    print(f"  - {args.slide_name}_attention_comparison.png")
    print(f"  - {args.slide_name}_combined_attention.png") 
    print(f"  - {args.slide_name}_heatmap_only.png")
    print(f"  - {args.slide_name}_combined_heatmap_only.png")
    if len(coordinates) > 16:
        print(f"  - {args.slide_name}_top_patches.png")
    print(f"  - {args.slide_name}_attention_stats.txt")


if __name__ == "__main__":
    main()
