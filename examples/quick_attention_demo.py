#!/usr/bin/env python3
"""
Quick demo script for gradient attention visualization.

This is a simplified version that can work with minimal setup
and demonstrates the core functionality.

Usage:
    python examples/quick_attention_demo.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.aggregator import CancerClassifier
from utils.attention_visualizer import AttentionExtractor, GradientAttentionVisualizer


def create_dummy_data(n_patches: int = 100, d_input: int = 1536):
    """Create dummy data for demonstration."""
    # Create random features
    features = np.random.randn(n_patches, d_input).astype(np.float32)
    
    # Create grid coordinates
    grid_size = int(np.sqrt(n_patches))
    x_coords = np.repeat(np.arange(grid_size), grid_size)[:n_patches] * 512
    y_coords = np.tile(np.arange(grid_size), grid_size)[:n_patches] * 512
    coordinates = np.column_stack([x_coords, y_coords])
    
    return features, coordinates


def create_dummy_slide_image(slide_dims: tuple = (5120, 5120)):
    """Create a dummy slide image for demonstration."""
    # Create a simple gradient background
    height, width = slide_dims
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient effect
    for i in range(height):
        for j in range(width):
            image[i, j, 0] = int(255 * (i / height))  # Red gradient
            image[i, j, 1] = int(255 * (j / width))   # Green gradient
            image[i, j, 2] = 200  # Fixed blue
    
    # Add some texture
    noise = np.random.randint(0, 50, (height, width, 3))
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return image


def main():
    print("HistoGPT Gradient Attention Visualization Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Create a dummy model (in practice, load your trained model)
    print("Creating model...")
    model = CancerClassifier(d_input=1536, d_model=768, num_cls=2, use_flash_attn=False)
    
    # Initialize with some reasonable weights
    torch.manual_seed(42)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    
    model.eval()
    
    # 2. Create dummy data
    print("Creating dummy data...")
    features, coordinates = create_dummy_data(n_patches=64, d_input=1536)
    
    print(f"Created {len(features)} patches")
    print(f"Feature shape: {features.shape}")
    print(f"Coordinate range: X({coordinates[:, 0].min()}-{coordinates[:, 0].max()}), "
          f"Y({coordinates[:, 1].min()}-{coordinates[:, 1].max()})")
    
    # 3. Extract attention weights
    print("Extracting attention weights...")
    extractor = AttentionExtractor(model)
    
    # Prepare input
    x = torch.tensor(features).unsqueeze(0).float()  # Add batch dimension
    pos = torch.zeros((1, features.shape[0], 2)).float()  # Dummy positions
    
    attention_data = extractor.extract_attention(x, pos)
    attention_weights = attention_data['attention_weights'][0].numpy()
    combined_attention = attention_data['combined_attention'][0].numpy()
    
    print(f"Attention stats - Mean: {attention_weights.mean():.4f}, "
          f"Max: {attention_weights.max():.4f}, Min: {attention_weights.min():.4f}")
    
    # 4. Create visualizations
    print("Creating visualizations...")
    visualizer = GradientAttentionVisualizer(patch_size=512)
    
    # Create slide dimensions based on coordinates
    slide_dims = (
        int(coordinates[:, 0].max() + 512),
        int(coordinates[:, 1].max() + 512)
    )
    
    # Create attention heatmap
    heatmap = visualizer.create_attention_heatmap(
        attention_weights, coordinates, slide_dims, downsample_factor=16
    )
    
    # Create dummy background image
    background_image = create_dummy_slide_image((heatmap.shape[0], heatmap.shape[1]))
    
    # Create overlay
    attention_overlay = visualizer.overlay_attention_on_image(
        background_image, heatmap, alpha=0.7, colormap='jet'
    )
    
    # 5. Create comparison plot
    print("Creating comparison plot...")
    fig = visualizer.plot_attention_comparison(
        background_image, attention_overlay, attention_weights,
        title="Demo: Gradient Attention for Histopathology",
        save_path=output_dir / "demo_attention_comparison.png"
    )
    
    plt.show()
    
    # 6. Create individual visualizations
    print("Creating individual heatmap...")
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar(label='Attention Weight')
    plt.title('Spatial Attention Heatmap (Demo)')
    plt.axis('off')
    plt.savefig(output_dir / "demo_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Create attention statistics plot
    print("Creating attention statistics...")
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Attention weights histogram
    plt.subplot(1, 3, 1)
    plt.hist(attention_weights, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.title('Attention Distribution')
    
    # Plot 2: Attention weights vs position
    plt.subplot(1, 3, 2)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], 
               c=attention_weights, cmap='jet', s=50)
    plt.colorbar(label='Attention')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Spatial Attention Distribution')
    
    # Plot 3: Top patches
    plt.subplot(1, 3, 3)
    sorted_indices = np.argsort(attention_weights)[::-1]
    top_10_weights = attention_weights[sorted_indices[:10]]
    plt.bar(range(10), top_10_weights, color='red', alpha=0.7)
    plt.xlabel('Patch Rank')
    plt.ylabel('Attention Weight')
    plt.title('Top 10 Attended Patches')
    
    plt.tight_layout()
    plt.savefig(output_dir / "demo_statistics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Save summary
    print("Saving summary...")
    with open(output_dir / "demo_summary.txt", 'w') as f:
        f.write("HistoGPT Attention Visualization Demo Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of patches: {len(attention_weights)}\n")
        f.write(f"Model parameters: d_input={1536}, d_model={768}\n")
        f.write(f"Slide dimensions: {slide_dims}\n")
        f.write(f"Attention statistics:\n")
        f.write(f"  Mean: {attention_weights.mean():.6f}\n")
        f.write(f"  Std:  {attention_weights.std():.6f}\n")
        f.write(f"  Min:  {attention_weights.min():.6f}\n")
        f.write(f"  Max:  {attention_weights.max():.6f}\n")
        f.write(f"\nGenerated files:\n")
        f.write(f"  - demo_attention_comparison.png\n")
        f.write(f"  - demo_heatmap.png\n")
        f.write(f"  - demo_statistics.png\n")
        f.write(f"  - demo_summary.txt\n")
    
    # Cleanup
    extractor.cleanup()
    
    print(f"\nDemo complete! Files saved to: {output_dir}")
    print("\nTo use with your real data:")
    print("1. Replace the dummy model with your trained model")
    print("2. Load real features and coordinates from your H5 files")
    print("3. Use a real slide thumbnail image")
    print("4. Run: python examples/visualize_attention.py --help for full options")


if __name__ == "__main__":
    main()
