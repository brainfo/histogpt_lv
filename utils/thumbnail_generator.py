#!/usr/bin/env python3
"""
Slide Thumbnail Generator for .svs Files

This module provides utilities to create thumbnail images from .svs files
that can be used with the attention visualization tools.

© Manuel Tran / Helmholtz Munich
"""

import numpy as np
import slideio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Tuple, Optional, List
import argparse
import sys


class SlideThumbnailGenerator:
    """
    Generate thumbnails from whole slide images (.svs files).
    """
    
    def __init__(self):
        self.supported_formats = ['.svs', '.tiff', '.tif', '.ndpi', '.vms', '.vmu', '.scn']
    
    def get_slide_info(self, slide_path: Path) -> dict:
        """
        Get basic information about the slide.
        
        Args:
            slide_path: Path to the slide file
            
        Returns:
            Dictionary with slide information
        """
        slide = slideio.open_slide(str(slide_path), "SVS")
        
        info = {
            'num_scenes': slide.num_scenes,
            'scenes': []
        }
        
        for scene_idx in range(slide.num_scenes):
            scene = slide.get_scene(scene_idx)
            scene_info = {
                'scene_index': scene_idx,
                'size': scene.size,  # (width, height)
                'resolution': scene.resolution,  # (x_res, y_res) in pixels per meter
                'compression': getattr(scene, 'compression', 'unknown'),
                'num_channels': scene.num_channels,
                'pixel_type': str(scene.pixel_type)
            }
            info['scenes'].append(scene_info)
        
        return info
    
    def create_thumbnail(
        self,
        slide_path: Path,
        output_path: Path,
        thumbnail_size: Tuple[int, int] = (1024, 1024),
        scene_index: int = 0,
        quality: int = 95,
        add_info_overlay: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Create a thumbnail image from an .svs file.
        
        Args:
            slide_path: Path to the .svs file
            output_path: Path where to save the thumbnail
            thumbnail_size: Target size (width, height) for the thumbnail
            scene_index: Which scene to use (usually 0 for main image)
            quality: JPEG quality for saved image (1-100)
            add_info_overlay: Whether to add slide info overlay
            
        Returns:
            Tuple of (thumbnail_array, slide_info)
        """
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide file not found: {slide_path}")
        
        # Open the slide
        slide = slideio.open_slide(str(slide_path), "SVS")
        
        if scene_index >= slide.num_scenes:
            raise ValueError(f"Scene index {scene_index} not available. Slide has {slide.num_scenes} scenes.")
        
        # Get the scene
        scene = slide.get_scene(scene_index)
        original_size = scene.size  # (width, height)
        
        # Calculate scaling to fit thumbnail size while maintaining aspect ratio
        scale_x = thumbnail_size[0] / original_size[0]
        scale_y = thumbnail_size[1] / original_size[1]
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_width = int(original_size[0] * scale)
        new_height = int(original_size[1] * scale)
        
        # Read the scene at reduced resolution
        thumbnail_array = scene.read_block(size=(new_width, new_height))
        
        # Convert to PIL Image for easier manipulation
        thumbnail_pil = Image.fromarray(thumbnail_array)
        
        # Create final canvas with exact thumbnail size (centered)
        final_thumbnail = Image.new('RGB', thumbnail_size, (255, 255, 255))
        
        # Center the thumbnail
        paste_x = (thumbnail_size[0] - new_width) // 2
        paste_y = (thumbnail_size[1] - new_height) // 2
        final_thumbnail.paste(thumbnail_pil, (paste_x, paste_y))
        
        # Add info overlay if requested
        if add_info_overlay:
            final_thumbnail = self._add_info_overlay(
                final_thumbnail, slide_path.name, original_size, scene.resolution
            )
        
        # Save the thumbnail
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_thumbnail.save(str(output_path), 'PNG', quality=quality)
        
        # Collect slide info
        slide_info = {
            'original_size': original_size,
            'thumbnail_size': thumbnail_size,
            'scale_factor': scale,
            'resolution': scene.resolution,
            'scene_index': scene_index,
            'file_path': str(slide_path)
        }
        
        return np.array(final_thumbnail), slide_info
    
    def _add_info_overlay(
        self,
        image: Image.Image,
        filename: str,
        original_size: Tuple[int, int],
        resolution: Tuple[float, float]
    ) -> Image.Image:
        """
        Add an information overlay to the thumbnail.
        
        Args:
            image: PIL Image to add overlay to
            filename: Name of the original file
            original_size: Original slide dimensions
            resolution: Slide resolution
            
        Returns:
            Image with overlay
        """
        draw = ImageDraw.Draw(image)
        
        # Try to use a nice font, fall back to default if not available
        try:
            font_size = max(12, min(20, image.size[1] // 50))
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Create overlay text
        mpp_x = 1000000 / resolution[0] if resolution[0] > 0 else 0  # microns per pixel
        mpp_y = 1000000 / resolution[1] if resolution[1] > 0 else 0
        
        overlay_text = [
            f"File: {filename}",
            f"Size: {original_size[0]} x {original_size[1]}",
            f"Resolution: {mpp_x:.2f} x {mpp_y:.2f} μm/px"
        ]
        
        # Calculate text box size
        text_height = len(overlay_text) * 25
        max_text_width = max([draw.textlength(text, font=font) for text in overlay_text])
        
        # Draw semi-transparent background
        overlay_box = [10, image.size[1] - text_height - 20, max_text_width + 30, image.size[1] - 10]
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(overlay_box, fill=(0, 0, 0, 128))  # Semi-transparent black
        
        # Composite overlay
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Draw text
        y_offset = image.size[1] - text_height - 15
        for text in overlay_text:
            draw.text((15, y_offset), text, fill=(255, 255, 255), font=font)
            y_offset += 25
        
        return image
    
    def create_multi_level_thumbnails(
        self,
        slide_path: Path,
        output_dir: Path,
        sizes: List[Tuple[int, int]] = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)],
        scene_index: int = 0
    ) -> List[Tuple[Path, dict]]:
        """
        Create multiple thumbnail sizes from the same slide.
        
        Args:
            slide_path: Path to the .svs file
            output_dir: Directory to save thumbnails
            sizes: List of (width, height) tuples for different thumbnail sizes
            scene_index: Which scene to use
            
        Returns:
            List of (thumbnail_path, slide_info) tuples
        """
        results = []
        slide_name = slide_path.stem
        
        for size in sizes:
            size_str = f"{size[0]}x{size[1]}"
            output_path = output_dir / f"{slide_name}_thumbnail_{size_str}.png"
            
            thumbnail_array, slide_info = self.create_thumbnail(
                slide_path, output_path, thumbnail_size=size, 
                scene_index=scene_index, add_info_overlay=(size == max(sizes))
            )
            
            results.append((output_path, slide_info))
        
        return results
    
    def create_thumbnail_with_patch_overlay(
        self,
        slide_path: Path,
        output_path: Path,
        coordinates: np.ndarray,
        patch_size: int = 512,
        thumbnail_size: Tuple[int, int] = (1024, 1024),
        patch_color: Tuple[int, int, int] = (255, 0, 0),
        patch_alpha: float = 0.3
    ) -> np.ndarray:
        """
        Create a thumbnail with patch locations overlaid.
        
        Args:
            slide_path: Path to the .svs file
            output_path: Path to save the thumbnail
            coordinates: Array of patch coordinates (N, 2) with (x, y) positions
            patch_size: Size of patches in original slide coordinates
            thumbnail_size: Target thumbnail size
            patch_color: RGB color for patch overlay
            patch_alpha: Transparency of patch overlay
            
        Returns:
            Thumbnail array with patch overlay
        """
        # Create base thumbnail
        thumbnail_array, slide_info = self.create_thumbnail(
            slide_path, Path("temp_thumbnail.png"), thumbnail_size, add_info_overlay=False
        )
        
        # Calculate scale factor
        scale = slide_info['scale_factor']
        
        # Create overlay for patches
        overlay = np.zeros_like(thumbnail_array)
        
        # Draw patches on overlay
        for x, y in coordinates:
            # Scale coordinates to thumbnail space
            thumb_x = int(x * scale)
            thumb_y = int(y * scale)
            thumb_patch_size = max(1, int(patch_size * scale))
            
            # Adjust for centering
            paste_x = (thumbnail_size[0] - int(slide_info['original_size'][0] * scale)) // 2
            paste_y = (thumbnail_size[1] - int(slide_info['original_size'][1] * scale)) // 2
            
            thumb_x += paste_x
            thumb_y += paste_y
            
            # Draw rectangle
            x1, y1 = thumb_x, thumb_y
            x2, y2 = min(thumbnail_size[0], thumb_x + thumb_patch_size), min(thumbnail_size[1], thumb_y + thumb_patch_size)
            
            if x1 < thumbnail_size[0] and y1 < thumbnail_size[1] and x2 > 0 and y2 > 0:
                x1, y1 = max(0, x1), max(0, y1)
                overlay[y1:y2, x1:x2] = patch_color
        
        # Blend overlay with thumbnail
        final_thumbnail = cv2.addWeighted(thumbnail_array, 1 - patch_alpha, overlay, patch_alpha, 0)
        
        # Save result
        Image.fromarray(final_thumbnail).save(str(output_path))
        
        return final_thumbnail


def create_thumbnails_from_directory(
    input_dir: Path,
    output_dir: Path,
    thumbnail_size: Tuple[int, int] = (1024, 1024),
    file_pattern: str = "*.svs"
) -> List[Tuple[Path, Path, dict]]:
    """
    Create thumbnails for all slides in a directory.
    
    Args:
        input_dir: Directory containing .svs files
        output_dir: Directory to save thumbnails
        thumbnail_size: Target thumbnail size
        file_pattern: Pattern to match slide files
        
    Returns:
        List of (input_path, output_path, slide_info) tuples
    """
    generator = SlideThumbnailGenerator()
    results = []
    
    slide_files = list(input_dir.glob(file_pattern))
    print(f"Found {len(slide_files)} slides to process")
    
    for slide_path in slide_files:
        print(f"Processing {slide_path.name}...")
        
        try:
            output_path = output_dir / f"{slide_path.stem}_thumbnail.png"
            thumbnail_array, slide_info = generator.create_thumbnail(
                slide_path, output_path, thumbnail_size
            )
            results.append((slide_path, output_path, slide_info))
            print(f"  ✓ Created thumbnail: {output_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing {slide_path.name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Create thumbnails from .svs files")
    parser.add_argument("input", type=Path, help="Input .svs file or directory")
    parser.add_argument("--output", "-o", type=Path, help="Output file or directory")
    parser.add_argument("--size", nargs=2, type=int, default=[1024, 1024],
                       help="Thumbnail size (width height)")
    parser.add_argument("--scene", type=int, default=0,
                       help="Scene index to use (default: 0)")
    parser.add_argument("--quality", type=int, default=95,
                       help="JPEG quality (1-100)")
    parser.add_argument("--multi-size", action="store_true",
                       help="Create multiple thumbnail sizes")
    parser.add_argument("--no-overlay", action="store_true",
                       help="Don't add info overlay")
    parser.add_argument("--info-only", action="store_true",
                       help="Only print slide information")
    
    args = parser.parse_args()
    
    generator = SlideThumbnailGenerator()
    
    # Handle single file vs directory
    if args.input.is_file():
        # Single file processing
        if args.info_only:
            info = generator.get_slide_info(args.input)
            print(f"Slide Information for {args.input.name}:")
            print(f"  Number of scenes: {info['num_scenes']}")
            for scene in info['scenes']:
                print(f"  Scene {scene['scene_index']}:")
                print(f"    Size: {scene['size'][0]} x {scene['size'][1]}")
                print(f"    Resolution: {scene['resolution']}")
                print(f"    Channels: {scene['num_channels']}")
                print(f"    Pixel type: {scene['pixel_type']}")
            return
        
        if not args.output:
            args.output = args.input.parent / f"{args.input.stem}_thumbnail.png"
        
        if args.multi_size:
            results = generator.create_multi_level_thumbnails(
                args.input, args.output.parent, scene_index=args.scene
            )
            print(f"Created {len(results)} thumbnails:")
            for path, info in results:
                print(f"  {path}")
        else:
            thumbnail_array, slide_info = generator.create_thumbnail(
                args.input, args.output, 
                thumbnail_size=tuple(args.size),
                scene_index=args.scene,
                quality=args.quality,
                add_info_overlay=not args.no_overlay
            )
            print(f"Created thumbnail: {args.output}")
            print(f"Original size: {slide_info['original_size']}")
            print(f"Scale factor: {slide_info['scale_factor']:.4f}")
    
    elif args.input.is_dir():
        # Directory processing
        if not args.output:
            args.output = args.input / "thumbnails"
        
        results = create_thumbnails_from_directory(
            args.input, args.output, tuple(args.size)
        )
        print(f"\nProcessed {len(results)} slides successfully")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()

