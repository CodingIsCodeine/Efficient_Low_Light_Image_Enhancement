"""
Inference Script for ADLNet

Supports:
- Single image inference
- Batch processing
- Directory processing
- Adaptive enhancement visualization
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import time
from tqdm import tqdm

from model import ADLNet


# ============================================================================
# Preprocessing & Postprocessing
# ============================================================================

def preprocess_image(image_path, target_size=None):
    """
    Load and preprocess image for inference
    
    Args:
        image_path: Path to input image
        target_size: Optional (H, W) for resizing
    
    Returns:
        img_tensor: Preprocessed tensor [1, 3, H, W]
        original_size: Original (H, W) for resizing back
    """
    img = Image.open(image_path).convert('RGB')
    original_size = img.size[::-1]  # (H, W)
    
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Convert to tensor and normalize to [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, original_size


def postprocess_image(img_tensor, original_size=None):
    """
    Convert tensor back to PIL Image
    
    Args:
        img_tensor: Output tensor [1, 3, H, W]
        original_size: Optional (H, W) to resize back
    
    Returns:
        img: PIL Image
    """
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    
    if original_size is not None:
        img = img.resize((original_size[1], original_size[0]), Image.BILINEAR)
    
    return img


# ============================================================================
# Inference Functions
# ============================================================================

class ADLNetInference:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model weights
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = ADLNet().to(self.device)
        
        # Load weights
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")
    
    @torch.no_grad()
    def enhance_image(self, image_path, output_path=None, target_size=None):
        """
        Enhance a single image
        
        Args:
            image_path: Path to input low-light image
            output_path: Optional path to save enhanced image
            target_size: Optional (H, W) for processing
        
        Returns:
            enhanced_img: Enhanced PIL Image
            inference_time: Time taken (seconds)
        """
        # Preprocess
        img_tensor, original_size = preprocess_image(image_path, target_size)
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        enhanced_tensor = self.model(img_tensor)
        inference_time = time.time() - start_time
        
        # Postprocess
        enhanced_img = postprocess_image(enhanced_tensor, original_size)
        
        # Save if output path provided
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            enhanced_img.save(output_path)
            print(f"Saved enhanced image to {output_path}")
        
        return enhanced_img, inference_time
    
    @torch.no_grad()
    def enhance_batch(self, image_paths, output_dir, target_size=(256, 256)):
        """
        Enhance a batch of images
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save enhanced images
            target_size: Processing size (H, W)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        total_time = 0
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            img_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"enhanced_{img_name}")
            
            _, inference_time = self.enhance_image(img_path, output_path, target_size)
            total_time += inference_time
        
        avg_time = total_time / len(image_paths)
        print(f"\nBatch processing complete!")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Average inference time: {avg_time:.4f}s")
        print(f"  FPS: {1.0/avg_time:.2f}")
    
    @torch.no_grad()
    def enhance_directory(self, input_dir, output_dir, target_size=(256, 256)):
        """
        Enhance all images in a directory
        
        Args:
            input_dir: Directory containing low-light images
            output_dir: Directory to save enhanced images
            target_size: Processing size (H, W)
        """
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(input_dir).glob(f'*{ext}'))
            image_paths.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        if len(image_paths) == 0:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_paths)} images in {input_dir}")
        self.enhance_batch(image_paths, output_dir, target_size)
    
    @torch.no_grad()
    def visualize_enhancement(self, image_path, output_path=None):
        """
        Create side-by-side visualization of input and output
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization
        """
        # Load original
        original_img = Image.open(image_path).convert('RGB')
        
        # Enhance
        enhanced_img, inference_time = self.enhance_image(image_path)
        
        # Create side-by-side comparison
        w, h = original_img.size
        comparison = Image.new('RGB', (w * 2, h))
        comparison.paste(original_img, (0, 0))
        comparison.paste(enhanced_img, (w, 0))
        
        if output_path is not None:
            comparison.save(output_path)
            print(f"Saved comparison to {output_path}")
        
        print(f"Inference time: {inference_time:.4f}s")
        
        return comparison


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ADLNet Inference')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image path or directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--size', type=int, nargs=2, default=None,
                        help='Target size (H W) for processing')
    parser.add_argument('--batch', action='store_true',
                        help='Process directory of images')
    parser.add_argument('--visualize', action='store_true',
                        help='Create side-by-side visualization')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = ADLNetInference(args.model, args.device)
    
    # Process
    if args.batch:
        # Batch processing
        target_size = tuple(args.size) if args.size else (256, 256)
        inference.enhance_directory(args.input, args.output, target_size)
    else:
        # Single image
        if args.visualize:
            inference.visualize_enhancement(args.input, args.output)
        else:
            target_size = tuple(args.size) if args.size else None
            inference.enhance_image(args.input, args.output, target_size)


if __name__ == "__main__":
    # Example usage:
    # python inference.py --model checkpoints/best_model.pth --input test.jpg --output enhanced.jpg
    # python inference.py --model checkpoints/best_model.pth --input ./test_images --output ./results --batch
    # python inference.py --model checkpoints/best_model.pth --input test.jpg --output comparison.jpg --visualize
    
    main()
