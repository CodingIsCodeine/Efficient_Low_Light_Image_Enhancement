"""
Utilities for ADLNet

Includes:
- Preprocessing helpers
- Evaluation metrics (SSIM, PSNR, etc.)
- No-reference quality assessment stubs
- Visualization tools
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


# ============================================================================
# Color Space Conversions
# ============================================================================

def rgb_to_yuv_numpy(rgb):
    """Convert RGB to YUV (numpy version)"""
    rgb = rgb.astype(np.float32)
    y = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    u = -0.14713 * rgb[:, :, 0] - 0.28886 * rgb[:, :, 1] + 0.436 * rgb[:, :, 2]
    v = 0.615 * rgb[:, :, 0] - 0.51499 * rgb[:, :, 1] - 0.10001 * rgb[:, :, 2]
    return np.stack([y, u, v], axis=-1)


def yuv_to_rgb_numpy(yuv):
    """Convert YUV to RGB (numpy version)"""
    yuv = yuv.astype(np.float32)
    r = yuv[:, :, 0] + 1.13983 * yuv[:, :, 2]
    g = yuv[:, :, 0] - 0.39465 * yuv[:, :, 1] - 0.58060 * yuv[:, :, 2]
    b = yuv[:, :, 0] + 2.03211 * yuv[:, :, 1]
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ============================================================================
# Image Quality Metrics
# ============================================================================

def calculate_psnr(img1, img2, max_value=255.0):
    """
    Calculate PSNR between two images
    
    Args:
        img1, img2: numpy arrays [H, W, C] or [H, W]
        max_value: Maximum pixel value (255 for uint8)
    
    Returns:
        psnr_value: PSNR in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    return psnr(img1, img2, data_range=max_value)


def calculate_ssim(img1, img2, max_value=255.0):
    """
    Calculate SSIM between two images
    
    Args:
        img1, img2: numpy arrays [H, W, C] or [H, W]
        max_value: Maximum pixel value (255 for uint8)
    
    Returns:
        ssim_value: SSIM score [0, 1]
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    # Handle multi-channel images
    if img1.ndim == 3:
        return ssim(img1, img2, data_range=max_value, channel_axis=2)
    else:
        return ssim(img1, img2, data_range=max_value)


def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))


# ============================================================================
# No-Reference Quality Metrics (Stubs)
# ============================================================================

class NoReferenceMetrics:
    """
    Stubs for no-reference quality metrics
    
    In practice, these would use pretrained models:
    - LIQE: Learning-based Image Quality Evaluator
    - MUSIQ: Multi-Scale Image Quality Transformer
    - Q-Align: Quality-Aware Alignment
    
    For competition, you'd integrate actual implementations.
    """
    
    @staticmethod
    def estimate_naturalness(img):
        """
        Estimate image naturalness (no-reference)
        
        Simple proxy: variance of local statistics
        Higher is generally better for enhanced images
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)
        
        # Local variance as naturalness proxy
        kernel_size = 7
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        variance = sqr_mean - mean ** 2
        
        return np.mean(variance)
    
    @staticmethod
    def estimate_sharpness(img):
        """
        Estimate image sharpness (no-reference)
        
        Using Laplacian variance
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        if img.ndim == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    @staticmethod
    def estimate_colorfulness(img):
        """
        Estimate image colorfulness
        
        Based on opponent color space statistics
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        if img.ndim != 3:
            return 0.0
        
        # Opponent color space
        rg = img[:, :, 0] - img[:, :, 1]
        yb = 0.5 * (img[:, :, 0] + img[:, :, 1]) - img[:, :, 2]
        
        # Standard deviation and mean
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        
        std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        
        return std_root + 0.3 * mean_root


# ============================================================================
# Evaluation Suite
# ============================================================================

def evaluate_enhancement(pred_img, gt_img, verbose=True):
    """
    Comprehensive evaluation of enhancement quality
    
    Args:
        pred_img: Enhanced image (numpy array or tensor)
        gt_img: Ground truth image (numpy array or tensor)
        verbose: Print results
    
    Returns:
        metrics: Dictionary of metric scores
    """
    # Convert to numpy if needed
    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.cpu().numpy()
    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.cpu().numpy()
    
    # Ensure [0, 255] range
    if pred_img.max() <= 1.0:
        pred_img = (pred_img * 255).astype(np.uint8)
    if gt_img.max() <= 1.0:
        gt_img = (gt_img * 255).astype(np.uint8)
    
    # Calculate metrics
    metrics = {
        'psnr': calculate_psnr(pred_img, gt_img),
        'ssim': calculate_ssim(pred_img, gt_img),
        'mae': calculate_mae(pred_img, gt_img),
        'naturalness': NoReferenceMetrics.estimate_naturalness(pred_img / 255.0),
        'sharpness': NoReferenceMetrics.estimate_sharpness(pred_img / 255.0),
        'colorfulness': NoReferenceMetrics.estimate_colorfulness(pred_img / 255.0),
    }
    
    if verbose:
        print("Enhancement Quality Metrics:")
        print("="*50)
        print(f"  PSNR:          {metrics['psnr']:.2f} dB")
        print(f"  SSIM:          {metrics['ssim']:.4f}")
        print(f"  MAE:           {metrics['mae']:.2f}")
        print(f"  Naturalness:   {metrics['naturalness']:.2f}")
        print(f"  Sharpness:     {metrics['sharpness']:.2f}")
        print(f"  Colorfulness:  {metrics['colorfulness']:.2f}")
        print("="*50)
    
    return metrics


# ============================================================================
# Visualization Tools
# ============================================================================

def visualize_yuv_decomposition(rgb_img, save_path=None):
    """
    Visualize YUV decomposition of an image
    
    Args:
        rgb_img: RGB image [H, W, 3], range [0, 1] or [0, 255]
        save_path: Optional path to save visualization
    """
    if isinstance(rgb_img, torch.Tensor):
        rgb_img = rgb_img.cpu().numpy()
    
    if rgb_img.max() <= 1.0:
        rgb_img = (rgb_img * 255).astype(np.uint8)
    
    # Convert to YUV
    yuv_img = rgb_to_yuv_numpy(rgb_img)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original RGB')
    axes[0].axis('off')
    
    axes[1].imshow(yuv_img[:, :, 0], cmap='gray')
    axes[1].set_title('Y (Luminance)')
    axes[1].axis('off')
    
    axes[2].imshow(yuv_img[:, :, 1], cmap='RdBu')
    axes[2].set_title('U (Chrominance)')
    axes[2].axis('off')
    
    axes[3].imshow(yuv_img[:, :, 2], cmap='RdBu')
    axes[3].set_title('V (Chrominance)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved YUV visualization to {save_path}")
    
    plt.close()


def create_comparison_grid(images_dict, save_path=None):
    """
    Create a grid comparison of multiple images
    
    Args:
        images_dict: Dictionary {name: image_array}
        save_path: Optional path to save
    """
    n_images = len(images_dict)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for ax, (name, img) in zip(axes, images_dict.items()):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison grid to {save_path}")
    
    plt.close()


# ============================================================================
# Data Statistics
# ============================================================================

def compute_exposure_statistics(image_dir):
    """
    Compute exposure statistics for a dataset
    
    Args:
        image_dir: Directory containing images
    
    Returns:
        stats: Dictionary of statistics
    """
    from pathlib import Path
    
    image_paths = list(Path(image_dir).glob('*.png')) + list(Path(image_dir).glob('*.jpg'))
    
    exposures = []
    
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Compute mean luminance as exposure proxy
        y = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
        exposure = np.mean(y)
        exposures.append(exposure)
    
    stats = {
        'mean': np.mean(exposures),
        'std': np.std(exposures),
        'min': np.min(exposures),
        'max': np.max(exposures),
        'median': np.median(exposures),
    }
    
    print("Dataset Exposure Statistics:")
    print("="*50)
    for k, v in stats.items():
        print(f"  {k:10s}: {v:.4f}")
    print("="*50)
    
    return stats


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test metrics
    print("Testing utility functions...")
    
    # Create dummy images
    img1 = np.random.rand(256, 256, 3)
    img2 = img1 + 0.1 * np.random.rand(256, 256, 3)
    img2 = np.clip(img2, 0, 1)
    
    # Test evaluation
    metrics = evaluate_enhancement(img1, img2)
    
    print("\nUtility functions test complete!")
