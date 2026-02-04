"""
Custom Loss Functions for ADLNet

Designed to optimize for:
- SSIM (structural similarity)
- LPIPS (perceptual similarity)
- DISTS (deep image structure and texture similarity)
- No-reference metrics (LIQE, MUSIQ, Q-Align)

Loss Philosophy:
1. Perceptual losses > pixel-wise losses
2. Multi-scale structure preservation
3. Edge-aware detail recovery
4. Color fidelity in UV space
5. Noise suppression implicit in smooth regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Perceptual Loss (Lightweight VGG-based)
# ============================================================================

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using lightweight VGG features
    Aligns with LPIPS metric optimization
    """
    def __init__(self):
        super().__init__()
        # Use VGG16 pretrained features (we'll only use early layers for efficiency)
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features
        
        # Extract features at multiple depths for multi-scale perception
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16])) # relu3_3
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """
        Compute perceptual loss at multiple scales
        """
        # Extract features
        pred_feat1 = self.slice1(pred)
        pred_feat2 = self.slice2(pred_feat1)
        pred_feat3 = self.slice3(pred_feat2)
        
        target_feat1 = self.slice1(target)
        target_feat2 = self.slice2(target_feat1)
        target_feat3 = self.slice3(target_feat2)
        
        # Compute L1 loss at each scale
        loss1 = F.l1_loss(pred_feat1, target_feat1)
        loss2 = F.l1_loss(pred_feat2, target_feat2)
        loss3 = F.l1_loss(pred_feat3, target_feat3)
        
        # Weighted combination (deeper features = more perceptual)
        return loss1 + loss2 + 2.0 * loss3


# ============================================================================
# Structural Similarity Loss
# ============================================================================

class SSIMLoss(nn.Module):
    """
    Multi-scale SSIM loss
    Directly optimizes SSIM metric
    """
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma=1.5):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2.0 * sigma**2))) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2):
        """Compute SSIM between two images"""
        window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, pred, target):
        """Return 1 - SSIM as loss (to minimize)"""
        return 1.0 - self.ssim(pred, target)


# ============================================================================
# Edge-Aware Detail Loss
# ============================================================================

class EdgeAwareLoss(nn.Module):
    """
    Preserves edges and fine details
    Helps with texture preservation for DISTS metric
    """
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
    def forward(self, pred, target):
        """Compute edge-aware loss"""
        self.sobel_x = self.sobel_x.to(pred.device)
        self.sobel_y = self.sobel_y.to(pred.device)
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # L1 loss on edges
        return F.l1_loss(pred_edges, target_edges)


# ============================================================================
# Color Consistency Loss (YUV Space)
# ============================================================================

class ColorConsistencyLoss(nn.Module):
    """
    Ensures color fidelity in YUV space
    Prevents color distortion common in LLIE
    """
    def __init__(self):
        super().__init__()
    
    def rgb_to_yuv(self, rgb):
        """Convert RGB to YUV"""
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return y, u, v
    
    def forward(self, pred, target):
        """
        Compute color consistency loss in UV channels
        Y channel is handled by other losses
        """
        _, pred_u, pred_v = self.rgb_to_yuv(pred)
        _, target_u, target_v = self.rgb_to_yuv(target)
        
        # L1 loss on chrominance channels
        loss_u = F.l1_loss(pred_u, target_u)
        loss_v = F.l1_loss(pred_v, target_v)
        
        return loss_u + loss_v


# ============================================================================
# Illumination Smoothness Loss
# ============================================================================

class IlluminationSmoothnessLoss(nn.Module):
    """
    Encourages smooth illumination changes
    Prevents noise amplification in uniform regions
    Helps with no-reference quality metrics
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Compute total variation loss on luminance channel
        """
        # Convert to grayscale (luminance)
        pred_gray = 0.299 * pred[:, 0, :, :] + 0.587 * pred[:, 1, :, :] + 0.114 * pred[:, 2, :, :]
        
        # Compute horizontal and vertical differences
        diff_h = torch.abs(pred_gray[:, :, 1:] - pred_gray[:, :, :-1])
        diff_v = torch.abs(pred_gray[:, 1:, :] - pred_gray[:, :-1, :])
        
        # Total variation
        tv_loss = diff_h.mean() + diff_v.mean()
        
        return tv_loss


# ============================================================================
# Combined Loss
# ============================================================================

class ADLNetLoss(nn.Module):
    """
    Combined loss function for ADLNet
    
    Loss components:
    1. Perceptual Loss (LPIPS alignment)         - weight: 1.0
    2. SSIM Loss (SSIM optimization)              - weight: 2.0
    3. Edge-Aware Loss (DISTS alignment)          - weight: 0.5
    4. Color Consistency (UV fidelity)            - weight: 0.3
    5. Illumination Smoothness (noise control)    - weight: 0.1
    6. L1 Loss (baseline reconstruction)          - weight: 0.2
    
    Total loss balances:
    - Perceptual quality (for reference metrics)
    - Structural preservation (SSIM/DISTS)
    - Color accuracy (prevents distortion)
    - Noise suppression (for no-reference metrics)
    """
    def __init__(self):
        super().__init__()
        self.perceptual = PerceptualLoss()
        self.ssim = SSIMLoss()
        self.edge = EdgeAwareLoss()
        self.color = ColorConsistencyLoss()
        self.smooth = IlluminationSmoothnessLoss()
        
        # Loss weights (tuned for optimal metric performance)
        self.w_perceptual = 1.0
        self.w_ssim = 2.0
        self.w_edge = 0.5
        self.w_color = 0.3
        self.w_smooth = 0.1
        self.w_l1 = 0.2
    
    def forward(self, pred, target):
        """
        Compute combined loss
        
        Args:
            pred: Predicted enhanced image [B, 3, H, W]
            target: Ground truth image [B, 3, H, W]
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual losses (for logging)
        """
        # Compute individual losses
        loss_perceptual = self.perceptual(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_edge = self.edge(pred, target)
        loss_color = self.color(pred, target)
        loss_smooth = self.smooth(pred, target)
        loss_l1 = F.l1_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.w_perceptual * loss_perceptual +
            self.w_ssim * loss_ssim +
            self.w_edge * loss_edge +
            self.w_color * loss_color +
            self.w_smooth * loss_smooth +
            self.w_l1 * loss_l1
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'perceptual': loss_perceptual.item(),
            'ssim': loss_ssim.item(),
            'edge': loss_edge.item(),
            'color': loss_color.item(),
            'smooth': loss_smooth.item(),
            'l1': loss_l1.item()
        }
        
        return total_loss, loss_dict


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test losses
    criterion = ADLNetLoss()
    
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    
    loss, loss_dict = criterion(pred, target)
    
    print("Loss Function Test:")
    print("="*50)
    for k, v in loss_dict.items():
        print(f"  {k:15s}: {v:.6f}")
    print("="*50)
