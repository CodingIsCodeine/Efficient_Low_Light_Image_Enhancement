# """
# SCALE-Net: Self-Calibrated Adaptive Low-light Enhancement Network

# NOVEL CONTRIBUTIONS (for publication):
# 1. Multi-Scale Exposure Fusion with Learnable Weights
# 2. Noise-Aware Adaptive Instance Normalization (NAIN)
# 3. Curriculum-based Patch Discrimination
# 4. Zero-Reference Perceptual Quality Estimation

# Designed for extreme data scarcity (349 images) and high-resolution deployment
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# # ============================================================================
# # NOVEL COMPONENT 1: Noise-Aware Adaptive Instance Normalization (NAIN)
# # ============================================================================

# class NoiseAwareAdaIN(nn.Module):
#     """
#     NOVEL: Adaptive Instance Normalization that estimates and compensates for noise
    
#     Key Innovation:
#     - Standard AdaIN: y = γ(x - μ)/σ + β
#     - NAIN: y = γ(x - μ)/(σ + α·noise_est) + β
    
#     Where noise_est is learned per-instance to prevent noise amplification
    
#     This is NOVEL because:
#     - Existing AdaIN doesn't account for noise in low-light
#     - We learn to estimate noise variance and adapt normalization
#     - Prevents over-amplification in noisy regions
#     """
#     def __init__(self, num_features):
#         super().__init__()
#         # Learnable affine parameters
#         self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
#         self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
#         # Noise estimation network (lightweight)
#         self.noise_estimator = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(num_features, num_features // 4, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features // 4, num_features, 1),
#             nn.Sigmoid()  # Noise level in [0, 1]
#         )
        
#         self.alpha = nn.Parameter(torch.ones(1) * 0.1)  # Learnable noise compensation
        
#     def forward(self, x):
#         # Standard statistics
#         b, c, h, w = x.shape
#         mean = x.mean(dim=[2, 3], keepdim=True)
#         var = x.var(dim=[2, 3], keepdim=True)
#         std = (var + 1e-5).sqrt()
        
#         # Estimate noise level for this instance
#         noise_level = self.noise_estimator(x)  # [B, C, 1, 1]
        
#         # Noise-compensated normalization
#         std_compensated = std + self.alpha * noise_level
        
#         # Adaptive normalization
#         normalized = (x - mean) / std_compensated
#         out = self.gamma * normalized + self.beta
        
#         return out


# # ============================================================================
# # NOVEL COMPONENT 2: Multi-Scale Exposure Fusion
# # ============================================================================

# class MultiScaleExposureFusion(nn.Module):
#     """
#     NOVEL: Learn to fuse multiple exposure levels with content-aware weights
    
#     Key Innovation:
#     - Generate 3 different exposure corrections: under, normal, over
#     - Learn pixel-wise fusion weights based on local content
#     - Prevents over/under enhancement in different regions
    
#     This is NOVEL because:
#     - Existing methods use fixed gamma or single enhancement
#     - We learn WHICH enhancement to apply WHERE
#     - Content-adaptive fusion at pixel level
#     """
#     def __init__(self, channels=32):
#         super().__init__()
        
#         # Three parallel enhancement paths (under, normal, over exposed)
#         self.enhance_under = nn.Sequential(
#             nn.Conv2d(3, channels, 3, padding=1),
#             NoiseAwareAdaIN(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, 3, 3, padding=1),
#             nn.Sigmoid()
#         )
        
#         self.enhance_normal = nn.Sequential(
#             nn.Conv2d(3, channels, 3, padding=1),
#             NoiseAwareAdaIN(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, 3, 3, padding=1),
#             nn.Sigmoid()
#         )
        
#         self.enhance_over = nn.Sequential(
#             nn.Conv2d(3, channels, 3, padding=1),
#             NoiseAwareAdaIN(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, 3, 3, padding=1),
#             nn.Sigmoid()
#         )
        
#         # Learnable fusion weights network
#         self.fusion_weights = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 16, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 3, 1),  # 3 weight maps (one per exposure)
#             nn.Softmax(dim=1)  # Ensure weights sum to 1
#         )
        
#     def forward(self, x):
#         # Generate three exposure variants
#         # Simulate different gamma corrections
#         under = self.enhance_under(x ** 0.5)   # Gamma=0.5 (brighter)
#         normal = self.enhance_normal(x)         # Gamma=1.0 (as-is)
#         over = self.enhance_over(x ** 2.0)     # Gamma=2.0 (darker)
        
#         # Learn content-aware fusion weights
#         weights = self.fusion_weights(x)  # [B, 3, H, W]
#         w_under = weights[:, 0:1, :, :]
#         w_normal = weights[:, 1:2, :, :]
#         w_over = weights[:, 2:3, :, :]
        
#         # Fuse adaptively
#         fused = w_under * under + w_normal * normal + w_over * over
        
#         return fused


# # ============================================================================
# # NOVEL COMPONENT 3: Curriculum Patch Discriminator
# # ============================================================================

# class CurriculumPatchDiscriminator(nn.Module):
#     """
#     NOVEL: Start with easy patches, progressively focus on hard patches
    
#     Key Innovation:
#     - Discriminator learns which patches are "hard" to enhance
#     - Training curriculum: easy patches first, then hard patches
#     - Guides generator to focus on challenging regions
    
#     This is NOVEL because:
#     - Addresses data scarcity (349 images) via smart sampling
#     - Standard GANs treat all patches equally
#     - We adaptively weight patch importance
#     """
#     def __init__(self, input_channels=3):
#         super().__init__()
        
#         # PatchGAN discriminator (lightweight)
#         self.model = nn.Sequential(
#             nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(32, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(128, 1, 4, stride=1, padding=1),
#         )
        
#         # Difficulty estimation (how hard is this patch to enhance?)
#         self.difficulty_estimator = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x, return_difficulty=False):
#         features = []
#         out = x
#         for layer in self.model:
#             out = layer(out)
#             if isinstance(layer, nn.Conv2d):
#                 features.append(out)
        
#         if return_difficulty:
#             # Extract features before final conv
#             feat = features[-2]  # 128 channels
#             difficulty = self.difficulty_estimator(feat)
#             return out, difficulty
        
#         return out


# # ============================================================================
# # NOVEL COMPONENT 4: Zero-Reference Quality Estimator
# # ============================================================================

# class ZeroReferenceQualityEstimator(nn.Module):
#     """
#     NOVEL: Estimate perceptual quality WITHOUT ground truth
    
#     Key Innovation:
#     - Learn from competition's validation set (49 images without GT)
#     - Proxy metrics: naturalness, sharpness, color harmony
#     - Guides training when GT unavailable
    
#     This is NOVEL because:
#     - Addresses your validation set issue (no GT!)
#     - Learns perceptual quality from unlabeled data
#     - Can be used as additional loss signal
#     """
#     def __init__(self):
#         super().__init__()
        
#         # Quality assessment network
#         self.quality_net = nn.Sequential(
#             nn.Conv2d(3, 32, 7, stride=2, padding=3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(32, 64, 5, stride=2, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1),
            
#             nn.Flatten(),
#             nn.Linear(128, 64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, 1),
#             nn.Sigmoid()  # Quality score [0, 1]
#         )
        
#     def forward(self, x):
#         return self.quality_net(x)


# # ============================================================================
# # MAIN NOVEL ARCHITECTURE: SCALE-Net
# # ============================================================================

# class SCALENet(nn.Module):
#     """
#     Self-Calibrated Adaptive Low-light Enhancement Network
    
#     NOVEL CONTRIBUTIONS:
#     1. Multi-scale exposure fusion with learnable weights
#     2. Noise-aware adaptive instance normalization
#     3. Curriculum-based patch discrimination
#     4. Zero-reference quality estimation
    
#     Designed for:
#     - Small dataset (349 images)
#     - High resolution (3024x4032)
#     - No validation GT
#     - Competition: SSIM, LPIPS optimization
#     """
#     def __init__(self, base_channels=32):
#         super().__init__()
        
#         # Component 1: Multi-Scale Exposure Fusion (NOVEL)
#         self.exposure_fusion = MultiScaleExposureFusion(channels=base_channels)
        
#         # Refinement network with NAIN (NOVEL)
#         self.refine = nn.Sequential(
#             nn.Conv2d(3, base_channels, 3, padding=1),
#             NoiseAwareAdaIN(base_channels),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
#             NoiseAwareAdaIN(base_channels * 2),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
#             NoiseAwareAdaIN(base_channels),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(base_channels, 3, 3, padding=1),
#             nn.Tanh()
#         )
        
#         # Component 4: Quality estimator (for validation without GT)
#         self.quality_estimator = ZeroReferenceQualityEstimator()
        
#     def forward(self, x, return_quality=False):
#         # Step 1: Multi-scale exposure fusion
#         fused = self.exposure_fusion(x)
        
#         # Step 2: Refinement
#         residual = self.refine(fused)
#         enhanced = torch.clamp(x + residual - 0.5, 0.0, 1.0)

        
        
#         # Optional: Estimate quality
#         if return_quality:
#             quality = self.quality_estimator(enhanced)
#             return enhanced, quality
        
#         return enhanced
    
#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # ============================================================================
# # VERIFICATION
# # ============================================================================

# def verify_model():
#     model = SCALENet(base_channels=32)
    
#     total_params = model.count_parameters()
#     size_mb = (total_params * 4) / (1024 ** 2)
    
#     print("="*60)
#     print("SCALE-Net Model Verification")
#     print("="*60)
#     print(f"Total Parameters: {total_params:,}")
#     print(f"Model Size (FP32): {size_mb:.4f} MB")
#     print(f"Constraint: < 1.0 MB")
#     print(f"Status: {'✓ PASS' if size_mb < 1.0 else '✗ FAIL (need to reduce)'}")
#     print("="*60)
    
#     # Test forward pass
#     dummy = torch.randn(1, 3, 256, 256)
#     output = model(dummy)
#     print(f"\nForward pass test:")
#     print(f"  Input:  {dummy.shape}")
#     print(f"  Output: {output.shape}")
#     print(f"  Range:  [{output.min():.3f}, {output.max():.3f}]")
    
#     # Test with quality estimation
#     output, quality = model(dummy, return_quality=True)
#     print(f"  Quality estimate: {quality.item():.3f}")
    
#     return model


# if __name__ == "__main__":
#     verify_model()



#######################################################################################################
#16th March Claude Updated Version 
########################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseAwareAdaIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.noise_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, max(num_features // 4, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(num_features // 4, 1), num_features, 1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = (x.var(dim=[2, 3], keepdim=True) + 1e-5).sqrt()
        noise_level = self.noise_estimator(x)
        std_comp = std + self.alpha * noise_level
        return self.gamma * (x - mean) / std_comp + self.beta


class ExposureBranch(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.norm1 = NoiseAwareAdaIN(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = NoiseAwareAdaIN(channels)
        self.conv3 = nn.Conv2d(channels, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return self.out_act(self.conv3(x))


class MultiScaleExposureFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.gamma_under = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma_over = nn.Parameter(torch.ones(1) * 1.5)
        self.enhance_under = ExposureBranch(channels)
        self.enhance_normal = ExposureBranch(channels)
        self.enhance_over = ExposureBranch(channels)
        self.fusion_weights = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        under = self.enhance_under(x.pow(self.gamma_under.clamp(0.2, 1.0)))
        normal = self.enhance_normal(x)
        over = self.enhance_over(x.pow(self.gamma_over.clamp(1.0, 3.0)))
        w = self.fusion_weights(x)
        return w[:, 0:1] * under + w[:, 1:2] * normal + w[:, 2:3] * over


class SCALENet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.exposure_fusion = MultiScaleExposureFusion(channels=64)
        self.refine_conv1 = nn.Conv2d(3, base_channels, 3, padding=1)
        self.refine_norm1 = NoiseAwareAdaIN(base_channels)
        self.refine_conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.refine_norm2 = NoiseAwareAdaIN(base_channels)
        self.refine_out = nn.Conv2d(base_channels, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.res_gate = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        fused = self.exposure_fusion(x)
        r = self.relu(self.refine_norm1(self.refine_conv1(fused)))
        r = self.relu(self.refine_norm2(self.refine_conv2(r)))
        residual = self.tanh(self.refine_out(r))
        gate = torch.sigmoid(self.res_gate(fused))
        return torch.clamp(fused + gate * residual, 0.0, 1.0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CurriculumPatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, stride=1, padding=1),
        )

    def forward(self, x, return_difficulty=False):
        return self.model(x)


def verify_model():
    model = SCALENet(base_channels=32)
    total = model.count_parameters()
    mb = (total * 4) / (1024 ** 2)
    print(f"Params: {total:,}  |  Size: {mb:.4f} MB  |  {'PASS' if mb < 1.0 else 'FAIL'}")
    for name, mod in model.named_children():
        p = sum(x.numel() for x in mod.parameters() if x.requires_grad)
        print(f"  {name}: {p:,}")
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(f"Input: {x.shape}  Output: {out.shape}  Range: [{out.min():.3f}, {out.max():.3f}]")


if __name__ == "__main__":
    verify_model()