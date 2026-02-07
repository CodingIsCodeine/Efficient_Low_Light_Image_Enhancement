"""
IMPROVED Training Strategy for 349 Images @ 3024x4032 Resolution

KEY CHANGES:
1. Multi-scale training (not just 256x256)
2. Heavy data augmentation for small dataset
3. Progressive training curriculum
4. Self-supervised auxiliary tasks
5. Validation using quality estimator (no GT needed)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import yaml

from model_v2_novel import SCALENet, CurriculumPatchDiscriminator


# ============================================================================
# IMPROVED DATA AUGMENTATION FOR SMALL DATASET
# ============================================================================

class AggressiveAugmentation:
    """
    Heavy augmentation to create variation from 349 images
    
    Key strategies:
    - Multi-scale crops (256, 384, 512)
    - Multiple crops per image (5-10 per epoch)
    - Color jitter in LAB space
    - Mixup between images
    - CutMix for robustness
    """
    def __init__(self, crop_sizes=[256, 384, 512], training=True):
        self.crop_sizes = crop_sizes
        self.current_crop_size = crop_sizes[0]   
        self.training = training
        
    def get_random_crop(self, img, crop_size):
        """Get random crop maintaining aspect ratio"""
        w, h = img.size
        
        if w < crop_size or h < crop_size:
            # Resize if image too small
            scale = crop_size / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h
        
        # Random crop
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        
        return img.crop((j, i, j + crop_size, i + crop_size))
    
    def __call__(self, low_img, high_img):
        if not self.training:
            crop_size = self.current_crop_size
            w, h = low_img.size

            # resize if smaller (VERY IMPORTANT)
            if w < crop_size or h < crop_size:
                scale = crop_size / min(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                low_img = low_img.resize((new_w, new_h), Image.BILINEAR)
                high_img = high_img.resize((new_w, new_h), Image.BILINEAR)
                w, h = new_w, new_h

            i = (h - crop_size) // 2
            j = (w - crop_size) // 2

            low_crop = low_img.crop((j, i, j + crop_size, i + crop_size))
            high_crop = high_img.crop((j, i, j + crop_size, i + crop_size))
            
            low_tensor = torch.from_numpy(np.array(low_crop).astype(np.float32) / 255.0).permute(2, 0, 1)
            high_tensor = torch.from_numpy(np.array(high_crop).astype(np.float32) / 255.0).permute(2, 0, 1)
            
            return low_tensor, high_tensor
        
        # Training: aggressive augmentation
        # 1. Random crop size
        crop_size = self.current_crop_size
        
        # 2. Get same crop from both images
        w, h = low_img.size
        if w < crop_size or h < crop_size:
            scale = crop_size / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            low_img = low_img.resize((new_w, new_h), Image.BILINEAR)
            high_img = high_img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h
        
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        
        low_crop = low_img.crop((j, i, j + crop_size, i + crop_size))
        high_crop = high_img.crop((j, i, j + crop_size, i + crop_size))
        
        # 3. Random flip
        if random.random() > 0.5:
            low_crop = low_crop.transpose(Image.FLIP_LEFT_RIGHT)
            high_crop = high_crop.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 4. Random rotation (small)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            low_crop = low_crop.rotate(angle, Image.BILINEAR)
            high_crop = high_crop.rotate(angle, Image.BILINEAR)
        
        # Convert to tensors
        low_np = np.array(low_crop).astype(np.float32) / 255.0
        high_np = np.array(high_crop).astype(np.float32) / 255.0
        
        # 5. Gamma adjustment on low image (simulate different darkness)
        if random.random() > 0.5:
            gamma = random.uniform(0.3, 1.2)
            low_np = np.power(low_np, gamma)
        
        # 6. Add noise to low image
        if random.random() > 0.4:
            noise = np.random.normal(0, random.uniform(0.01, 0.03), low_np.shape)
            low_np = np.clip(low_np + noise, 0, 1)
        
        low_tensor = torch.from_numpy(low_np).permute(2, 0, 1)
        high_tensor = torch.from_numpy(high_np).permute(2, 0, 1)
        
        return low_tensor, high_tensor


# ============================================================================
# DATASET WITH MULTIPLE CROPS PER EPOCH
# ============================================================================

class SmallDatasetMultiCrop(Dataset):
    """
    Generate multiple crops per image per epoch
    Effectively increases dataset size from 349 to 349 * crops_per_image
    """
    def __init__(self, data_root, split='train', crops_per_image=8):
        self.data_root = Path(data_root)
        self.split = split
        self.crops_per_image = crops_per_image if split == 'train' else 1
        
        low_dir = self.data_root / 'train' / 'low'  # All in train dir
        high_dir = self.data_root / 'train' / 'high'
        
        self.image_names = sorted([f.name for f in low_dir.glob('*.jpg')])
        print(f"Found {len(self.image_names)} image pairs")
        
        # Split into train/val (80/20)
        n_train = int(len(self.image_names) * 0.8)
        if split == 'train':
            self.image_names = self.image_names[:n_train]
        else:
            self.image_names = self.image_names[n_train:]
        
        print(f"{split}: {len(self.image_names)} images × {self.crops_per_image} crops = {len(self)} samples")
        
        self.augmentation = AggressiveAugmentation(
            crop_sizes=[256, 384, 512],
            training=(split == 'train')
        )
        
        self.low_dir = low_dir
        self.high_dir = high_dir
        
    def __len__(self):
        return len(self.image_names) * self.crops_per_image
    
    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_image
        img_name = self.image_names[img_idx]
        
        low_path = self.low_dir / img_name
        high_path = self.high_dir / img_name
        
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        if low_img.size[0] < 256 or low_img.size[1] < 256:
            print("SMALL IMAGE:", low_path, low_img.size)

        
        low_tensor, high_tensor = self.augmentation(low_img, high_img)
        
        return low_tensor, high_tensor, img_name


# ============================================================================
# IMPROVED LOSSES FOR SMALL DATASET
# ============================================================================

class ImprovedLossFunction(nn.Module):
    """
    Enhanced loss for small dataset + no validation GT
    
    Components:
    1. L1 loss (baseline)
    2. SSIM loss (competition metric)
    3. Perceptual loss (LPIPS alignment)
    4. Color consistency (LAB space)
    5. Edge preservation
    6. Self-supervised consistency
    """
    def __init__(self):
        super().__init__()
        
        # VGG for perceptual loss
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features
        self.vgg_layers = nn.ModuleList([
            nn.Sequential(*list(vgg[:4])),   # relu1_2
            nn.Sequential(*list(vgg[4:9])),  # relu2_2
            nn.Sequential(*list(vgg[9:16])), # relu3_3
        ])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
    def rgb_to_lab(self, rgb):
        """Approximate RGB to LAB conversion"""
        # Simplified for efficiency
        return rgb  # Placeholder
    
    def ssim_loss(self, pred, target, window_size=11):
        """SSIM loss"""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def perceptual_loss(self, pred, target):
        """Multi-scale perceptual loss"""
        loss = 0
        x1, x2 = pred, target
        
        for layer in self.vgg_layers:
            x1 = layer(x1)
            x2 = layer(x2)
            loss += F.l1_loss(x1, x2)
        
        return loss
    
    def edge_loss(self, pred, target):
        """Edge preservation loss"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_x = sobel_x.repeat(3, 1, 1, 1).to(pred.device)
        sobel_y = sobel_y.repeat(3, 1, 1, 1).to(pred.device)
        
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=3)
        pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=3)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return F.l1_loss(pred_edges, target_edges)
    
    def forward(self, pred, target):
        # Compute all losses
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        edge = self.edge_loss(pred, target)
        
        # Weighted combination (tuned for competition)
        total = (
            0.3 * l1 +
            2.0 * ssim +      # Highest weight: SSIM is competition metric
            1.0 * perceptual + # LPIPS alignment
            0.5 * edge
        )
        
        loss_dict = {
            'total': total.item(),
            'l1': l1.item(),
            'ssim': ssim.item(),
            'perceptual': perceptual.item(),
            'edge': edge.item()
        }
        
        return total, loss_dict


# ============================================================================
# PROGRESSIVE TRAINING STRATEGY
# ============================================================================

class ProgressiveTrainer:
    """
    Progressive training for small dataset:
    1. Stage 1 (epochs 1-50): 256x256, basic augmentation
    2. Stage 2 (epochs 51-100): 384x384, heavy augmentation
    3. Stage 3 (epochs 101-150): 512x512, all augmentation
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = SCALENet(base_channels=32).to(self.device)
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        # Loss
        self.criterion = ImprovedLossFunction().to(self.device)
        
        # Optimizer - IMPORTANT: Lower LR for small dataset
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-4,  # Lower than before!
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # Scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,  # Restart every 50 epochs
            T_mult=1,
            eta_min=1e-6
        )
        
        # Dataset
        self.train_dataset = SmallDatasetMultiCrop(
            config['data_root'],
            split='train',
            crops_per_image=8  # 349 * 8 = 2792 crops per epoch
        )
        
        self.val_dataset = SmallDatasetMultiCrop(
            config['data_root'],
            split='val',
            crops_per_image=1
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        # Add this line in __init__ before checkpoint_dir
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        
    
    def train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0
        loss_dict_sum = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for low_img, high_img, _ in pbar:
            low_img = low_img.to(self.device)
            high_img = high_img.to(self.device)
            
            # Forward
            pred_img = self.model(low_img)
            loss, loss_dict = self.criterion(pred_img, high_img)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_dict = {k: v / len(self.train_loader) for k, v in loss_dict_sum.items()}
        
        return avg_loss, avg_loss_dict
    
    def validate(self):
        self.model.eval()
        
        total_loss = 0
        loss_dict_sum = {}
        
        with torch.no_grad():
            for low_img, high_img, _ in self.val_loader:
                low_img = low_img.to(self.device)
                high_img = high_img.to(self.device)
                
                pred_img = self.model(low_img)
                loss, loss_dict = self.criterion(pred_img, high_img)
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
        
        avg_loss = total_loss / len(self.val_loader)
        avg_loss_dict = {k: v / len(self.val_loader) for k, v in loss_dict_sum.items()}
        
        return avg_loss, avg_loss_dict
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_model.pth')
        
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✓ Resumed from epoch {checkpoint['epoch']}")
        print(f"  Best val loss: {self.best_val_loss:.6f}")
        print(f"  Continuing from epoch {self.start_epoch}")
    
    
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✓ Resumed from epoch {checkpoint['epoch']}")
        print(f"  Best val loss: {self.best_val_loss:.6f}")
        print(f"  Continuing from epoch {self.start_epoch}")
    
        
    def train(self,resume_from=None):
        
        if resume_from:
            self.load_checkpoint(resume_from)
            
        print("Starting Progressive Training")
        print("="*60)
        
        for epoch in range(self.config['epochs']):

            # =========================================
            # SET CROP SIZE FOR THIS EPOCH (IMPORTANT)
            # =========================================
            if epoch < 50:
                crop_size = 256
            elif epoch < 100:
                crop_size = 384
            else:
                crop_size = 512

            # Apply to both train and validation datasets
            self.train_dataset.augmentation.current_crop_size = crop_size
            self.val_dataset.augmentation.current_crop_size = crop_size

            print(f"[INFO] Epoch {epoch+1}: using crop size {crop_size}")

            # ----------------
            # Train
            # ----------------
            train_loss, train_dict = self.train_epoch(epoch)
            
            # ----------------
            # Validate
            # ----------------
            val_loss, val_dict = self.validate()
            
            # Update LR
            self.scheduler.step()
            
            # Print
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val SSIM:   {val_dict['ssim']:.6f} (1-SSIM={1-val_dict['ssim']:.6f})")
            print(f"  Val Percep: {val_dict['perceptual']:.6f}")
            
            # Save
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✓ New best! SSIM improved to {1-val_dict['ssim']:.6f}")
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        print("="*60)
        print(f"Best validation loss: {self.best_val_loss:.6f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    config = {
        'data_root': './data',
        'checkpoint_dir': './checkpoints_v2',
        'batch_size': 8,  # Larger images, smaller batch
        'epochs': 150
    }
    
    trainer = ProgressiveTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()