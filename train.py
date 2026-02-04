"""
Training Script for ADLNet

Training Strategy:
- Optimizer: AdamW with cosine annealing
- Learning rate: 1e-3 -> 1e-5
- Batch size: 16 (adjust based on GPU memory)
- Epochs: 150-200
- Data augmentation: LLIE-specific transforms
- Mixed precision training for efficiency
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path

from model import ADLNet
from losses import ADLNetLoss


# ============================================================================
# LLIE-Specific Data Augmentation
# ============================================================================

class LLIEAugmentation:
    """
    Augmentation specifically designed for low-light image enhancement
    
    Key principles:
    - Simulate various low-light conditions
    - Preserve natural degradation patterns
    - Avoid unrealistic distortions
    """
    def __init__(self, training=True):
        self.training = training
        
    def __call__(self, low_img, high_img):
        """
        Args:
            low_img: PIL Image (low-light)
            high_img: PIL Image (normal-light, ground truth)
        """
        if not self.training:
            # Validation: only resize and normalize
            low_img = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor()
            ])(low_img)
            high_img = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor()
            ])(high_img)
            return low_img, high_img
        
        # Training augmentations
        # 1. Random crop to maintain paired correspondence
        if np.random.rand() > 0.5:
            i, j, h, w = T.RandomCrop.get_params(low_img, output_size=(256, 256))
            low_img = T.functional.crop(low_img, i, j, h, w)
            high_img = T.functional.crop(high_img, i, j, h, w)
        else:
            low_img = T.Resize((256, 256))(low_img)
            high_img = T.Resize((256, 256))(high_img)
        
        # 2. Random horizontal flip (paired)
        if np.random.rand() > 0.5:
            low_img = T.functional.hflip(low_img)
            high_img = T.functional.hflip(high_img)
        
        # 3. Random rotation (small angles, paired)
        if np.random.rand() > 0.7:
            angle = np.random.uniform(-10, 10)
            low_img = T.functional.rotate(low_img, angle)
            high_img = T.functional.rotate(high_img, angle)
        
        # Convert to tensor
        low_img = T.ToTensor()(low_img)
        high_img = T.ToTensor()(high_img)
        
        # 4. Simulate varying darkness levels (on low_img only)
        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.5, 1.5)
            low_img = torch.pow(low_img, gamma)
        
        # 5. Add synthetic noise to low-light image (common in real low-light)
        if np.random.rand() > 0.6:
            noise_level = np.random.uniform(0.0, 0.05)
            noise = torch.randn_like(low_img) * noise_level
            low_img = torch.clamp(low_img + noise, 0, 1)
        
        return low_img, high_img


# ============================================================================
# Dataset
# ============================================================================

class LLIEDataset(Dataset):
    """
    Dataset for Low-Light Image Enhancement
    
    Expected structure:
        data_root/
            train/
                low/
                    img001.png
                    img002.png
                    ...
                high/
                    img001.png
                    img002.png
                    ...
            val/
                low/
                high/
    """
    def __init__(self, data_root, split='train'):
        self.data_root = Path(data_root)
        self.split = split
        
        self.low_dir = self.data_root / split / 'low'
        self.high_dir = self.data_root / split / 'high'
        
        # Get image list
        self.image_names = sorted([f.name for f in self.low_dir.glob('*.png')])
        if len(self.image_names) == 0:
            self.image_names = sorted([f.name for f in self.low_dir.glob('*.jpg')])
        
        print(f"Loaded {len(self.image_names)} image pairs for {split}")
        
        # Augmentation
        self.augmentation = LLIEAugmentation(training=(split == 'train'))
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load images
        low_path = self.low_dir / img_name
        high_path = self.high_dir / img_name
        
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        # Apply augmentation
        low_img, high_img = self.augmentation(low_img, high_img)
        
        return low_img, high_img, img_name


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = ADLNet().to(self.device)
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        # Loss
        self.criterion = ADLNetLoss().to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Datasets
        self.train_dataset = LLIEDataset(config['data_root'], split='train')
        self.val_dataset = LLIEDataset(config['data_root'], split='val')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Best validation loss
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        loss_dict_sum = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (low_img, high_img, _) in enumerate(pbar):
            low_img = low_img.to(self.device)
            high_img = high_img.to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                pred_img = self.model(low_img)
                loss, loss_dict = self.criterion(pred_img, high_img)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate losses
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_dict = {k: v / len(self.train_loader) for k, v in loss_dict_sum.items()}
        
        return avg_loss, avg_loss_dict
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        loss_dict_sum = {}
        
        with torch.no_grad():
            for low_img, high_img, _ in tqdm(self.val_loader, desc="Validating"):
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
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            # Also save model only (for deployment)
            torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_model.pth')
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_loss_dict = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_loss_dict = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val SSIM Loss: {val_loss_dict['ssim']:.6f}")
            print(f"  Val Perceptual: {val_loss_dict['perceptual']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✓ New best model! (val_loss: {val_loss:.6f})")
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("="*60)


# ============================================================================
# Main
# ============================================================================

def main():
    # Configuration
    config = {
        'data_root': './data',  # Update with your data path
        'checkpoint_dir': './checkpoints',
        'batch_size': 16,
        'num_workers': 4,
        'epochs': 150,
        'learning_rate': 1e-3,
        'min_lr': 1e-5,
        'weight_decay': 1e-4,
    }
    
    # Save config
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    with open(os.path.join(config['checkpoint_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
