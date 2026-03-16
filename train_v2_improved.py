# """
# IMPROVED Training Strategy for 349 Images @ 3024x4032 Resolution

# KEY CHANGES:
# 1. Multi-scale training (not just 256x256)
# 2. Heavy data augmentation for small dataset
# 3. Progressive training curriculum
# 4. Self-supervised auxiliary tasks
# 5. Validation using quality estimator (no GT needed)
# """

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# from PIL import Image
# import numpy as np
# from pathlib import Path
# import random
# from tqdm import tqdm
# import yaml
# import copy
# import lpips
# from model_v2_novel import SCALENet, CurriculumPatchDiscriminator



# # ============================================================================
# # IMPROVED DATA AUGMENTATION FOR SMALL DATASET
# # ============================================================================

# class AggressiveAugmentation:
#     """
#     Heavy augmentation to create variation from 349 images
    
#     Key strategies:
#     - Multi-scale crops (256, 384, 512)
#     - Multiple crops per image (5-10 per epoch)
#     - Color jitter in LAB space
#     - Mixup between images
#     - CutMix for robustness
#     """
#     def __init__(self, crop_sizes=[256, 384, 512], training=True):
#         self.crop_sizes = crop_sizes
#         self.current_crop_size = crop_sizes[0]   
#         self.training = training
        
#     def get_random_crop(self, img, crop_size):
#         """Get random crop maintaining aspect ratio"""
#         w, h = img.size
        
#         if w < crop_size or h < crop_size:
#             # Resize if image too small
#             scale = crop_size / min(w, h)
#             new_w, new_h = int(w * scale), int(h * scale)
#             img = img.resize((new_w, new_h), Image.BILINEAR)
#             w, h = new_w, new_h
        
#         # Random crop
#         i = random.randint(0, h - crop_size)
#         j = random.randint(0, w - crop_size)
        
#         return img.crop((j, i, j + crop_size, i + crop_size))
    
#     def __call__(self, low_img, high_img):
#         if not self.training:
#             crop_size = self.current_crop_size
#             w, h = low_img.size

#             # resize if smaller (VERY IMPORTANT)
#             if w < crop_size or h < crop_size:
#                 scale = crop_size / min(w, h)
#                 new_w, new_h = int(w * scale), int(h * scale)
#                 low_img = low_img.resize((new_w, new_h), Image.BILINEAR)
#                 high_img = high_img.resize((new_w, new_h), Image.BILINEAR)
#                 w, h = new_w, new_h

#             i = (h - crop_size) // 2
#             j = (w - crop_size) // 2

#             low_crop = low_img.crop((j, i, j + crop_size, i + crop_size))
#             high_crop = high_img.crop((j, i, j + crop_size, i + crop_size))
            
#             low_tensor = torch.from_numpy(np.array(low_crop)/ 255.0).permute(2, 0, 1).float()
#             high_tensor = torch.from_numpy(np.array(high_crop)/ 255.0).permute(2, 0, 1).float()
            
#             return low_tensor, high_tensor
        
#         # Training: aggressive augmentation
#         # 1. Random crop size
#         crop_size = self.current_crop_size
        
#         # 2. Get same crop from both images
#         w, h = low_img.size
#         if w < crop_size or h < crop_size:
#             scale = crop_size / min(w, h)
#             new_w, new_h = int(w * scale), int(h * scale)
#             low_img = low_img.resize((new_w, new_h), Image.BILINEAR)
#             high_img = high_img.resize((new_w, new_h), Image.BILINEAR)
#             w, h = new_w, new_h
        
#         i = random.randint(0, h - crop_size)
#         j = random.randint(0, w - crop_size)
        
#         low_crop = low_img.crop((j, i, j + crop_size, i + crop_size))
#         high_crop = high_img.crop((j, i, j + crop_size, i + crop_size))
        
#         # 3. Random flip
#         if random.random() > 0.5:
#             low_crop = low_crop.transpose(Image.FLIP_LEFT_RIGHT)
#             high_crop = high_crop.transpose(Image.FLIP_LEFT_RIGHT)
        
#         # 4. Random rotation (small)
#         if random.random() > 0.5:
#             angle = random.uniform(-15, 15)
#             low_crop = low_crop.rotate(angle, Image.BILINEAR)
#             high_crop = high_crop.rotate(angle, Image.BILINEAR)
        
#         # Convert to tensors
#         low_np = np.array(low_crop).astype(np.float32) / 255.0
#         high_np = np.array(high_crop).astype(np.float32) / 255.0
        
#         # 5. Gamma adjustment on low image (simulate different darkness)
#         if random.random() > 0.5:
#             gamma = random.uniform(0.3, 1.2)
#             low_np = np.power(low_np, gamma)
        
#         # 6. Add noise to low image
#         if random.random() > 0.4:
#             noise = np.random.normal(0, random.uniform(0.01, 0.03), low_np.shape)
#             low_np = np.clip(low_np + noise, 0, 1)
        
#         low_tensor = torch.from_numpy(low_np).permute(2, 0, 1).float()
#         high_tensor = torch.from_numpy(high_np).permute(2, 0, 1).float()
        
#         return low_tensor, high_tensor


# # ============================================================================
# # DATASET WITH MULTIPLE CROPS PER EPOCH
# # ============================================================================

# class SmallDatasetMultiCrop(Dataset):
#     """
#     Generate multiple crops per image per epoch
#     Effectively increases dataset size from 349 to 349 * crops_per_image
#     """
#     def __init__(self, data_root, split='train', crops_per_image=8):
#         self.data_root = Path(data_root)
#         self.split = split
#         self.crops_per_image = crops_per_image if split == 'train' else 1
        
#         low_dir = self.data_root / 'train' / 'low'  # All in train dir
#         high_dir = self.data_root / 'train' / 'high'
        
#         self.image_names = sorted([f.name for f in low_dir.glob('*.jpg')])
#         print(f"Found {len(self.image_names)} image pairs")
        
#         # Split into train/val (80/20)
#         n_train = int(len(self.image_names) * 0.8)
#         if split == 'train':
#             self.image_names = self.image_names[:n_train]
#         else:
#             self.image_names = self.image_names[n_train:]
        
#         print(f"{split}: {len(self.image_names)} images × {self.crops_per_image} crops = {len(self)} samples")
        
#         self.augmentation = AggressiveAugmentation(
#             crop_sizes=[256, 384, 512],
#             training=(split == 'train')
#         )
        
#         self.low_dir = low_dir
#         self.high_dir = high_dir
        
#     def __len__(self):
#         return len(self.image_names) * self.crops_per_image
    
#     def __getitem__(self, idx):
#         img_idx = idx // self.crops_per_image
#         img_name = self.image_names[img_idx]
        
#         low_path = self.low_dir / img_name
#         high_path = self.high_dir / img_name
        
#         low_img = Image.open(low_path).convert('RGB')
#         high_img = Image.open(high_path).convert('RGB')
        
#         if low_img.size[0] < 256 or low_img.size[1] < 256:
#             print("SMALL IMAGE:", low_path, low_img.size)

        
#         low_tensor, high_tensor = self.augmentation(low_img, high_img)
        
#         return low_tensor, high_tensor, img_name


# # ============================================================================
# # IMPROVED LOSSES FOR SMALL DATASET
# # ============================================================================

# class ImprovedLossFunction(nn.Module):
#     """
#     Enhanced loss for small dataset + no validation GT
    
#     Components:
#     1. L1 loss (baseline)
#     2. SSIM loss (competition metric)
#     3. Perceptual loss (LPIPS alignment)
#     4. Color consistency (LAB space)
#     5. Edge preservation
#     6. Self-supervised consistency
#     """
#     def __init__(self):
#         super().__init__()
        
#         # VGG for perceptual loss
#         from torchvision.models import vgg16
#         vgg = vgg16(pretrained=True).features
#         self.vgg_layers = nn.ModuleList([
#             nn.Sequential(*list(vgg[:4])),   # relu1_2
#             nn.Sequential(*list(vgg[4:9])),  # relu2_2
#             nn.Sequential(*list(vgg[9:16])), # relu3_3
#         ])
#         for param in self.vgg_layers.parameters():
#             param.requires_grad = False
        
#     def rgb_to_lab(self, rgb):
#         """Approximate RGB to LAB conversion"""
#         # Simplified for efficiency
#         return rgb  # Placeholder
    
#     def ssim_loss(self, pred, target, window_size=11):
#         """SSIM loss"""
#         C1, C2 = 0.01 ** 2, 0.03 ** 2
        
#         mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
#         mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
#         mu1_sq = mu1 ** 2
#         mu2_sq = mu2 ** 2
#         mu1_mu2 = mu1 * mu2
        
#         sigma1_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
#         sigma2_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
#         sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
#                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
#         return 1 - ssim_map.mean()
    
#     def perceptual_loss(self, pred, target):
#         """Multi-scale perceptual loss"""
#         loss = 0
#         x1, x2 = pred, target
        
#         for layer in self.vgg_layers:
#             x1 = layer(x1)
#             x2 = layer(x2)
#             loss += F.l1_loss(x1, x2)
        
#         return loss
    
#     def edge_loss(self, pred, target):
#         """Edge preservation loss"""
#         # Sobel filters
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
#         sobel_x = sobel_x.repeat(3, 1, 1, 1).to(pred.device)
#         sobel_y = sobel_y.repeat(3, 1, 1, 1).to(pred.device)
        
#         pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=3)
#         pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=3)
#         pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        
#         target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=3)
#         target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=3)
#         target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
#         return F.l1_loss(pred_edges, target_edges)
    
#     def forward(self, pred, target):
#         # Compute all losses
#         l1 = F.l1_loss(pred, target)
#         ssim = self.ssim_loss(pred, target)
#         perceptual = self.perceptual_loss(pred, target)
#         edge = self.edge_loss(pred, target)
        
#         # Weighted combination (tuned for competition)
#         total = (
#             0.3 * l1 +
#             2.0 * ssim +      # Highest weight: SSIM is competition metric
#             1.0 * perceptual + # LPIPS alignment
#             0.5 * edge
#         )
        
#         loss_dict = {
#             'total': total.item(),
#             'l1': l1.item(),
#             'ssim': ssim.item(),
#             'perceptual': perceptual.item(),
#             'edge': edge.item()
#         }
        
#         return total, loss_dict


# # ============================================================================
# # PROGRESSIVE TRAINING STRATEGY
# # ============================================================================

# class ProgressiveTrainer:
#     """
#     Progressive training for small dataset:
#     1. Stage 1 (epochs 1-50): 256x256, basic augmentation
#     2. Stage 2 (epochs 51-100): 384x384, heavy augmentation
#     3. Stage 3 (epochs 101-150): 512x512, all augmentation
#     """
#     def __init__(self, config):
#         self.config = config
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Model
#         self.model = SCALENet(base_channels=32).to(self.device)
#         print(f"Model parameters: {self.model.count_parameters():,}")
        
#         if torch.cuda.device_count() > 1:
#             print(f"Using {torch.cuda.device_count()} GPUs!")
#             self.model = nn.DataParallel(self.model)

#         self.model = self.model.to(self.device)

#         print(f"Model parameters: {self.model.module.count_parameters() if isinstance(self.model, nn.DataParallel) else self.model.count_parameters():,}")
        
#         # Loss
#         self.criterion = ImprovedLossFunction().to(self.device)
        
#         self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
#         self.lpips_model.eval()

#         # Optimizer - IMPORTANT: Lower LR for small dataset
#         self.optimizer = torch.optim.AdamW(
#             self.model.parameters(),
#             lr=5e-6,  # Lower than before!
#             betas=(0.9, 0.999),
#             weight_decay=1e-4
#         )
#                 # after optimizer creation
#         self.ema_model = copy.deepcopy(self.model).to(self.device)
#         for p in self.ema_model.parameters():
#             p.requires_grad = False

#         self.ema_decay = 0.999

        
#         # Scheduler with warmup
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             self.optimizer,
#             T_0=50,  # Restart every 50 epochs
#             T_mult=1,
#             eta_min=1e-6
#         )
        
#         # Dataset
#         self.train_dataset = SmallDatasetMultiCrop(
#             config['data_root'],
#             split='train',
#             crops_per_image=8  # 349 * 8 = 2792 crops per epoch
#         )
        
#         self.val_dataset = SmallDatasetMultiCrop(
#             config['data_root'],
#             split='val',
#             crops_per_image=1
#         )
        
#         self.train_loader = DataLoader(
#             self.train_dataset,
#             batch_size=config['batch_size'],
#             shuffle=True,
#             num_workers=12,
#             pin_memory=True
#         )
        
#         self.val_loader = DataLoader(
#             self.val_dataset,
#             batch_size=config['batch_size'],
#             shuffle=False,
#             num_workers=12
#         )
#         self.start_epoch = 0
#         self.best_val_loss = float('inf')
#         # Add this line in __init__ before checkpoint_dir
#         self.start_epoch = 0
#         self.best_val_loss = float('inf')
        
#         self.checkpoint_dir = Path(config['checkpoint_dir'])
#         self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
#         self.best_val_loss = float('inf')
        
    
#     def train_epoch(self, epoch):
#         self.model.train()
        
#         total_loss = 0
#         loss_dict_sum = {}
        
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
#         for low_img, high_img, _ in pbar:
#             low_img = low_img.to(self.device)
#             high_img = high_img.to(self.device)
            
            
#                         # tiny label smoothing for regression
#             noise = torch.randn_like(high_img) * 0.01
#             high_img = torch.clamp(high_img + noise, 0, 1)

            
#             # Forward
#             pred_img = self.model(low_img)
#             loss, loss_dict = self.criterion(pred_img, high_img)
            
#             # Backward
#             self.optimizer.zero_grad()
#             loss.backward()
            
#             # Gradient clipping for stability
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
#             self.optimizer.step()
            
#                         # AFTER self.optimizer.step()
#             with torch.no_grad():
#                 for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
#                     ema_p.data = self.ema_decay * ema_p.data + (1 - self.ema_decay) * p.data

            
#             # Accumulate
#             total_loss += loss.item()
#             for k, v in loss_dict.items():
#                 loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v
            
#             pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
#         avg_loss = total_loss / len(self.train_loader)
#         avg_loss_dict = {k: v / len(self.train_loader) for k, v in loss_dict_sum.items()}
        
#         return avg_loss, avg_loss_dict
    
#     def validate(self):
#         self.model.eval()
        
#         total_loss = 0
#         loss_dict_sum = {}
        
#         with torch.no_grad():
#             for low_img, high_img, _ in self.val_loader:
#                 low_img = low_img.to(self.device)
#                 high_img = high_img.to(self.device)
                
#                 pred_img = self.ema_model(low_img)
#                 loss, loss_dict = self.criterion(pred_img, high_img)

#                 with torch.no_grad():
#                     lpips_val = self.lpips_model(pred_img * 2 - 1, high_img * 2 - 1).mean()

#                 loss_dict['lpips'] = lpips_val.item()

#                 total_loss += loss.item()
#                 for k, v in loss_dict.items():
#                     loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v

        
#         avg_loss = total_loss / len(self.val_loader)
#         avg_loss_dict = {k: v / len(self.val_loader) for k, v in loss_dict_sum.items()}
        
#         return avg_loss, avg_loss_dict
    
#     def save_checkpoint(self, epoch, val_loss, is_best=False):
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict(), 
#             'val_loss': val_loss
#         }
        
#         torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
#         if is_best:
#             torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
#             torch.save(self.model.state_dict(), self.checkpoint_dir / 'best_model.pth')
        
    
    
#     def load_checkpoint(self, checkpoint_path):
#         print(f"\nLoading checkpoint: {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)

#         # -------------------------------------------------
#         # CASE 1 — FULL CHECKPOINT (resume training)
#         # -------------------------------------------------
#         if 'model_state_dict' in checkpoint:
#             print("Resuming full training checkpoint")

#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#             if 'scheduler_state_dict' in checkpoint:
#                 self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

#             self.start_epoch = checkpoint['epoch'] + 1
#             self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

#             print(f"✓ Resumed from epoch {checkpoint['epoch']}")
#             print(f"  Continuing from epoch {self.start_epoch}")

#         # -------------------------------------------------
#         # CASE 2 — WEIGHTS ONLY (fine-tuning mode) ⭐
#         # -------------------------------------------------
#         else:
#             print("🔥 Loading weights-only checkpoint for FINE-TUNING")

#             self.model.load_state_dict(checkpoint)

#             # IMPORTANT: start fresh training loop
#             self.start_epoch = 0
#             self.best_val_loss = float('inf')

#             print("✓ Starting fine-tuning from epoch 1")
    
        
#     def train(self,resume_from=None):
        
#         if resume_from:
#             self.load_checkpoint(resume_from)
            
#         print("Starting Progressive Training")
#         print("="*60)
        
#         for epoch in range(self.start_epoch, self.config['epochs']):

#             # =========================================
#             # SET CROP SIZE FOR THIS EPOCH (IMPORTANT)
#             # # =========================================
#             # if epoch < 50:
#             #     crop_size = 256
#             #     new_batch_size = 32
#             # elif epoch < 100:
#             #     crop_size = 384
#             #     new_batch_size = 16
#             # else:
#             crop_size = 512
#             new_batch_size = 8
                
         
#             if self.train_loader.batch_size != new_batch_size:
#                 print(f"[INFO] Updating batch size to {new_batch_size} for crop {crop_size}")

#                 self.train_loader = DataLoader(
#                     self.train_dataset,
#                     batch_size=new_batch_size,
#                     shuffle=True,
#                     num_workers=12,
#                     pin_memory=True,
#                     persistent_workers=True

#                 )

#                 self.val_loader = DataLoader(
#                     self.val_dataset,
#                     batch_size=new_batch_size,
#                     shuffle=False,
#                     num_workers=12,
#                     persistent_workers=True

#                 )


#             # Apply to both train and validation datasets
#             self.train_dataset.augmentation.current_crop_size = crop_size
#             self.val_dataset.augmentation.current_crop_size = crop_size

#             print(f"[INFO] Epoch {epoch+1}: using crop size {crop_size}")

#             # ----------------
#             # Train
#             # ----------------
#             train_loss, train_dict = self.train_epoch(epoch)
            
#             # ----------------
#             # Validate
#             # ----------------
#             val_loss, val_dict = self.validate()
            
#             # Update LR
#             self.scheduler.step()
            
#             # Print
#             print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
#             print(f"  Train Loss: {train_loss:.6f}")
#             print(f"  Val Loss:   {val_loss:.6f}")
#             print(f"  Val SSIM:   {val_dict['ssim']:.6f} (1-SSIM={1-val_dict['ssim']:.6f})")
#             print(f"  Val Percep: {val_dict['perceptual']:.6f}")
#             print(f"  Val LPIPS:  {val_dict['lpips']:.6f}")

            
#             # Save
#             is_best = val_loss < self.best_val_loss
#             if is_best:
#                 self.best_val_loss = val_loss
#                 print(f"  ✓ New best! SSIM improved to {1-val_dict['ssim']:.6f}")
            
#             self.save_checkpoint(epoch, val_loss, is_best)
        
#         print("="*60)
#         print(f"Best validation loss: {self.best_val_loss:.6f}")


# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--resume', type=str, default=None)
#     args = parser.parse_args()
    
#     config = {
#         'data_root': './data',
#         'checkpoint_dir': './checkpoints_v2',
#         'batch_size': 32,  # Larger images, smaller batch
#         'epochs': 40
#     }
    
#     trainer = ProgressiveTrainer(config)
#     trainer.train(resume_from=args.resume)


# if __name__ == "__main__":
#     main()


###################################################################################
#16th March Claude Updated Code Version
###################################################################################

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import copy
import lpips
from model_v2_novel import SCALENet, CurriculumPatchDiscriminator
from losses_v2 import CompetitionLoss


class AggressiveAugmentation:
    def __init__(self, crop_sizes=[256, 384, 512], training=True):
        self.crop_sizes = crop_sizes
        self.current_crop_size = crop_sizes[0]
        self.training = training

    def __call__(self, low_img, high_img):
        crop_size = self.current_crop_size
        w, h = low_img.size

        if w < crop_size or h < crop_size:
            scale = crop_size / min(w, h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            low_img = low_img.resize((new_w, new_h), Image.BILINEAR)
            high_img = high_img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h

        if not self.training:
            i = (h - crop_size) // 2
            j = (w - crop_size) // 2
            low_crop = low_img.crop((j, i, j + crop_size, i + crop_size))
            high_crop = high_img.crop((j, i, j + crop_size, i + crop_size))
            low_t = torch.from_numpy(np.array(low_crop) / 255.0).permute(2, 0, 1).float()
            high_t = torch.from_numpy(np.array(high_crop) / 255.0).permute(2, 0, 1).float()
            return low_t, high_t

        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        low_crop = low_img.crop((j, i, j + crop_size, i + crop_size))
        high_crop = high_img.crop((j, i, j + crop_size, i + crop_size))

        if random.random() > 0.5:
            low_crop = low_crop.transpose(Image.FLIP_LEFT_RIGHT)
            high_crop = high_crop.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            low_crop = low_crop.transpose(Image.FLIP_TOP_BOTTOM)
            high_crop = high_crop.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            low_crop = low_crop.rotate(angle, Image.BILINEAR)
            high_crop = high_crop.rotate(angle, Image.BILINEAR)

        low_np = np.array(low_crop).astype(np.float32) / 255.0
        high_np = np.array(high_crop).astype(np.float32) / 255.0

        if random.random() > 0.5:
            gamma = random.uniform(0.3, 1.2)
            low_np = np.power(np.clip(low_np, 1e-8, 1.0), gamma)

        if random.random() > 0.4:
            noise = np.random.normal(0, random.uniform(0.01, 0.03), low_np.shape)
            low_np = np.clip(low_np + noise, 0, 1)

        return (torch.from_numpy(low_np).permute(2, 0, 1).float(),
                torch.from_numpy(high_np).permute(2, 0, 1).float())


class SmallDatasetMultiCrop(Dataset):
    def __init__(self, data_root, split='train', crops_per_image=8):
        self.data_root = Path(data_root)
        self.split = split
        self.crops_per_image = crops_per_image if split == 'train' else 1

        low_dir = self.data_root / 'train' / 'low'
        high_dir = self.data_root / 'train' / 'high'

        all_names = sorted([f.name for f in low_dir.glob('*.jpg')])
        print(f"Found {len(all_names)} image pairs")

        n_train = int(len(all_names) * 0.8)
        self.image_names = all_names[:n_train] if split == 'train' else all_names[n_train:]
        print(f"{split}: {len(self.image_names)} images x {self.crops_per_image} crops = {len(self)} samples")

        self.augmentation = AggressiveAugmentation(
            crop_sizes=[256, 384, 512],
            training=(split == 'train')
        )
        self.low_dir = low_dir
        self.high_dir = high_dir

    def __len__(self):
        return len(self.image_names) * self.crops_per_image

    def __getitem__(self, idx):
        img_name = self.image_names[idx // self.crops_per_image]
        low_img = Image.open(self.low_dir / img_name).convert('RGB')
        high_img = Image.open(self.high_dir / img_name).convert('RGB')
        low_t, high_t = self.augmentation(low_img, high_img)

        # MixUp augmentation (30% probability, training only)
        if self.split == 'train' and random.random() < 0.3:
            idx2 = random.randint(0, len(self.image_names) - 1)
            img_name2 = self.image_names[idx2]
            low2 = Image.open(self.low_dir / img_name2).convert('RGB')
            high2 = Image.open(self.high_dir / img_name2).convert('RGB')
            low2, high2 = self.augmentation(low2, high2)
            alpha = random.uniform(0.3, 0.7)
            low_t = alpha * low_t + (1 - alpha) * low2
            high_t = alpha * high_t + (1 - alpha) * high2

        return low_t, high_t, img_name


class ProgressiveTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SCALENet(base_channels=32).to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        n_params = (self.model.module if isinstance(self.model, nn.DataParallel)
                    else self.model).count_parameters()
        print(f"Model parameters: {n_params:,}  ({n_params*4/1024**2:.3f} MB)")

        self.criterion = CompetitionLoss().to(self.device)

        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_model.eval()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=2e-4,
            betas=(0.9, 0.9),
            weight_decay=1e-4
        )

        self.ema_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.ema_decay = 0.999

        self.train_dataset = SmallDatasetMultiCrop(config['data_root'], 'train', crops_per_image=8)
        self.val_dataset = SmallDatasetMultiCrop(config['data_root'], 'val', crops_per_image=1)

        self._current_batch_size = None
        self.train_loader = None
        self.val_loader = None
        self._rebuild_loaders(config['batch_size'])

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=2e-4,
            steps_per_epoch=len(self.train_loader),
            epochs=config['epochs'],
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1e4
        )

        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _rebuild_loaders(self, batch_size):
        if batch_size == self._current_batch_size:
            return
        self._current_batch_size = batch_size
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4,
            pin_memory=True, persistent_workers=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4,
            persistent_workers=True
        )

    def _get_stage(self, epoch):
        if epoch < 15:
            return 256, 16
        elif epoch < 30:
            return 384, 8
        else:
            return 512, 4

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_dict_sum = {}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for low_img, high_img, _ in pbar:
            low_img = low_img.to(self.device)
            high_img = high_img.to(self.device)

            pred = self.model(low_img)
            loss, loss_dict = self.criterion(pred, high_img)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                for ep, p in zip(self.ema_model.parameters(), self.model.parameters()):
                    ep.data = self.ema_decay * ep.data + (1 - self.ema_decay) * p.data

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        n = len(self.train_loader)
        return total_loss / n, {k: v / n for k, v in loss_dict_sum.items()}

    def validate(self):
        self.ema_model.eval()
        total_loss = 0
        loss_dict_sum = {}

        with torch.no_grad():
            for low_img, high_img, _ in self.val_loader:
                low_img = low_img.to(self.device)
                high_img = high_img.to(self.device)

                pred = self.ema_model(low_img)
                loss, loss_dict = self.criterion(pred, high_img)

                lp = self.lpips_model(pred * 2 - 1, high_img * 2 - 1).mean()
                loss_dict['lpips_direct'] = lp.item()

                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v

        n = len(self.val_loader)
        return total_loss / n, {k: v / n for k, v in loss_dict_sum.items()}

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        # Save EMA weights only as lightweight checkpoint (for resuming with weights-only)
        ema_sd = {k.replace('module.', ''): v
                  for k, v in self.ema_model.state_dict().items()}
        torch.save(ema_sd, self.checkpoint_dir / 'best_model.pth')

        # Full checkpoint for resuming training
        model_sd = self.model.state_dict()
        ema_full_sd = self.ema_model.state_dict()

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model_sd,
            'ema_state_dict': ema_full_sd,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        torch.save(ckpt, self.checkpoint_dir / 'latest.pth')

        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best.pth')
            print(f"  Saved best checkpoint (SSIM improving)")

    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'ema_state_dict' in ckpt:
                self.ema_model.load_state_dict(ckpt['ema_state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {ckpt['epoch']}, continuing from epoch {self.start_epoch}")
        else:
            # weights-only
            sd = {('module.' + k if not k.startswith('module.') and
                   torch.cuda.device_count() > 1 else k): v
                  for k, v in ckpt.items()}
            self.model.load_state_dict(sd, strict=False)
            self.start_epoch = 0
            self.best_val_loss = float('inf')
            print("Loaded weights-only checkpoint for fine-tuning")
            epochs_done = self.start_epoch
            epochs_left = self.config['epochs'] - epochs_done
            if epochs_left > 0:
                _, resume_batch_size = self._get_stage(self.start_epoch)
                self._rebuild_loaders(resume_batch_size)
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=5e-5,
                    steps_per_epoch=len(self.train_loader),
                    epochs=epochs_left,
                    pct_start=0.05,
                    div_factor=10,
                    final_div_factor=1e3
                )
                print(f"Rebuilt scheduler for {epochs_left} remaining epochs, max_lr=5e-5")
                
    def train(self, resume_from=None):
        if resume_from:
            self.load_checkpoint(resume_from)

        print("Starting Progressive Training")
        print("=" * 60)
        for epoch in range(self.start_epoch, self.config['epochs']):
            crop_size, batch_size = self._get_stage(epoch)
            self._rebuild_loaders(batch_size)
            self.train_dataset.augmentation.current_crop_size = crop_size
            self.val_dataset.augmentation.current_crop_size = crop_size

            print(f"\n[Epoch {epoch+1}/{self.config['epochs']}] "
                  f"crop={crop_size} batch={batch_size} "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

            train_loss, train_dict = self.train_epoch(epoch)
            val_loss, val_dict = self.validate()

            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val   Loss: {val_loss:.6f}")
            print(f"  Val   SSIM: {val_dict.get('ssim_score', 0):.4f}")
            print(f"  Val   LPIPS(direct): {val_dict.get('lpips_direct', 0):.4f}")
            print(f"  Val   FFT:  {val_dict.get('fft', 0):.6f}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save every epoch (latest) and on improvement (best)
            self.save_checkpoint(epoch, val_loss, is_best)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_v2')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    config = {
        'data_root': args.data_root,
        'checkpoint_dir': args.checkpoint_dir,
        'batch_size': 16,
        'epochs': args.epochs
    }

    trainer = ProgressiveTrainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()