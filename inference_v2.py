# """
# Improved Inference for High-Resolution Images

# CRITICAL FIXES:
# 1. Handles 3024x4032 images properly (not just 256x256)
# 2. Tile-based processing for large images
# 3. Better memory management
# 4. Works with SCALE-Net
# """

# import torch
# import numpy as np
# from PIL import Image
# import argparse
# from pathlib import Path
# import time
# from tqdm import tqdm

# from model_v2_novel import SCALENet


# class ImprovedInference:
#     """
#     Inference engine optimized for high-resolution images
    
#     Key improvements:
#     - Tile-based processing for large images (>1024x1024)
#     - Overlap-tile fusion to avoid artifacts
#     - Multi-scale processing option
#     - Memory-efficient
#     """
#     def __init__(self, model_path, device='cuda', tile_size=512, overlap=32):
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.tile_size = tile_size
#         self.overlap = overlap
        
#         # Load model
#         self.model = SCALENet(base_channels=32).to(self.device)
        
#         if Path(model_path).exists():
#             state_dict = torch.load(model_path, map_location=self.device)
#             if 'model_state_dict' in state_dict:
#                 state_dict = state_dict['model_state_dict']
#             self.model.load_state_dict(state_dict)
#             print(f"✓ Loaded model from {model_path}")
#         else:
#             raise FileNotFoundError(f"Model not found: {model_path}")
        
#         self.model.eval()
#         print(f"✓ Model loaded on {self.device}")
    
#     def preprocess(self, image_path):
#         """Load and preprocess image"""
#         img = Image.open(image_path).convert('RGB')
#         img_np = np.array(img).astype(np.float32) / 255.0
#         return img, img_np
    
#     def postprocess(self, img_tensor):
#         """Convert tensor to PIL Image"""
#         img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#         img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
#         return Image.fromarray(img_np)
    
#     @torch.no_grad()
#     def enhance_tile(self, tile_np):
#         """Enhance a single tile"""
#         tile_tensor = torch.from_numpy(tile_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
#         enhanced_tensor = self.model(tile_tensor)
#         return enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
#     @torch.no_grad()
#     def enhance_large_image(self, img_np):
#         """
#         Enhance large image using tile-based processing
        
#         For images > 1024x1024, split into tiles with overlap
#         Fuse tiles smoothly to avoid boundary artifacts
#         """
#         h, w, c = img_np.shape
        
#         # If image is small enough, process directly
#         if h <= 1024 and w <= 1024:
#             tile_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
#             enhanced_tensor = self.model(tile_tensor)
#             return enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
#         # Otherwise, use tile-based processing
#         print(f"  Large image ({h}x{w}), using tile-based processing...")
        
#         tile_size = self.tile_size
#         overlap = self.overlap
#         stride = tile_size - overlap
        
#         # Calculate number of tiles
#         n_tiles_h = (h - overlap) // stride + (1 if (h - overlap) % stride > 0 else 0)
#         n_tiles_w = (w - overlap) // stride + (1 if (w - overlap) % stride > 0 else 0)
        
#         # Output accumulator
#         output = np.zeros_like(img_np)
#         weight_map = np.zeros((h, w, 1))
        
#         # Process each tile
#         for i in tqdm(range(n_tiles_h), desc="  Processing tiles", leave=False):
#             for j in range(n_tiles_w):
#                 # Calculate tile coordinates
#                 y1 = i * stride
#                 x1 = j * stride
#                 y2 = min(y1 + tile_size, h)
#                 x2 = min(x1 + tile_size, w)
                
#                 # Adjust if at boundary
#                 if y2 == h:
#                     y1 = h - tile_size
#                 if x2 == w:
#                     x1 = w - tile_size
                
#                 # Extract tile
#                 tile = img_np[y1:y2, x1:x2, :]
                
#                 # Enhance tile
#                 enhanced_tile = self.enhance_tile(tile)
                
#                 # Create smooth weight map (avoid hard boundaries)
#                 tile_h, tile_w = enhanced_tile.shape[:2]
#                 weight = np.ones((tile_h, tile_w, 1))
                
#                 # Fade at boundaries
#                 if overlap > 0:
#                     # Fade top
#                     if y1 > 0:
#                         fade = np.linspace(0, 1, overlap)[:, np.newaxis, np.newaxis]
#                         weight[:overlap, :, :] *= fade
                    
#                     # Fade left
#                     if x1 > 0:
#                         fade = np.linspace(0, 1, overlap)[np.newaxis, :, np.newaxis]
#                         weight[:, :overlap, :] *= fade
                    
#                     # Fade bottom
#                     if y2 < h:
#                         fade = np.linspace(1, 0, overlap)[:, np.newaxis, np.newaxis]
#                         weight[-overlap:, :, :] *= fade
                    
#                     # Fade right
#                     if x2 < w:
#                         fade = np.linspace(1, 0, overlap)[np.newaxis, :, np.newaxis]
#                         weight[:, -overlap:, :] *= fade
                
#                 # Accumulate
#                 output[y1:y2, x1:x2, :] += enhanced_tile * weight
#                 weight_map[y1:y2, x1:x2, :] += weight
        
#         # Normalize by weight
#         output = output / (weight_map + 1e-8)
        
#         return output
    
#     def enhance_image(self, image_path, output_path=None):
#         """
#         Enhance a single image
        
#         Automatically chooses:
#         - Direct processing for small images
#         - Tile-based processing for large images
#         """
#         print(f"\nProcessing: {image_path}")
        
#         # Load
#         img_pil, img_np = self.preprocess(image_path)
#         h, w = img_np.shape[:2]
#         print(f"  Resolution: {w}x{h}")
        
#         # Enhance
#         start_time = time.time()
#         enhanced_np = self.enhance_large_image(img_np)
#         inference_time = time.time() - start_time
        
#         print(f"  Inference time: {inference_time:.2f}s")
        
#         # Convert to PIL
#         enhanced_pil = Image.fromarray((enhanced_np * 255).astype(np.uint8))
        
#         # Save
#         if output_path:
#             Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#             enhanced_pil.save(output_path, quality=95)
#             print(f"  ✓ Saved to: {output_path}")
        
#         return enhanced_pil, inference_time
    
#     def enhance_directory(self, input_dir, output_dir):
#         """Enhance all images in directory"""
#         input_dir = Path(input_dir)
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Find all images
#         image_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
#         image_paths = []
#         for ext in image_exts:
#             image_paths.extend(input_dir.glob(f'*{ext}'))
        
#         print(f"\nFound {len(image_paths)} images in {input_dir}")
        
#         total_time = 0
#         for img_path in image_paths:
#             output_path = output_dir / f"enhanced_{img_path.name}"
#             _, inference_time = self.enhance_image(str(img_path), str(output_path))
#             total_time += inference_time
        
#         print(f"\n{'='*60}")
#         print(f"Batch processing complete!")
#         print(f"  Total images: {len(image_paths)}")
#         print(f"  Total time: {total_time:.2f}s")
#         print(f"  Average time: {total_time/len(image_paths):.2f}s per image")
#         print(f"{'='*60}")


# def main():
#     parser = argparse.ArgumentParser(description='SCALE-Net Inference')
    
#     parser.add_argument('--model', type=str, required=True,
#                         help='Path to trained model')
#     parser.add_argument('--input', type=str, required=True,
#                         help='Input image or directory')
#     parser.add_argument('--output', type=str, required=True,
#                         help='Output image or directory')
#     parser.add_argument('--device', type=str, default='cuda',
#                         choices=['cuda', 'cpu'])
#     parser.add_argument('--tile_size', type=int, default=512,
#                         help='Tile size for large images')
#     parser.add_argument('--batch', action='store_true',
#                         help='Process directory')
    
#     args = parser.parse_args()
    
#     # Initialize
#     inference = ImprovedInference(
#         args.model,
#         device=args.device,
#         tile_size=args.tile_size
#     )
    
#     # Process
#     if args.batch:
#         inference.enhance_directory(args.input, args.output)
#     else:
#         inference.enhance_image(args.input, args.output)


# if __name__ == "__main__":
#     # Example usage:
#     # python inference_v2.py --model checkpoints_v2/best_model.pth --input test.jpg --output result.jpg
#     # python inference_v2.py --model checkpoints_v2/best_model.pth --input ./test_dir --output ./results --batch
    
#     main()


###################################################################################
#16th March Claude Updated Code Version
###################################################################################

import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import time
from tqdm import tqdm

from model_v2_novel import SCALENet


class ImprovedInference:
    def __init__(self, model_path, device='cuda', tile_size=512, overlap=32):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tile_size = tile_size
        self.overlap = overlap

        self.model = SCALENet(base_channels=32).to(self.device)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'ema_state_dict' in state_dict:
            state_dict = state_dict['ema_state_dict']

        # Strip DataParallel prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Model loaded from {model_path} on {self.device}")

    def preprocess(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        return img, img_np

    @torch.no_grad()
    def enhance_tile(self, tile_np):
        t = torch.from_numpy(tile_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(t)
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

    @torch.no_grad()
    def enhance_large_image(self, img_np):
        h, w, _ = img_np.shape

        if h <= 1024 and w <= 1024:
            t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            out = self.model(t)
            return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

        print(f"  Tiled processing ({h}x{w})...")
        tile_size = self.tile_size
        overlap = self.overlap
        stride = tile_size - overlap

        n_h = max(1, (h - overlap + stride - 1) // stride)
        n_w = max(1, (w - overlap + stride - 1) // stride)

        output = np.zeros_like(img_np)
        weight_map = np.zeros((h, w, 1))

        for i in tqdm(range(n_h), desc="  Tiles", leave=False):
            for j in range(n_w):
                y1 = min(i * stride, max(0, h - tile_size))
                x1 = min(j * stride, max(0, w - tile_size))
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)

                tile = img_np[y1:y2, x1:x2, :]
                enhanced = self.enhance_tile(tile)

                th, tw = enhanced.shape[:2]
                weight = np.ones((th, tw, 1))

                if overlap > 0:
                    if y1 > 0:
                        weight[:overlap] *= np.linspace(0, 1, overlap)[:, None, None]
                    if x1 > 0:
                        weight[:, :overlap] *= np.linspace(0, 1, overlap)[None, :, None]
                    if y2 < h:
                        weight[-overlap:] *= np.linspace(1, 0, overlap)[:, None, None]
                    if x2 < w:
                        weight[:, -overlap:] *= np.linspace(1, 0, overlap)[None, :, None]

                output[y1:y2, x1:x2] += enhanced * weight
                weight_map[y1:y2, x1:x2] += weight

        return output / (weight_map + 1e-8)

    def enhance_image(self, image_path, output_path=None):
        print(f"\nProcessing: {image_path}")
        _, img_np = self.preprocess(image_path)
        h, w = img_np.shape[:2]
        print(f"  Resolution: {w}x{h}")

        t0 = time.time()
        enhanced_np = self.enhance_large_image(img_np)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.2f}s")

        enhanced_pil = Image.fromarray((np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8))

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            enhanced_pil.save(output_path, quality=95)
            print(f"  Saved: {output_path}")

        return enhanced_pil, elapsed

    def enhance_directory(self, input_dir, output_dir):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        paths = [p for e in exts for p in input_dir.glob(f'*{e}')]
        print(f"Found {len(paths)} images in {input_dir}")

        total_time = 0
        for p in paths:
            _, t = self.enhance_image(str(p), str(output_dir / f"enhanced_{p.name}"))
            total_time += t

        print(f"\nDone: {len(paths)} images in {total_time:.1f}s "
              f"({total_time/max(len(paths),1):.2f}s avg)")


def main():
    parser = argparse.ArgumentParser(description='SCALE-Net Inference')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--batch', action='store_true')
    args = parser.parse_args()

    engine = ImprovedInference(args.model, device=args.device,
                               tile_size=args.tile_size, overlap=args.overlap)

    if args.batch:
        engine.enhance_directory(args.input, args.output)
    else:
        engine.enhance_image(args.input, args.output)


if __name__ == "__main__":
    main()