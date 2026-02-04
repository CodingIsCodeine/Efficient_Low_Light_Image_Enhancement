# ADLNet Usage Guide

Complete guide for training, inference, and deployment of ADLNet.

---

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Inference](#inference)
5. [Evaluation](#evaluation)
6. [Model Export](#model-export)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Clone Repository (or download files)

```bash
cd /path/to/adlnet
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n adlnet python=3.8
conda activate adlnet

# Or using venv
python -m venv adlnet_env
source adlnet_env/bin/activate  # Linux/Mac
# adlnet_env\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "from model import ADLNet; m = ADLNet(); print(f'Model params: {m.count_parameters():,}')"
```

Expected output:
```
PyTorch version: 1.x.x
Model params: ~170,000
```

---

## Data Preparation

### Expected Data Structure

```
data/
├── train/
│   ├── low/
│   │   ├── img001.png
│   │   ├── img002.png
│   │   └── ...
│   └── high/
│       ├── img001.png
│       ├── img002.png
│       └── ...
├── val/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
```

### Data Requirements

1. **Paired Images**: Each image in `low/` must have corresponding image in `high/`
2. **Same Filename**: Corresponding images must have identical filenames
3. **Format**: PNG or JPG (PNG preferred for lossless quality)
4. **Resolution**: Any resolution (will be resized to 256×256 during training)
5. **Naming**: Consistent naming scheme (e.g., `img001.png`, `img002.png`)

### Preparing Your Dataset

#### Option 1: LOL Dataset (Low-Light Dataset)

```bash
# Download LOL dataset
wget https://daooshee.github.io/BMVC2018website/LOLdataset.zip
unzip LOLdataset.zip

# Organize
mkdir -p data/train/low data/train/high
mkdir -p data/val/low data/val/high

# Move files (example)
cp LOLdataset/our485/low/* data/train/low/
cp LOLdataset/our485/high/* data/train/high/
cp LOLdataset/eval15/low/* data/val/low/
cp LOLdataset/eval15/high/* data/val/high/
```

#### Option 2: Custom Dataset

```python
# If you have unpaired data, you can create synthetic low-light images
from PIL import Image
import numpy as np

def create_low_light(high_img_path, gamma=0.4):
    """Create synthetic low-light image"""
    img = Image.open(high_img_path)
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Apply gamma correction to simulate low-light
    low_img_np = np.power(img_np, gamma)
    
    # Add noise
    noise = np.random.normal(0, 0.02, img_np.shape)
    low_img_np = np.clip(low_img_np + noise, 0, 1)
    
    low_img = Image.fromarray((low_img_np * 255).astype(np.uint8))
    return low_img

# Process your dataset
import os
from pathlib import Path

high_dir = Path('my_images/')
low_dir = Path('data/train/low/')
low_dir.mkdir(parents=True, exist_ok=True)

for img_path in high_dir.glob('*.jpg'):
    low_img = create_low_light(str(img_path))
    low_img.save(low_dir / img_path.name)
```

### Data Verification

```bash
# Check data structure
python -c "
from pathlib import Path
train_low = len(list(Path('data/train/low').glob('*.png')))
train_high = len(list(Path('data/train/high').glob('*.png')))
val_low = len(list(Path('data/val/low').glob('*.png')))
val_high = len(list(Path('data/val/high').glob('*.png')))

print(f'Train: {train_low} low, {train_high} high')
print(f'Val: {val_low} low, {val_high} high')
assert train_low == train_high, 'Mismatch in training data!'
assert val_low == val_high, 'Mismatch in validation data!'
print('✓ Data structure verified!')
"
```

---

## Training

### Quick Start Training

```bash
# Train with default settings
python train.py
```

### Custom Training Configuration

Edit `config.yaml`:

```yaml
# Key parameters to adjust
data:
  data_root: "./data"  # Your data path

training:
  batch_size: 16       # Reduce if GPU memory limited
  num_epochs: 150      # Increase for better convergence
  learning_rate: 0.001 # Adjust based on dataset size
```

Then train:

```bash
python train.py
```

### Training from Checkpoint

```bash
# Resume training from latest checkpoint
python train.py --resume checkpoints/latest.pth
```

### Multi-GPU Training (if available)

```bash
# Distributed training on 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train.py
```

### Monitoring Training

#### Option 1: Console Output

```
Epoch 1/150
Loss: 0.1234, SSIM: 0.856, Perceptual: 0.234
...
```

#### Option 2: TensorBoard

```bash
# In another terminal
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### Training Tips

1. **Start Small**: Train for 20-30 epochs first, check results
2. **Monitor Validation**: If val loss stops improving, consider early stopping
3. **Save Best Model**: Script automatically saves best model based on val SSIM
4. **GPU Memory**: If OOM error, reduce batch_size in config.yaml

---

## Inference

### Single Image Enhancement

```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --input test_image.jpg \
    --output enhanced_image.jpg
```

### Batch Processing (Directory)

```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --input ./test_images \
    --output ./results \
    --batch
```

### Create Comparison Visualization

```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --input test_image.jpg \
    --output comparison.jpg \
    --visualize
```

This creates side-by-side comparison (input | output).

### Custom Resolution Processing

```bash
# Process at specific resolution
python inference.py \
    --model checkpoints/best_model.pth \
    --input test.jpg \
    --output enhanced.jpg \
    --size 512 512  # Height Width
```

### CPU Inference (No GPU)

```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --input test.jpg \
    --output enhanced.jpg \
    --device cpu
```

### Python API Usage

```python
from inference import ADLNetInference

# Initialize
model = ADLNetInference('checkpoints/best_model.pth', device='cuda')

# Enhance single image
enhanced_img, latency = model.enhance_image('input.jpg', 'output.jpg')
print(f"Inference time: {latency:.3f}s")

# Batch processing
model.enhance_batch(
    image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    output_dir='./results'
)

# Directory processing
model.enhance_directory(
    input_dir='./test_images',
    output_dir='./enhanced'
)
```

---

## Evaluation

### Evaluate on Test Set

```python
# evaluation.py
from model import ADLNet
from utils import evaluate_enhancement
import torch
from PIL import Image
import numpy as np
from pathlib import Path

# Load model
model = ADLNet()
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Evaluate
test_low = Path('data/test/low')
test_high = Path('data/test/high')

metrics_all = []

for img_path in test_low.glob('*.png'):
    # Load images
    low_img = Image.open(img_path)
    high_img = Image.open(test_high / img_path.name)
    
    # Preprocess
    low_tensor = torch.from_numpy(np.array(low_img).astype(np.float32) / 255.0)
    low_tensor = low_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        pred_tensor = model(low_tensor)
    
    # Postprocess
    pred_img = (pred_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    high_img_np = np.array(high_img)
    
    # Evaluate
    metrics = evaluate_enhancement(pred_img, high_img_np, verbose=False)
    metrics_all.append(metrics)

# Average metrics
avg_metrics = {k: np.mean([m[k] for m in metrics_all]) for k in metrics_all[0].keys()}

print("Test Set Results:")
print("="*50)
for k, v in avg_metrics.items():
    print(f"  {k:15s}: {v:.4f}")
print("="*50)
```

Run:
```bash
python evaluation.py
```

### Compute Specific Metrics

```python
from utils import calculate_ssim, calculate_psnr

# Load images
pred = np.array(Image.open('enhanced.jpg'))
gt = np.array(Image.open('ground_truth.jpg'))

# Calculate
ssim_score = calculate_ssim(pred, gt)
psnr_score = calculate_psnr(pred, gt)

print(f"SSIM: {ssim_score:.4f}")
print(f"PSNR: {psnr_score:.2f} dB")
```

---

## Model Export

### Export to All Formats

```bash
python export.py \
    --model checkpoints/best_model.pth \
    --format all \
    --output_dir exported_models
```

This creates:
- `adlnet_model.pth` (PyTorch)
- `adlnet_model.onnx` (ONNX)
- `adlnet_model.pt` (TorchScript)
- `adlnet_model.tflite` (TensorFlow Lite, if dependencies available)

### Export Specific Format

```bash
# ONNX only
python export.py \
    --model checkpoints/best_model.pth \
    --format onnx

# With inference benchmark
python export.py \
    --model checkpoints/best_model.pth \
    --format onnx \
    --benchmark
```

### Verify Exported Model

```python
# For ONNX
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('exported_models/adlnet_model.onnx')

# Test inference
dummy_input = np.random.rand(1, 3, 256, 256).astype(np.float32)
outputs = session.run(None, {'input': dummy_input})

print(f"Output shape: {outputs[0].shape}")
print("✓ ONNX model working!")
```

### Deploy on Mobile (TFLite)

```python
# load_tflite.py
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='exported_models/adlnet_model.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test inference
input_data = np.random.rand(1, 256, 256, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"TFLite output shape: {output_data.shape}")
```

---

## Troubleshooting

### Issue: Out of Memory (OOM) During Training

**Solution**:
```yaml
# In config.yaml, reduce batch size
training:
  batch_size: 8  # or 4, or 1

# Or use gradient accumulation
training:
  batch_size: 4
  accumulate_grad_steps: 4  # Effective batch size = 16
```

### Issue: Training Not Converging

**Checklist**:
1. ✓ Data pairs matched correctly?
2. ✓ Learning rate appropriate? (try 5e-4)
3. ✓ Sufficient training epochs? (150+)
4. ✓ Data augmentation helping? (try disabling gamma adjustment)

**Debug**:
```python
# Check data loading
from train import LLIEDataset
dataset = LLIEDataset('data', 'train')
low, high, name = dataset[0]
print(f"Low range: [{low.min()}, {low.max()}]")
print(f"High range: [{high.min()}, {high.max()}]")
```

### Issue: Model Size Exceeds 1MB

**Verify**:
```python
python -c "
from model import ADLNet
model = ADLNet()
params = model.count_parameters()
size_mb = (params * 4) / (1024 ** 2)
print(f'Parameters: {params:,}')
print(f'Size: {size_mb:.4f} MB')
assert size_mb < 1.0, 'Size constraint violated!'
"
```

**If exceeded, reduce parameters in model.py**:
```python
# Reduce base channels
LuminancePath(base_channels=20)  # instead of 24
ChrominancePath(base_channels=12)  # instead of 16
```

### Issue: Color Distortion in Results

**Possible causes**:
1. UV-channel loss weight too low
2. Color augmentation too aggressive

**Solutions**:
```yaml
# In config.yaml or losses.py
loss:
  weight_color: 0.5  # Increase from 0.3

# In train.py, reduce color jitter in augmentation
```

### Issue: Inference Too Slow

**Optimizations**:
1. Use ONNX runtime (2-3× faster)
2. Batch processing
3. Reduce resolution (process at 256×256, upscale after)
4. Use GPU if available

```python
# Batch inference
from inference import ADLNetInference

model = ADLNetInference('model.pth', device='cuda')
model.enhance_batch(image_paths, output_dir, target_size=(256, 256))
```

### Issue: ONNX Export Fails

**Common cause**: Unsupported operations

**Solution**:
```bash
# Update ONNX
pip install --upgrade onnx onnxruntime

# Try different opset version
# In export.py, change:
opset_version=11  # try 10, 11, 12, 13
```

---

## Advanced Usage

### Custom Loss Weights

```python
# In losses.py, modify ADLNetLoss.__init__():
self.w_perceptual = 1.5  # Increase perceptual importance
self.w_ssim = 1.5         # Reduce SSIM importance
# Retrain
```

### Fine-tuning on Custom Data

```bash
# Start from pretrained model
python train.py \
    --checkpoint checkpoints/best_model.pth \
    --learning_rate 1e-4 \  # Lower LR for fine-tuning
    --epochs 50              # Fewer epochs
```

### Visualize YUV Decomposition

```python
from utils import visualize_yuv_decomposition
from PIL import Image

img = Image.open('test.jpg')
img_np = np.array(img).astype(np.float32) / 255.0

visualize_yuv_decomposition(img_np, save_path='yuv_viz.png')
```

---

## Command Reference

### Training
```bash
python train.py                                    # Basic training
python train.py --resume checkpoints/latest.pth   # Resume training
```

### Inference
```bash
python inference.py --model M --input I --output O           # Single image
python inference.py --model M --input DIR --output DIR --batch  # Batch
python inference.py --model M --input I --output O --visualize  # Compare
```

### Export
```bash
python export.py --model M --format all              # All formats
python export.py --model M --format onnx --benchmark # ONNX + speed test
```

### Evaluation
```bash
python evaluation.py  # Custom script for test set evaluation
```

---

## Getting Help

1. **Check this guide first**
2. **Verify installation**: `python -c "from model import ADLNet; print('OK')"`
3. **Check data structure**: Files in correct directories?
4. **Read error messages**: Usually self-explanatory
5. **Review code comments**: Extensive documentation in source files

---

**Happy enhancing! 🚀**
