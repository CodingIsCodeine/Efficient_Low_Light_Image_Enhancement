"""
Model Export Script for ADLNet

Supports exporting to:
- ONNX (cross-platform)
- TensorFlow Lite (mobile/edge)
- Torchscript (production PyTorch)

Includes size verification to ensure <1MB constraint
"""

import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

from model import ADLNet


# ============================================================================
# Size Verification
# ============================================================================

def get_model_size_mb(model_path):
    """Get model file size in MB"""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


def verify_size_constraint(model_path, max_size_mb=1.0):
    """Verify model meets size constraint"""
    size_mb = get_model_size_mb(model_path)
    
    status = "✓ PASS" if size_mb < max_size_mb else "✗ FAIL"
    
    print(f"\nModel Size Verification:")
    print(f"  File: {os.path.basename(model_path)}")
    print(f"  Size: {size_mb:.4f} MB")
    print(f"  Constraint: < {max_size_mb} MB")
    print(f"  Status: {status}")
    
    return size_mb < max_size_mb


# ============================================================================
# PyTorch Model Export
# ============================================================================

def export_pytorch(model, save_path):
    """
    Export PyTorch model (standard .pth)
    
    Args:
        model: ADLNet model
        save_path: Path to save .pth file
    """
    # Save model state dict only (smallest format)
    torch.save(model.state_dict(), save_path)
    
    print(f"\n✓ Exported PyTorch model to {save_path}")
    verify_size_constraint(save_path)
    
    return save_path


# ============================================================================
# ONNX Export
# ============================================================================

def export_onnx(model, save_path, input_size=(1, 3, 256, 256)):
    """
    Export model to ONNX format
    
    Args:
        model: ADLNet model
        save_path: Path to save .onnx file
        input_size: Input tensor size (B, C, H, W)
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"\n✓ Exported ONNX model to {save_path}")
    verify_size_constraint(save_path)
    
    # Test ONNX inference
    print("\nTesting ONNX inference...")
    ort_session = ort.InferenceSession(save_path)
    
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare with PyTorch output
    with torch.no_grad():
        torch_output = model(dummy_input)
    
    max_diff = np.abs(ort_outputs[0] - torch_output.numpy()).max()
    print(f"  Max difference (ONNX vs PyTorch): {max_diff:.6f}")
    print(f"  {'✓ ONNX inference working!' if max_diff < 1e-5 else '✗ Large difference detected'}")
    
    return save_path


# ============================================================================
# TorchScript Export
# ============================================================================

def export_torchscript(model, save_path, input_size=(1, 3, 256, 256)):
    """
    Export model to TorchScript
    
    Args:
        model: ADLNet model
        save_path: Path to save .pt file
        input_size: Input tensor size
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save
    traced_model.save(save_path)
    
    print(f"\n✓ Exported TorchScript model to {save_path}")
    verify_size_constraint(save_path)
    
    # Test TorchScript inference
    print("\nTesting TorchScript inference...")
    loaded_model = torch.jit.load(save_path)
    
    with torch.no_grad():
        torch_output = model(dummy_input)
        script_output = loaded_model(dummy_input)
    
    max_diff = torch.abs(torch_output - script_output).max().item()
    print(f"  Max difference (TorchScript vs PyTorch): {max_diff:.6f}")
    print(f"  {'✓ TorchScript inference working!' if max_diff < 1e-5 else '✗ Large difference detected'}")
    
    return save_path


# ============================================================================
# TensorFlow Lite Export (Optional)
# ============================================================================

def export_tflite(onnx_path, save_path):
    """
    Convert ONNX to TensorFlow Lite
    
    Requires: onnx-tf, tensorflow
    
    Args:
        onnx_path: Path to ONNX model
        save_path: Path to save .tflite file
    """
    try:
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph('temp_tf_model')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model('temp_tf_model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"\n✓ Exported TFLite model to {save_path}")
        verify_size_constraint(save_path)
        
        # Cleanup
        import shutil
        if os.path.exists('temp_tf_model'):
            shutil.rmtree('temp_tf_model')
        
        return save_path
        
    except ImportError:
        print("\n✗ TFLite export requires onnx-tf and tensorflow")
        print("  Install with: pip install onnx-tf tensorflow")
        return None


# ============================================================================
# Complete Export Pipeline
# ============================================================================

def export_all_formats(model_path, output_dir='./exported_models'):
    """
    Export model to all supported formats
    
    Args:
        model_path: Path to trained PyTorch model (.pth)
        output_dir: Directory to save exported models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = ADLNet()
    state_dict = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\nModel loaded from {model_path}")
    print(f"Total parameters: {model.count_parameters():,}")
    
    print("\n" + "="*60)
    print("Exporting Model to All Formats")
    print("="*60)
    
    # 1. PyTorch (optimized)
    pytorch_path = os.path.join(output_dir, 'adlnet_model.pth')
    export_pytorch(model, pytorch_path)
    
    # 2. ONNX
    onnx_path = os.path.join(output_dir, 'adlnet_model.onnx')
    export_onnx(model, onnx_path)
    
    # 3. TorchScript
    torchscript_path = os.path.join(output_dir, 'adlnet_model.pt')
    export_torchscript(model, torchscript_path)
    
    # 4. TFLite (optional)
    tflite_path = os.path.join(output_dir, 'adlnet_model.tflite')
    export_tflite(onnx_path, tflite_path)
    
    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    
    # Summary
    print("\nExported Models:")
    for fmt, path in [
        ('PyTorch', pytorch_path),
        ('ONNX', onnx_path),
        ('TorchScript', torchscript_path),
        ('TFLite', tflite_path)
    ]:
        if os.path.exists(path):
            size = get_model_size_mb(path)
            print(f"  {fmt:12s}: {path} ({size:.4f} MB)")
    
    return {
        'pytorch': pytorch_path,
        'onnx': onnx_path,
        'torchscript': torchscript_path,
        'tflite': tflite_path if os.path.exists(tflite_path) else None
    }


# ============================================================================
# Benchmark Inference Speed
# ============================================================================

def benchmark_inference(model, num_runs=100, input_size=(1, 3, 256, 256)):
    """
    Benchmark inference speed
    
    Args:
        model: ADLNet model
        num_runs: Number of inference runs
        input_size: Input tensor size
    
    Returns:
        avg_time: Average inference time (seconds)
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    dummy_input = torch.randn(*input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nInference Benchmark ({num_runs} runs):")
    print(f"  Average time: {avg_time*1000:.2f} ms")
    print(f"  Std dev:      {std_time*1000:.2f} ms")
    print(f"  FPS:          {1.0/avg_time:.2f}")
    
    return avg_time


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Export ADLNet model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pth)')
    parser.add_argument('--output_dir', type=str, default='./exported_models',
                        help='Output directory for exported models')
    parser.add_argument('--format', type=str, default='all',
                        choices=['all', 'pytorch', 'onnx', 'torchscript', 'tflite'],
                        help='Export format')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark')
    
    args = parser.parse_args()
    
    if args.format == 'all':
        export_all_formats(args.model, args.output_dir)
    else:
        # Export specific format
        model = ADLNet()
        state_dict = torch.load(args.model, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        model.eval()
        
        output_path = os.path.join(args.output_dir, f'adlnet_model')
        
        if args.format == 'pytorch':
            export_pytorch(model, output_path + '.pth')
        elif args.format == 'onnx':
            export_onnx(model, output_path + '.onnx')
        elif args.format == 'torchscript':
            export_torchscript(model, output_path + '.pt')
        elif args.format == 'tflite':
            onnx_path = output_path + '.onnx'
            export_onnx(model, onnx_path)
            export_tflite(onnx_path, output_path + '.tflite')
    
    if args.benchmark:
        model = ADLNet()
        state_dict = torch.load(args.model, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        benchmark_inference(model)


if __name__ == "__main__":
    # Example usage:
    # python export.py --model checkpoints/best_model.pth --format all
    # python export.py --model checkpoints/best_model.pth --format onnx --benchmark
    
    main()
