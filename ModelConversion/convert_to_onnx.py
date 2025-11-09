#!/usr/bin/env python3
"""
Independent ONNX Conversion Script
Converts MobileNetV3 to ONNX format (FP32, FP16)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchvision.models as models


def convert_mobilenetv3_onnx(output_dir="./models", model_size="small"):
    """
    Convert MobileNetV3 to ONNX format (FP32, FP16)
    """
    print(f"\n{'='*60}")
    print(f"Converting MobileNetV3-{model_size.upper()} to ONNX")
    print('='*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pre-trained PyTorch model
    if model_size.lower() == "small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        base_name = "mobilenetv3_small"
    else:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        base_name = "mobilenetv3_large"
    
    model.eval()
    print(f"✅ Loaded {base_name} from torchvision")
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX (FP32)
    print("\n📦 Converting to ONNX (FP32)...")
    fp32_path = os.path.join(output_dir, f"{base_name}_fp32.onnx")
    
    # Use simple export without dynamic axes to avoid crashes
    # Note: PyTorch 2.1 supports up to opset 17
    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        export_params=True,
        opset_version=17,  # Use opset 17 (maximum for PyTorch 2.1)
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    file_size = os.path.getsize(fp32_path) / 1024 / 1024
    print(f"   ✅ Saved: {fp32_path} ({file_size:.2f} MB)")
    
    # Simplify ONNX model (optional)
    try:
        import onnx
        from onnxsim import simplify
        
        print("   📦 Simplifying ONNX model...")
        onnx_model = onnx.load(fp32_path)
        model_simplified, check = simplify(onnx_model)
        if check:
            onnx.save(model_simplified, fp32_path)
            print(f"   ✅ Simplified and saved")
        else:
            print(f"   ⚠️  Simplification check failed, using original model")
    except Exception as e:
        print(f"   ⚠️  Simplification failed: {e}")
    
    # Convert to FP16
    print("\n📦 Converting to FP16...")
    fp16_path = os.path.join(output_dir, f"{base_name}_fp16.onnx")
    
    try:
        import onnx
        from onnxconverter_common import float16
        
        onnx_model = onnx.load(fp32_path)
        onnx_fp16 = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_fp16, fp16_path)
        
        file_size = os.path.getsize(fp16_path) / 1024 / 1024
        print(f"   ✅ Saved: {fp16_path} ({file_size:.2f} MB)")
    except Exception as e:
        print(f"   ⚠️  FP16 conversion failed: {e}")
        print("   💡 Tip: Install with: pip install onnxconverter-common")
    
    print(f"\n{'='*60}")
    print(f"✅ ONNX conversion complete!")
    print('='*60)
    
    # Verify the models
    print("\n📋 Verifying models...")
    try:
        import onnxruntime as ort
        
        for onnx_file in [fp32_path, fp16_path]:
            if os.path.exists(onnx_file):
                try:
                    session = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
                    input_name = session.get_inputs()[0].name
                    input_shape = session.get_inputs()[0].shape
                    output_shape = session.get_outputs()[0].shape
                    print(f"   ✅ {os.path.basename(onnx_file)}: "
                          f"Input {input_shape}, Output {output_shape}")
                except Exception as e:
                    print(f"   ❌ {os.path.basename(onnx_file)}: {e}")
    except ImportError:
        print("   ⚠️  onnxruntime not available for verification")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert MobileNetV3 to ONNX formats"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for converted models"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "large"],
        default="small",
        help="Model size: small or large"
    )
    
    args = parser.parse_args()
    
    print("🚀 ONNX Model Conversion Script")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Model size: {args.model_size}")
    
    convert_mobilenetv3_onnx(args.output_dir, args.model_size)
    
    print(f"\n{'='*60}")
    print("🎉 All conversions complete!")
    print('='*60)
    print(f"\nNext steps:")
    print(f"1. Copy the .onnx files to your iOS Xcode project")
    print(f"2. Ensure files are added to the app bundle (Target Membership)")


if __name__ == "__main__":
    main()

