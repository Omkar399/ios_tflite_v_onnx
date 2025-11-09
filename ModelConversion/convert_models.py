#!/usr/bin/env python3
"""
Model Conversion Script for TFLite vs ORT Benchmark
Converts MobileNetV3 to both TFLite and ONNX formats with multiple precisions
"""

import os
import sys
import argparse
import numpy as np

# NOTE: Libraries are imported lazily within functions to avoid
# threading conflicts on macOS ARM (TensorFlow + PyTorch + onnxsim)

def check_packages():
    """Check if required packages are installed"""
    # Import each package separately to avoid threading conflicts
    packages = ['tensorflow', 'torch', 'torchvision', 'onnx', 'onnxsim', 'onnxruntime']
    missing = []
    
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("\nInstall requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    return True


def convert_mobilenetv3_tflite(output_dir="./models", model_size="small"):
    """
    Convert MobileNetV3 to TFLite format (FP32, FP16)
    """
    # Lazy import to avoid threading conflicts
    import tensorflow as tf
    
    print(f"\n{'='*60}")
    print(f"Converting MobileNetV3-{model_size.upper()} to TFLite")
    print('='*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pre-trained model
    if model_size.lower() == "small":
        model = tf.keras.applications.MobileNetV3Small(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        base_name = "mobilenetv3_small"
    else:
        model = tf.keras.applications.MobileNetV3Large(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        base_name = "mobilenetv3_large"
    
    print(f"✅ Loaded {base_name} from Keras Applications")
    
    # Convert FP32
    print("\n📦 Converting to FP32...")
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp32.optimizations = []
    tflite_fp32 = converter_fp32.convert()
    
    fp32_path = os.path.join(output_dir, f"{base_name}_fp32.tflite")
    with open(fp32_path, 'wb') as f:
        f.write(tflite_fp32)
    print(f"   Saved: {fp32_path} ({len(tflite_fp32) / 1024 / 1024:.2f} MB)")
    
    # Convert FP16
    print("\n📦 Converting to FP16...")
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter_fp16.convert()
    
    fp16_path = os.path.join(output_dir, f"{base_name}_fp16.tflite")
    with open(fp16_path, 'wb') as f:
        f.write(tflite_fp16)
    print(f"   Saved: {fp16_path} ({len(tflite_fp16) / 1024 / 1024:.2f} MB)")
    
    # Convert INT8 (optional, requires representative dataset)
    print("\n📦 Converting to INT8 (post-training quantization)...")
    
    def representative_dataset():
        """Generate representative data for quantization"""
        for _ in range(100):
            data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            yield [data]
    
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.uint8
    converter_int8.inference_output_type = tf.uint8
    
    try:
        tflite_int8 = converter_int8.convert()
        int8_path = os.path.join(output_dir, f"{base_name}_int8.tflite")
        with open(int8_path, 'wb') as f:
            f.write(tflite_int8)
        print(f"   Saved: {int8_path} ({len(tflite_int8) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"   ⚠️  INT8 conversion failed: {e}")
    
    print(f"\n✅ TFLite conversion complete!")
    return True


def convert_mobilenetv3_onnx(output_dir="./models", model_size="small"):
    """
    Convert MobileNetV3 to ONNX format (FP32, FP16)
    """
    # Lazy import to avoid threading conflicts
    import torch
    import torchvision.models as models
    import onnx
    from onnxsim import simplify
    from onnx import helper, numpy_helper
    
    print(f"\n{'='*60}")
    print(f"Converting MobileNetV3-{model_size.upper()} to ONNX")
    print('='*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pre-trained PyTorch model
    if model_size.lower() == "small":
        model = models.mobilenet_v3_small(pretrained=True)
        base_name = "mobilenetv3_small"
    else:
        model = models.mobilenet_v3_large(pretrained=True)
        base_name = "mobilenetv3_large"
    
    model.eval()
    print(f"✅ Loaded {base_name} from torchvision")
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX (FP32)
    print("\n📦 Converting to ONNX (FP32)...")
    fp32_path = os.path.join(output_dir, f"{base_name}_fp32.onnx")
    
    # Disable dynamo for better stability on macOS ARM
    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        export_params=True,
        opset_version=17,  # Use opset 17 for better compatibility  
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # Disable dynamo to avoid crashes on macOS ARM
    )
    
    # Simplify ONNX model
    print("   Simplifying ONNX model...")
    onnx_model = onnx.load(fp32_path)
    model_simplified, check = simplify(onnx_model)
    if check:
        onnx.save(model_simplified, fp32_path)
        print(f"   ✅ Simplified and saved: {fp32_path}")
    else:
        print(f"   ⚠️  Simplification failed, using original model")
    
    file_size = os.path.getsize(fp32_path) / 1024 / 1024
    print(f"   Saved: {fp32_path} ({file_size:.2f} MB)")
    
    # Convert to FP16
    print("\n📦 Converting to FP16...")
    fp16_path = os.path.join(output_dir, f"{base_name}_fp16.onnx")
    
    # Use ONNX's built-in fp16 conversion
    try:
        from onnxconverter_common import float16
        onnx_fp16 = float16.convert_float_to_float16(onnx_model)
    except ImportError:
        # Fallback: use onnxmltools if available
        try:
            from onnxmltools.utils import float16_converter
            onnx_fp16 = float16_converter.convert_float_to_float16(onnx_model)
        except (ImportError, AttributeError):
            print("   ⚠️  FP16 conversion not available, skipping...")
            onnx_fp16 = None
    
    if onnx_fp16:
        onnx.save(onnx_fp16, fp16_path)
        file_size = os.path.getsize(fp16_path) / 1024 / 1024
        print(f"   Saved: {fp16_path} ({file_size:.2f} MB)")
    
    print(f"\n✅ ONNX conversion complete!")
    return True


def verify_models(models_dir="./models"):
    """
    Verify that converted models can be loaded and run inference
    """
    # Lazy import to avoid threading conflicts
    import tensorflow as tf
    import onnxruntime as ort
    
    print(f"\n{'='*60}")
    print("Verifying Models")
    print('='*60)
    
    # Test TFLite models
    tflite_files = [f for f in os.listdir(models_dir) if f.endswith('.tflite')]
    for tflite_file in tflite_files:
        path = os.path.join(models_dir, tflite_file)
        try:
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Run dummy inference
            dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            print(f"✅ {tflite_file}: Input {input_details[0]['shape']}, Output {output.shape}")
        except Exception as e:
            print(f"❌ {tflite_file}: {e}")
    
    # Test ONNX models
    onnx_files = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
    for onnx_file in onnx_files:
        path = os.path.join(models_dir, onnx_file)
        try:
            session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # Run dummy inference
            dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
            output = session.run([output_name], {input_name: dummy_input})
            
            print(f"✅ {onnx_file}: Input (1,3,224,224), Output {output[0].shape}")
        except Exception as e:
            print(f"❌ {onnx_file}: {e}")
    
    print("\n✅ Model verification complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MobileNetV3 to TFLite and ONNX formats"
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
    parser.add_argument(
        "--skip-tflite",
        action="store_true",
        help="Skip TFLite conversion"
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX conversion"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify converted models"
    )
    
    args = parser.parse_args()
    
    # Note: Package checking is disabled to avoid threading conflicts on macOS
    # Packages will be imported lazily when needed
    
    print("🚀 Model Conversion Script")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Model size: {args.model_size}")
    
    # Convert to TFLite
    if not args.skip_tflite:
        convert_mobilenetv3_tflite(args.output_dir, args.model_size)
    
    # Convert to ONNX
    if not args.skip_onnx:
        convert_mobilenetv3_onnx(args.output_dir, args.model_size)
    
    # Verify models
    if args.verify:
        verify_models(args.output_dir)
    
    print(f"\n{'='*60}")
    print("🎉 All conversions complete!")
    print('='*60)
    print(f"\nNext steps:")
    print(f"1. Copy the .tflite files to TFLiteCoreMLDemo Xcode project")
    print(f"2. Copy the .onnx files to ORTCoreMLDemo Xcode project")
    print(f"3. Ensure files are added to the app bundle (Target Membership)")
    print(f"4. Build and run benchmarks on device")


if __name__ == "__main__":
    main()

