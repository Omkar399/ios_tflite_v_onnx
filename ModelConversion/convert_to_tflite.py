#!/usr/bin/env python3
"""
Independent TFLite Conversion Script
Converts MobileNetV3 to TFLite format (FP32, FP16, INT8)
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf


def convert_mobilenetv3_tflite(output_dir="./models", model_size="small"):
    """
    Convert MobileNetV3 to TFLite format (FP32, FP16, INT8)
    """
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
    
    # Convert FP32 with FIXED input shapes (critical for Core ML delegate)
    print("\n📦 Converting to FP32 (fixed shapes for Core ML)...")
    
    # Create a concrete function with fixed input shape
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
    def model_fn(x):
        return model(x, training=False)
    
    concrete_func = model_fn.get_concrete_function()
    converter_fp32 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter_fp32.optimizations = []
    tflite_fp32 = converter_fp32.convert()
    
    fp32_path = os.path.join(output_dir, f"{base_name}_fp32.tflite")
    with open(fp32_path, 'wb') as f:
        f.write(tflite_fp32)
    print(f"   ✅ Saved: {fp32_path} ({len(tflite_fp32) / 1024 / 1024:.2f} MB)")
    
    # Convert FP16 with FIXED input shapes
    print("\n📦 Converting to FP16 (fixed shapes for Core ML)...")
    converter_fp16 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter_fp16.convert()
    
    fp16_path = os.path.join(output_dir, f"{base_name}_fp16.tflite")
    with open(fp16_path, 'wb') as f:
        f.write(tflite_fp16)
    print(f"   ✅ Saved: {fp16_path} ({len(tflite_fp16) / 1024 / 1024:.2f} MB)")
    
    # Convert INT8
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
        print(f"   ✅ Saved: {int8_path} ({len(tflite_int8) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"   ⚠️  INT8 conversion failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ TFLite conversion complete!")
    print('='*60)
    
    # Verify the models
    print("\n📋 Verifying models...")
    for tflite_file in [fp32_path, fp16_path]:
        if os.path.exists(tflite_file):
            try:
                interpreter = tf.lite.Interpreter(model_path=tflite_file)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print(f"   ✅ {os.path.basename(tflite_file)}: "
                      f"Input {input_details[0]['shape']}, "
                      f"Output {output_details[0]['shape']}")
            except Exception as e:
                print(f"   ❌ {os.path.basename(tflite_file)}: {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert MobileNetV3 to TFLite formats"
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
    
    print("🚀 TFLite Model Conversion Script")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Model size: {args.model_size}")
    
    convert_mobilenetv3_tflite(args.output_dir, args.model_size)
    
    print(f"\n{'='*60}")
    print("🎉 All conversions complete!")
    print('='*60)
    print(f"\nNext steps:")
    print(f"1. Copy the .tflite files to your iOS Xcode project")
    print(f"2. Ensure files are added to the app bundle (Target Membership)")


if __name__ == "__main__":
    main()

