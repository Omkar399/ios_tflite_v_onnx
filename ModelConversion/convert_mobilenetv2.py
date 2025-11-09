#!/usr/bin/env python3
"""
Convert MobileNetV2 to TFLite and ONNX (Core ML compatible)
MobileNetV2 has better Core ML support than V3
"""

import os
import tensorflow as tf
import torch
import torchvision.models as models
import onnx
from onnxsim import simplify

def convert_mobilenetv2_tflite():
    """Convert MobileNetV2 to TFLite FP32 with fixed shapes"""
    print("\n" + "="*60)
    print("Converting MobileNetV2 to TFLite FP32")
    print("="*60)
    
    # Load MobileNetV2
    model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=True,
        input_shape=(224, 224, 3)
    )
    print("✅ Loaded MobileNetV2 from Keras")
    
    # Create concrete function with fixed shape
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
    def model_fn(x):
        return model(x, training=False)
    
    concrete_func = model_fn.get_concrete_function()
    
    # Convert to TFLite FP32
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = []
    tflite_model = converter.convert()
    
    output_path = "./models/mobilenetv2_fp32.tflite"
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Saved: {output_path} ({len(tflite_model) / 1024 / 1024:.2f} MB)")
    
    # Verify
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"   Input: {input_details['shape']}, dtype: {input_details['dtype'].__name__}")
    print(f"   Output: {output_details['shape']}, dtype: {output_details['dtype'].__name__}")
    
    return output_path


def convert_mobilenetv2_onnx():
    """Convert MobileNetV2 to ONNX FP32"""
    print("\n" + "="*60)
    print("Converting MobileNetV2 to ONNX FP32")
    print("="*60)
    
    # Load PyTorch MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    print("✅ Loaded MobileNetV2 from torchvision")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = "./models/mobilenetv2_fp32.onnx"
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ Saved: {output_path} ({file_size:.2f} MB)")
    
    # Simplify
    print("   Simplifying ONNX model...")
    onnx_model = onnx.load(output_path)
    model_simplified, check = simplify(onnx_model)
    if check:
        onnx.save(model_simplified, output_path)
        print(f"   ✅ Simplified and saved")
    else:
        print(f"   ⚠️  Simplification failed, using original")
    
    return output_path


if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    
    print("🚀 MobileNetV2 Model Conversion")
    print("   (Better Core ML compatibility than V3)")
    
    tflite_path = convert_mobilenetv2_tflite()
    onnx_path = convert_mobilenetv2_onnx()
    
    print("\n" + "="*60)
    print("🎉 MobileNetV2 conversion complete!")
    print("="*60)
    print(f"\nGenerated models:")
    print(f"  - TFLite: {tflite_path}")
    print(f"  - ONNX:   {onnx_path}")
    print(f"\nNext: Copy these to your Xcode projects and test Core ML!")

