# Model Conversion for TFLite vs ORT Benchmark

This directory contains scripts to convert MobileNetV3 models to both TFLite and ONNX formats for benchmarking on iOS.

## ✅ Successfully Converted Models

All models have been converted and are ready to use:

- **TFLite Models:**
  - `mobilenetv3_small_fp32.tflite` (9.7 MB)
  - `mobilenetv3_small_fp16.tflite` (4.9 MB)
  - `mobilenetv3_small_int8.tflite` (2.7 MB)

- **ONNX Models:**
  - `mobilenetv3_small_fp32.onnx` (9.7 MB)
  - `mobilenetv3_small_fp16.onnx` (4.9 MB)

## 🛠️ Setup Instructions

### 1. Create Virtual Environment

**Important:** Use Python 3.11 (not 3.13) for compatibility:

```bash
cd ModelConversion

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Version Notes

The `requirements.txt` file pins specific versions that work on macOS ARM (Apple Silicon):

- **PyTorch 2.1.0**: Later versions (2.9.0+) have ONNX export crashes on macOS ARM
- **NumPy <2.0**: PyTorch 2.1.0 requires NumPy 1.x

## 📦 Conversion Scripts

### Independent Scripts (Recommended)

Two clean, independent scripts for conversion:

#### 1. TFLite Conversion

```bash
python convert_to_tflite.py --model-size small --output-dir models
```

**Features:**
- Converts to FP32, FP16, and INT8 formats
- Uses TensorFlow/Keras pre-trained models
- Includes verification step

#### 2. ONNX Conversion  

```bash
python convert_to_onnx.py --model-size small --output-dir models
```

**Features:**
- Converts to FP32 and FP16 formats
- Uses PyTorch/torchvision pre-trained models
- Includes simplification and verification

### Options

Both scripts support:

```bash
--model-size {small,large}  # Choose model size (default: small)
--output-dir PATH           # Output directory (default: ./models)
```

## ⚠️ Known Issues

### PyTorch ONNX Export Crashes

On macOS ARM with Python 3.11+, PyTorch 2.9.0 ONNX export causes segmentation faults. This is resolved by:

1. Using PyTorch 2.1.0
2. Using NumPy 1.x
3. Using ONNX opset 17 (maximum supported by PyTorch 2.1.0)

### Workaround for FP16 ONNX

If the ONNX script crashes during FP16 conversion, you can manually convert:

```bash
python -c "
import onnx
from onnxconverter_common import float16

model = onnx.load('models/mobilenetv3_small_fp32.onnx')
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, 'models/mobilenetv3_small_fp16.onnx')
print('✅ FP16 conversion complete!')
"
```

## 🔍 Model Verification

### Verify TFLite Models

```bash
python -c "
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='models/mobilenetv3_small_fp32.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f'Input shape: {input_details[0][\"shape\"]}')
print(f'Output shape: {output_details[0][\"shape\"]}')
"
```

### Verify ONNX Models

```bash
python -c "
import onnxruntime as ort

session = ort.InferenceSession('models/mobilenetv3_small_fp32.onnx', 
                                providers=['CPUExecutionProvider'])

print(f'Input: {session.get_inputs()[0].name}, shape: {session.get_inputs()[0].shape}')
print(f'Output: {session.get_outputs()[0].name}, shape: {session.get_outputs()[0].shape}')
"
```

## 📱 Next Steps

1. **Copy models to iOS projects:**
   - Copy `.tflite` files to the TFLite Xcode project
   - Copy `.onnx` files to the ORT Xcode project

2. **Verify Target Membership:**
   - In Xcode, select each model file
   - Ensure it's checked under "Target Membership" in the File Inspector
   - This ensures the files are included in the app bundle

3. **Update model paths in code:**
   - Update the model filenames in your iOS app code if needed

## 📊 Model Specifications

### Input

- **Shape:** `(1, 224, 224, 3)` for TFLite (NHWC format)
- **Shape:** `(1, 3, 224, 224)` for ONNX (NCHW format)
- **Type:** Float32 (or quantized types for INT8)
- **Range:** [0, 1] (normalized RGB values)

### Output

- **Shape:** `(1, 1000)`
- **Type:** Float32
- **Content:** ImageNet class probabilities

## 🔧 Troubleshooting

### Issue: ImportError for onnxconverter-common

**Solution:**
```bash
pip install onnxconverter-common
```

### Issue: ImportError for onnxscript

**Solution:**
```bash
pip install onnxscript
```

### Issue: Segmentation fault during ONNX export

**Solution:**
1. Downgrade to PyTorch 2.1.0 and NumPy <2.0 (as specified in requirements.txt)
2. Use the manual FP16 conversion workaround above
3. The FP32 model is usually created before the crash - check the `models/` directory

### Issue: NumPy version conflict

**Solution:**
```bash
pip install 'numpy<2.0'
```

## 📚 References

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)

