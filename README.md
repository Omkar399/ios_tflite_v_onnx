# iOS Benchmark: TensorFlow Lite vs ONNX Runtime

> **Comprehensive performance comparison of TensorFlow Lite and ONNX Runtime on iOS with hardware acceleration (Neural Engine + Core ML)**

[![iOS 16+](https://img.shields.io/badge/iOS-16%2B-blue.svg)](https://developer.apple.com/ios/)
[![Swift 5.5+](https://img.shields.io/badge/Swift-5.5%2B-orange.svg)](https://swift.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 📊 Quick Results

Both frameworks running **Mobile-ViT** (256×256 RGB input) with **Core ML acceleration**:

| Framework | Backend | Mean Latency | Throughput | Winner |
|-----------|---------|--------------|-----------|--------|
| **ONNX Runtime** | Core ML + Neural Engine | **4.65 ms** ⚡ | **215 inf/s** | ✅ **Supported** |
| **TensorFlow Lite** | Core ML Delegate | ❌ | ❌ | **Incompatible** |

**Comparison with MobileNetV2 (224×224):**

| Backend | TensorFlow Lite | ONNX Runtime | Winner |
|---------|-----------------|--------------|--------|
| **Core ML (Neural Engine)** | 1.089 ms | **0.635 ms** | ONNX (47% faster) |
| **CPU Only** | **4.279 ms** | 8.489 ms | TFLite (98% faster) |

## 🎯 Project Goal

Conduct a rigorous, reproducible benchmark comparing two major ML inference frameworks on iOS:
- **Same models** across frameworks
- **Same hardware acceleration** (Apple Neural Engine via Core ML)
- **Statistical rigor** (p50, p90, p95, p99 latencies)
- **Real-world conditions** (Release builds, physical devices)

## 🏗️ Project Structure

```
ios_tflite_ort_comparison/
├── 📱 TFLiteCoreMLDemo/              # TensorFlow Lite benchmark app
│   ├── Podfile                       # Dependencies
│   ├── ContentView.swift             # SwiftUI UI
│   ├── TFLiteRunner.swift            # Inference engine
│   └── mobilevit_float.tflite        # Mobile-ViT model
│
├── 📱 ORTCoreMLDemo/                 # ONNX Runtime benchmark app
│   ├── Podfile                       # Dependencies
│   ├── ContentView.swift             # SwiftUI UI
│   ├── ORTRunner.swift               # Inference engine
│   └── mobilevit_combined.onnx       # Mobile-ViT model
│
├── 🛠️ SharedUtils/                   # Shared code
│   ├── ImageProcessing.swift         # Preprocessing, synthetic data
│   ├── BenchmarkResult.swift         # Statistical calculations
│   └── Logger.swift                  # CSV export
│
├── 🐍 ModelConversion/               # Python utilities
│   ├── convert_models.py             # TFLite/ONNX conversion
│   ├── analyze_results.py            # Results analysis & plots
│   └── requirements.txt              # Dependencies
│
├── 📖 claude.md                      # Comprehensive reference (642 lines)
└── README.md                         # This file
```

## 🚀 Quick Start (10 minutes)

### Prerequisites
- **macOS** with Xcode 15+
- **iPhone** running iOS 16+ (physical device required)
- **CocoaPods**: `sudo gem install cocoapods`

### Setup TensorFlow Lite App

```bash
cd TFLiteCoreMLDemo
pod install
open TFLiteCoreMLDemo.xcworkspace
```

**In Xcode:**
1. Drag `mobilevit_float.tflite` to project
2. Add `SharedUtils/*.swift` files
3. Set iOS Deployment Target: **16.0+**
4. Product → Build (⌘B) → Run (⌘R)

### Setup ONNX Runtime App

```bash
cd ../ORTCoreMLDemo
pod install
open ORTCoreMLDemo.xcworkspace
```

**In Xcode:**
1. Drag `mobilevit_combined.onnx` to project
2. Add `SharedUtils/*.swift` files
3. Set iOS Deployment Target: **16.0+**
4. Product → Build → Run

### Run Benchmarks

1. **Prepare device**: Charge, Airplane Mode ON, cool for 5 min
2. **Select model**: "Mobile-ViT" from picker
3. **Enable acceleration**: Toggle "Use Core ML"
4. **Tap**: "Run Benchmark"
5. **Export**: "Export CSV" for analysis

## 🔬 Key Features

### ✅ Models Supported
- **MobileNetV2** (224×224) - CNN baseline
- **MobileNetV3-Small** (224×224) - Optimized CNN
- **Mobile-ViT** (256×256) - **Vision Transformer** (NEW!)

### ✅ Backends Tested
| Framework | CPU | Core ML | Precision |
|-----------|-----|---------|-----------|
| TensorFlow Lite | ✅ | ✅ (with limitations) | FP32, FP16, INT8 |
| ONNX Runtime | ✅ | ✅ (full support) | FP32, FP16 |

### ✅ Metrics Collected
- **Latency**: p50, p90, p95, p99 percentiles
- **Throughput**: Inferences per second
- **Stability**: Standard deviation
- **Memory**: Peak resident memory
- **Energy**: Via Xcode Instruments

### ✅ Statistical Rigor
- 100+ measured runs per test (after 10 warmup)
- High-precision timing (microsecond accuracy)
- Synthetic data (reproducibility)
- CSV export for external analysis

## 📱 Mobile-ViT Integration

**What is Mobile-ViT?**  
A hybrid CNN+Transformer architecture designed for mobile devices. It combines:
- Local CNN layers (efficient feature extraction)
- Global attention blocks (long-range context)
- ~5-6M parameters

**Testing Results:**

✅ **ONNX Runtime + Core ML EP**: **4.65 ms** per inference (Neural Engine accelerated!)

❌ **TensorFlow Lite + Core ML Delegate**: Incompatible with transformer ops
- Error: "Mean op is only supported for 4D input"
- Workaround: CPU-only mode (not recommended)

**Insight**: ONNX Runtime's Core ML integration is more robust for modern architectures.

## 🔧 Technical Highlights

### Dynamic Input Shape Detection

Both apps automatically detect model input dimensions:

```swift
// Automatically works for 224×224 or 256×256 inputs
private func queryInputMetadata() throws {
    if let tensorInfo = try? session.inputTypeInfo(at: 0) as? ORTTensorTypeAndShapeInfo {
        inputShape = tensorInfo.shape  // [1, 3, 256, 256] for Mobile-ViT
    }
}
```

### Synthetic Test Data

Pure inference speed testing with random RGB "images":

```swift
let randomTensor = ImageProcessor.generateRandomTensor(
    width: 256,     // Or 224 for MobileNetV3
    height: 256,
    channels: 3
)
// Result: 196,608 random floats (no I/O overhead)
```

### ML Program Format (ONNX Runtime Fix)

Critical fix for external data loading with Core ML:

```swift
let coreMLOptions: [String: String] = [
    "ModelFormat": "MLProgram",        // Modern Core ML format
    "MLComputeUnits": "ALL"            // Neural Engine + GPU + CPU
]
try options.appendCoreMLExecutionProvider(withOptions: coreMLOptions)
```

**Result**: Mobile-ViT external weights load correctly! ✅

## 📊 Performance Analysis

### Core ML Acceleration Impact

**ONNX Runtime gains 13.4× speedup** when using Core ML:
- CPU only: 8.489 ms
- Core ML: 0.635 ms
- **Speedup: 13.4×** ⚡

**TensorFlow Lite gains only 3.9× speedup**:
- CPU only: 4.279 ms
- Core ML: 1.089 ms
- **Speedup: 3.9×**

**Interpretation**: ONNX Runtime benefits more from hardware acceleration. Better Core ML integration → higher speedup.

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **CocoaPods errors** | `pod repo update && pod install` |
| **Model not found** | Check File Inspector → Target Membership |
| **Build fails** | Set iOS Deployment Target to 16.0+ |
| **Core ML not working** | Requires iOS 12+, A11 chip+. Check console logs. |
| **TFLite crashes on Mobile-ViT** | Use CPU-only (Core ML Delegate unsupported for transformers) |

See **`claude.md`** for detailed troubleshooting.

## 📈 Analyze Results

### Python Analysis Script

```python
import pandas as pd
df = pd.read_csv('ort_results.csv')

print(f"Mean: {df['latency_ms'].mean():.3f} ms")
print(f"P50:  {df['latency_ms'].quantile(0.50):.3f} ms")
print(f"P90:  {df['latency_ms'].quantile(0.90):.3f} ms")
```

### Generate Plots

```bash
cd ModelConversion
pip install -r analysis_requirements.txt
python analyze_results.py \
    --tflite-csv path/to/tflite.csv \
    --ort-csv path/to/ort.csv \
    --output-dir results/
```

Generates:
- Latency distribution histograms
- Box plots comparison
- Time series analysis
- Summary statistics

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **README.md** | This file - quick overview |
| **claude.md** | Complete 642-line reference guide |
| `ModelConversion/README.md` | Model conversion details |

👉 **Start here**: Check `claude.md` for exhaustive documentation including:
- Complete setup guide
- All performance results
- Technical architecture
- Research insights
- Complete API reference

## 🎓 Research Findings

### Key Insight: Backend-Dependent Performance

**The "better" framework is not absolute—it depends on your target:**

✅ **Use ONNX Runtime if:**
- Targeting modern iOS devices with Neural Engine
- Models fully compatible with Core ML
- Latency-critical applications (real-time video, AR/VR)
- Want maximum hardware acceleration

✅ **Use TensorFlow Lite if:**
- Supporting older devices without Neural Engine
- Models have Core ML-unsupported ops
- Need robust CPU fallback performance
- Require broad platform support (Android, embedded)

## 📱 Device Requirements

| Requirement | Details |
|-------------|---------|
| **iOS Version** | 16.0+ (tested 16, 17, 18) |
| **Device** | iPhone with A12 Bionic or newer |
| **Recommended** | iPhone 12+ for best Core ML performance |
| **Neural Engine** | Required for Core ML acceleration |

## 📄 License

MIT License - See repository for full details.

## 🤝 Contributing

This is a research benchmark project. Contributions welcome for:
- Additional models (EfficientNet, VGG, etc.)
- Additional frameworks (PyTorch Mobile, TensorRT)
- Analysis improvements
- Documentation enhancements

## 📞 Support

**Questions or issues?**
1. Check the troubleshooting section in `claude.md`
2. Review console logs for detailed error messages
3. Verify device is iOS 16+ and connected
4. Try Release build (Debug builds are slower)

## 🎉 Quick Links

- 📖 [Full Documentation](claude.md)
- 🐍 [Model Conversion Guide](ModelConversion/README.md)
- 🚀 [Quickstart Script](QUICKSTART.sh)

---

**Status**: ✅ Production-ready  
**Last Updated**: November 2025  
**Experiment**: iOS TFLite vs ONNX Runtime Mobile Benchmark

**Ready to benchmark?** Follow the [Quick Start](#-quick-start-10-minutes) above! 🚀

