# ONNX Runtime iOS App - Xcode Setup Guide

## ✅ Files Ready

All files have been copied and created:
- ✅ ONNX models: `mobilenetv2_fp32.onnx`, `mobilenetv3_small_fp32.onnx`
- ✅ `ORTRunner.swift` (Core ML EP support)
- ✅ `ContentView.swift` (UI with model selector)
- ✅ `ORTCoreMLDemoApp.swift` (app entry point)
- ✅ `SharedUtils/` (BenchmarkResult, Logger, ImageProcessing)
- ✅ Bridging Header

## 📱 Xcode Setup Steps

### 1. Open the Workspace
```bash
cd /Users/omkarpodey/ai_exp/experiment_cs259/ios_tflite_ort_comparison/ORTCoreMLDemo
open ORTCoreMLDemo.xcworkspace
```
**Important**: Open `.xcworkspace`, NOT `.xcodeproj`!

### 2. Add Files to Xcode Project

In Xcode Navigator (left panel):

#### a) Add ONNX Models
1. Right-click on `ORTCoreMLDemo` folder
2. "Add Files to 'ORTCoreMLDemo'..."
3. Navigate to `ORTCoreMLDemo/ORTCoreMLDemo/ORTCoreMLDemo/`
4. Select both `.onnx` files:
   - `mobilenetv2_fp32.onnx`
   - `mobilenetv3_small_fp32.onnx`
5. ✅ Check "Copy items if needed"
6. ✅ Check "Add to targets: ORTCoreMLDemo"
7. Click "Add"

#### b) Add Swift Files
Repeat the same process for these files (if not already in Xcode):
- `ORTRunner.swift`
- `ContentView.swift`
- `ORTCoreMLDemoApp.swift`
- `SharedUtils/` folder (entire folder with all 3 files)

#### c) Configure Bridging Header
1. Select project (blue icon) → `ORTCoreMLDemo` target
2. Build Settings → search "bridging"
3. Set **Objective-C Bridging Header** to:
   ```
   ORTCoreMLDemo/ORTCoreMLDemo-Bridging-Header.h
   ```

### 3. Verify Target Membership

For each file you added:
1. Select file in Navigator
2. File Inspector (right panel) → **Target Membership**
3. Ensure `ORTCoreMLDemo` is **checked**

### 4. Build Settings

1. Select project → `ORTCoreMLDemo` target
2. **General** tab:
   - Bundle Identifier: Make it unique (e.g., `com.yourname.ORTCoreMLDemo`)
   - Deployment Target: iOS 16.0
3. **Signing & Capabilities**:
   - ✅ Automatically manage signing
   - Select your Team (add Apple ID in Xcode → Settings → Accounts if needed)

### 5. Build & Run

1. **Clean Build Folder**: Product → Clean Build Folder (Shift+⌘K)
2. **Select Device**: Choose your iPhone or a simulator
3. **Build**: Product → Build (⌘B)
4. **Run**: Product → Run (⌘R)

## 🚀 Testing the App

### Test 1: MobileNetV2 + Core ML
- Model: **MobileNetV2**
- Core ML EP: **ON**
- Runs: 50, Warmup: 10
- Tap "Run Benchmark"

**Expected**: Fast inference with Core ML (~1-2 ms)

### Test 2: MobileNetV3 + Core ML
- Model: **MobileNetV3**
- Core ML EP: **ON**
- Tap "Run Benchmark"

**Expected**: Should also work with Core ML (ONNX handles MobileNetV3 better than TFLite)

## 📊 Compare with TFLite Results

| Framework | Model | Backend | Mean Latency |
|-----------|-------|---------|--------------|
| TFLite | MobileNetV2 | Core ML | ~1.0 ms |
| ONNX Runtime | MobileNetV2 | Core ML | ? |
| ONNX Runtime | MobileNetV3 | Core ML | ? |

## 🔧 Troubleshooting

### "No such module 'onnxruntime_objc'"
- Make sure you opened `.xcworkspace` (not `.xcodeproj`)
- Clean Build Folder and rebuild

### Bridging Header Not Found
- Check the path in Build Settings → Objective-C Bridging Header
- Should be: `ORTCoreMLDemo/ORTCoreMLDemo-Bridging-Header.h`

### Model Not Found
- Check Target Membership for `.onnx` files
- Ensure files are in the correct nested directory

### Core ML EP Not Working
- Check console logs for errors
- Fallback to CPU is automatic if Core ML fails

## ✅ Success Indicators

Console should show:
```
✅ ORT: Environment created
✅ ORT: Core ML Execution Provider enabled
✅ ORT: Session created successfully
🏃 Starting ORT benchmark: 110 runs (10 warmup)
```

If you see all ✅, Core ML EP is working!

