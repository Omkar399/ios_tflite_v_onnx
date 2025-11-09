#!/bin/bash
# Quick Start Script for TFLite vs ORT iOS Benchmark
# This script automates the initial setup steps

set -e  # Exit on error

echo "🚀 TFLite vs ORT iOS Benchmark - Quick Start"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "   Install from: https://www.python.org/downloads/"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found: $(python3 --version)${NC}"
echo ""

# Check CocoaPods
echo -e "${BLUE}Checking CocoaPods installation...${NC}"
if ! command -v pod &> /dev/null; then
    echo -e "${YELLOW}⚠️  CocoaPods not found. Installing...${NC}"
    sudo gem install cocoapods
fi
echo -e "${GREEN}✅ CocoaPods found: $(pod --version)${NC}"
echo ""

# Step 1: Convert Models
echo -e "${BLUE}Step 1: Converting Models${NC}"
echo "----------------------------------------"
cd ModelConversion

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "Converting MobileNetV3-Small to TFLite and ONNX..."
python convert_models.py --model-size small --output-dir models --verify

echo -e "${GREEN}✅ Models converted successfully!${NC}"
echo ""

# Step 2: Set up TFLite App
echo -e "${BLUE}Step 2: Setting up TFLite iOS App${NC}"
echo "----------------------------------------"
cd ../TFLiteCoreMLDemo

echo "Installing CocoaPods dependencies for TFLite app..."
pod install

echo -e "${GREEN}✅ TFLite app dependencies installed!${NC}"
echo ""

# Step 3: Set up ORT App
echo -e "${BLUE}Step 3: Setting up ORT iOS App${NC}"
echo "----------------------------------------"
cd ../ORTCoreMLDemo

echo "Installing CocoaPods dependencies for ORT app..."
pod install

echo -e "${GREEN}✅ ORT app dependencies installed!${NC}"
echo ""

cd ..

# Summary
echo ""
echo "=============================================="
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "📁 Models location: ModelConversion/models/"
echo ""
echo "Next steps:"
echo ""
echo "1. Open TFLite app in Xcode:"
echo "   $ cd TFLiteCoreMLDemo"
echo "   $ open TFLiteCoreMLDemo.xcworkspace"
echo ""
echo "   Then in Xcode:"
echo "   - Add .tflite model files to project"
echo "   - Add SharedUtils/*.swift files"
echo "   - Configure signing (select your team)"
echo "   - Build and run on device"
echo ""
echo "2. Open ORT app in Xcode:"
echo "   $ cd ORTCoreMLDemo"
echo "   $ open ORTCoreMLDemo.xcworkspace"
echo ""
echo "   Then in Xcode:"
echo "   - Add .onnx model files to project"
echo "   - Add SharedUtils/*.swift files"
echo "   - Configure signing"
echo "   - Build and run on device"
echo ""
echo "3. Run benchmarks on your iPhone (see SETUP_GUIDE.md)"
echo ""
echo "4. Analyze results:"
echo "   $ cd ModelConversion"
echo "   $ source venv/bin/activate"
echo "   $ python analyze_results.py --tflite-csv path/to/tflite.csv --ort-csv path/to/ort.csv"
echo ""
echo "📚 For detailed instructions, see:"
echo "   - SETUP_GUIDE.md (step-by-step setup)"
echo "   - README.md (quick reference)"
echo "   - PROJECT_SUMMARY.md (complete overview)"
echo ""
echo -e "${GREEN}Happy benchmarking! 🚀${NC}"

