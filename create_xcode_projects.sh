#!/bin/bash
# Script to create Xcode projects for both apps

set -e

echo "🔨 Creating Xcode Projects..."
echo ""

# Check if Xcode command line tools are available
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ Xcode command line tools not found."
    echo "   Run: xcode-select --install"
    exit 1
fi

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to create a basic Xcode project
create_project() {
    local PROJECT_NAME=$1
    local PROJECT_DIR=$2
    
    echo -e "${BLUE}Creating ${PROJECT_NAME}...${NC}"
    
    cd "$PROJECT_DIR"
    
    # Create a temporary Swift package to bootstrap the project
    # This is a workaround since we can't easily create .xcodeproj from command line
    
    echo "⚠️  Manual step required for ${PROJECT_NAME}:"
    echo ""
    echo "1. Open Xcode"
    echo "2. File → New → Project"
    echo "3. Choose: iOS → App"
    echo "4. Settings:"
    echo "   - Product Name: ${PROJECT_NAME}"
    echo "   - Organization Identifier: com.yourname"
    echo "   - Interface: SwiftUI"
    echo "   - Language: Swift"
    echo "5. Save to: ${PROJECT_DIR}"
    echo "6. After creating, replace the default files with our files:"
    echo "   - Delete ContentView.swift and ${PROJECT_NAME}App.swift"
    echo "   - Add our versions from the folder"
    echo ""
}

echo "════════════════════════════════════════════════════════════"
echo "📱 Xcode Project Creation Required"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "CocoaPods requires actual Xcode projects (.xcodeproj files)."
echo "Unfortunately, these can't be easily created from the command line."
echo ""
echo "Please follow these steps:"
echo ""

create_project "TFLiteCoreMLDemo" "$(pwd)/TFLiteCoreMLDemo"
echo "---"
create_project "ORTCoreMLDemo" "$(pwd)/ORTCoreMLDemo"

echo "════════════════════════════════════════════════════════════"
echo ""
echo "After creating both projects in Xcode:"
echo ""
echo "1. Close Xcode"
echo "2. Run: cd TFLiteCoreMLDemo && pod install"
echo "3. Run: cd ../ORTCoreMLDemo && pod install"
echo "4. Open the .xcworkspace files (not .xcodeproj)"
echo ""
echo "════════════════════════════════════════════════════════════"

