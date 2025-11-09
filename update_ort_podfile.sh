#!/bin/bash
# This will update the Podfile after you create the Xcode project

cd ORTCoreMLDemo

# Check if project exists
if [ ! -d "ORTCoreMLDemo/ORTCoreMLDemo.xcodeproj" ]; then
    echo "❌ Please create the Xcode project first!"
    echo "   1. Open Xcode"
    echo "   2. File → New → Project → iOS → App"
    echo "   3. Name: ORTCoreMLDemo"
    echo "   4. Save to this folder"
    exit 1
fi

# Update Podfile to point to the project
cat > Podfile << 'PODFILE'
# Podfile for ONNX Runtime Mobile + Core ML Demo
platform :ios, '16.0'
use_frameworks!

# Specify the Xcode project location
project 'ORTCoreMLDemo/ORTCoreMLDemo.xcodeproj'

target 'ORTCoreMLDemo' do
  # Use Objective-C wrapper - much easier than C API
  pod 'onnxruntime-objc'
end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '16.0'
    end
  end
end
PODFILE

echo "✅ Podfile updated! Running pod install..."
pod install --repo-update

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! ORT app ready!"
    echo ""
    echo "Open: ORTCoreMLDemo.xcworkspace"
fi
