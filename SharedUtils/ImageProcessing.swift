// ImageProcessing.swift
// Shared preprocessing and postprocessing for vision models

import UIKit
import CoreGraphics
import Accelerate

struct ImageProcessor {
    
    // MARK: - Preprocessing
    
    /// Resize and normalize image to model input format
    /// - Parameters:
    ///   - image: Input UIImage
    ///   - targetSize: Target dimensions (e.g., 224x224)
    ///   - normalize: Normalization method
    /// - Returns: Flattened Float array in RGB order
    static func preprocess(
        _ image: UIImage,
        targetSize: CGSize = CGSize(width: 224, height: 224),
        normalize: NormalizationMode = .zeroToOne
    ) -> [Float]? {
        
        guard let resized = image.resized(to: targetSize),
              let cgImage = resized.cgImage else {
            return nil
        }
        
        let width = Int(targetSize.width)
        let height = Int(targetSize.height)
        let pixelCount = width * height
        
        // Extract RGB pixels
        guard let pixels = extractRGBPixels(from: cgImage, width: width, height: height) else {
            return nil
        }
        
        // Normalize based on mode
        var normalized = [Float](repeating: 0, count: pixelCount * 3)
        
        switch normalize {
        case .zeroToOne:
            // [0, 255] -> [0, 1]
            for i in 0..<normalized.count {
                normalized[i] = Float(pixels[i]) / 255.0
            }
            
        case .minusOneToOne:
            // [0, 255] -> [-1, 1]
            for i in 0..<normalized.count {
                normalized[i] = (Float(pixels[i]) / 127.5) - 1.0
            }
            
        case .imagenet:
            // ImageNet normalization: (x/255 - mean) / std
            let means: [Float] = [0.485, 0.456, 0.406]
            let stds: [Float] = [0.229, 0.224, 0.225]
            
            for i in 0..<pixelCount {
                for c in 0..<3 {
                    let idx = i * 3 + c
                    let normalized_value = Float(pixels[idx]) / 255.0
                    normalized[idx] = (normalized_value - means[c]) / stds[c]
                }
            }
        }
        
        return normalized
    }
    
    /// Generate random tensor for synthetic benchmarking
    static func generateRandomTensor(
        width: Int = 224,
        height: Int = 224,
        channels: Int = 3
    ) -> [Float] {
        let count = width * height * channels
        return (0..<count).map { _ in Float.random(in: 0...1) }
    }
    
    // MARK: - Postprocessing
    
    /// Extract top-k predictions from softmax output
    static func postprocess(
        _ output: [Float],
        topK: Int = 5
    ) -> [(index: Int, confidence: Float)] {
        
        // Apply softmax
        let softmaxed = softmax(output)
        
        // Get top-k indices
        let indexed = softmaxed.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }
        
        return Array(sorted.prefix(topK))
    }
    
    static func softmax(_ input: [Float]) -> [Float] {
        let maxVal = input.max() ?? 0
        let exps = input.map { exp($0 - maxVal) }
        let sumExps = exps.reduce(0, +)
        return exps.map { $0 / sumExps }
    }
    
    // MARK: - Private Helpers
    
    private static func extractRGBPixels(
        from cgImage: CGImage,
        width: Int,
        height: Int
    ) -> [UInt8]? {
        
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        
        guard let ctx = context else { return nil }
        
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert RGBA to RGB
        var rgbData = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgbData[i * 3 + 0] = pixelData[i * 4 + 0] // R
            rgbData[i * 3 + 1] = pixelData[i * 4 + 1] // G
            rgbData[i * 3 + 2] = pixelData[i * 4 + 2] // B
        }
        
        return rgbData
    }
}

// MARK: - Normalization Modes

enum NormalizationMode {
    case zeroToOne      // [0, 1]
    case minusOneToOne  // [-1, 1]
    case imagenet       // ImageNet statistics
}

// MARK: - UIImage Extension

extension UIImage {
    func resized(to targetSize: CGSize) -> UIImage? {
        let size = self.size
        
        let widthRatio  = targetSize.width  / size.width
        let heightRatio = targetSize.height / size.height
        
        let ratio = min(widthRatio, heightRatio)
        let newSize = CGSize(width: size.width * ratio, height: size.height * ratio)
        
        let rect = CGRect(
            x: (targetSize.width - newSize.width) / 2,
            y: (targetSize.height - newSize.height) / 2,
            width: newSize.width,
            height: newSize.height
        )
        
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        self.draw(in: rect)
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}

// MARK: - Memory Utilities

struct MemoryMonitor {
    static func getMemoryUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if result == KERN_SUCCESS {
            return Double(info.resident_size) / 1024.0 / 1024.0 // MB
        }
        return 0
    }
}

