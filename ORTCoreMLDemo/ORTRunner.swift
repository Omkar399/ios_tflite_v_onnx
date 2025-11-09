// ORTRunner.swift
// ONNX Runtime Mobile inference runner with Core ML EP support
// Using onnxruntime-objc wrapper (much simpler than C API)

import Foundation
import UIKit
import Accelerate

enum ORTError: Error {
    case modelLoadFailed
    case sessionInitFailed
    case inferenceFailed
    case invalidInput
    case invalidOutput
}

final class ORTRunner {
    
    private var env: ORTEnv
    private var session: ORTSession
    
    let modelName: String
    let useCoreML: Bool
    let precision: String
    
    private var inputName: String = ""
    private var inputShape: [NSNumber] = []
    private var inputWidth: Int = 224
    private var inputHeight: Int = 224
    
    // MARK: - Initialization
    
    init(
        modelName: String,
        useCoreML: Bool = true,
        precision: String = "fp16"
    ) throws {
        
        self.modelName = modelName
        self.useCoreML = useCoreML
        self.precision = precision
        
        // Locate model file
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "onnx") else {
            print("❌ ORT: Model file '\(modelName).onnx' not found in bundle")
            throw ORTError.modelLoadFailed
        }
        
        print("📍 ORT: Model path: \(modelURL.path)")
        print("📍 ORT: Model directory: \(modelURL.deletingLastPathComponent().path)")
        
        // For models with external data, verify the .data file exists
        if modelName == "model" {
            print("🔍 ORT: Checking for external data file...")
            let dataFilename = "\(modelName).data"
            if let dataURL = Bundle.main.url(forResource: modelName, withExtension: "data") {
                print("✅ ORT: Found external data file: \(dataURL.path)")
                print("   Data file size: \(try? FileManager.default.attributesOfItem(atPath: dataURL.path)[.size] ?? "unknown")")
            } else {
                print("❌ ORT: External data file '\(dataFilename)' not found in bundle")
                let allDataFiles = Bundle.main.paths(forResourcesOfType: "data", inDirectory: nil)
                print("   Available .data files: \(allDataFiles)")
                // Don't throw here - let ONNX Runtime try and give a better error
            }
        }
        
        // Create ORT environment
        do {
            self.env = try ORTEnv(loggingLevel: .warning)
            print("✅ ORT: Environment created")
        } catch {
            print("❌ ORT: Failed to create environment: \(error)")
            throw ORTError.sessionInitFailed
        }
        
        // Create session options
        let options: ORTSessionOptions
        do {
            options = try ORTSessionOptions()
            try options.setLogSeverityLevel(.warning)
            try options.setIntraOpNumThreads(4)
            
            // Enable Core ML Execution Provider if requested
            if useCoreML {
                // Core ML EP configuration with ML Program format (fixes external data bug!)
                // iOS 15+ / macOS 12+ required for ML Program support
                // Note: All values must be Strings (not Bool or NSNumber) for ONNX Runtime ObjC bridge
                let coreMLOptions: [String: String] = [
                    "ModelFormat": "MLProgram",           // Use modern ML Program format (fixes external data!)
                    "MLComputeUnits": "ALL"               // Use Neural Engine + GPU + CPU (correct key!)
                ]
                
                do {
                    try options.appendCoreMLExecutionProvider(withOptions: coreMLOptions)
                    print("✅ ORT: Core ML EP enabled with ML Program format")
                    print("   ModelFormat: MLProgram (modern, fixes external data)")
                    print("   MLComputeUnits: ALL (Neural Engine + GPU + CPU)")
                } catch {
                    print("⚠️ ORT: Failed to enable Core ML EP with ML Program: \(error)")
                    print("   Trying without options...")
                    do {
                        try options.appendCoreMLExecutionProvider()
                        print("✅ ORT: Core ML EP enabled (fallback mode)")
                    } catch {
                        print("❌ ORT: Core ML EP completely failed, using CPU only")
                    }
                }
            } else {
                print("✅ ORT: Using CPU Execution Provider only")
            }
            
        } catch {
            print("❌ ORT: Failed to create session options: \(error)")
            throw ORTError.sessionInitFailed
        }
        
        // Create session
        do {
            self.session = try ORTSession(
                env: env,
                modelPath: modelURL.path,
                sessionOptions: options
            )
            print("✅ ORT: Session created successfully")
        } catch {
            print("❌ ORT: Failed to create session: \(error)")
            throw ORTError.sessionInitFailed
        }
        
        // Query input metadata
        do {
            try queryInputMetadata()
        } catch {
            print("⚠️ ORT: Failed to query metadata: \(error)")
            // Use defaults
            inputName = "input"
            inputShape = [1, 3, 224, 224]
        }
    }
    
    // MARK: - Metadata Query
    
    private func queryInputMetadata() throws {
        // Get input name
        let inputNames = try session.inputNames()
        if let firstName = inputNames.first {
            inputName = firstName
            print("📊 ORT: Input name: \(inputName)")
        }
        
        // Try to get input shape info (API may not be available in all versions)
        // For now, use default shape based on model
        inputShape = [1, 3, 256, 256]  // Mobile-ViT default
        inputHeight = 256
        inputWidth = 256
        print("📊 ORT: Using default input shape: \(inputShape)")
    }
    
    // MARK: - Inference
    
    func run(input: [Float]) throws -> (output: [Float], latencyMs: Double) {
        
        // Validate input size
        let expectedSize = inputShape.reduce(1) { $0 * $1.intValue }
        guard input.count == expectedSize else {
            print("❌ ORT: Input size mismatch. Expected: \(expectedSize), Got: \(input.count)")
            throw ORTError.invalidInput
        }
        
        // Create input tensor
        let inputTensor: ORTValue
        do {
            let inputData = Data(bytes: input, count: input.count * MemoryLayout<Float>.size)
            inputTensor = try ORTValue(
                tensorData: NSMutableData(data: inputData),
                elementType: .float,
                shape: inputShape
            )
        } catch {
            print("❌ ORT: Failed to create input tensor: \(error)")
            throw ORTError.invalidInput
        }
        
        // Get output names
        let outputNames = try session.outputNames()
        guard let outputName = outputNames.first else {
            throw ORTError.invalidOutput
        }
        
        // Run inference with timing
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let outputs: [String: ORTValue]
        do {
            outputs = try session.run(
                withInputs: [inputName: inputTensor],
                outputNames: Set(outputNames),
                runOptions: nil
            )
        } catch {
            print("❌ ORT: Inference failed: \(error)")
            throw ORTError.inferenceFailed
        }
        
        let elapsedTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        
        // Extract output
        guard let outputValue = outputs[outputName] else {
            throw ORTError.invalidOutput
        }
        
        let outputData: Data
        do {
            outputData = try outputValue.tensorData() as Data
        } catch {
            print("❌ ORT: Failed to get output data: \(error)")
            throw ORTError.invalidOutput
        }
        
        // Convert to float array
        let output = outputData.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> [Float] in
            let floatPtr = ptr.bindMemory(to: Float.self)
            return Array(floatPtr)
        }
        
        return (output, elapsedTime)
    }
    
    // MARK: - Benchmark
    
    func benchmark(
        runs: Int = 110,
        warmup: Int = 10,
        useSyntheticData: Bool = true,
        testImage: UIImage? = nil
    ) -> BenchmarkResult {
        
        var latencies: [Double] = []
        var outputs: [[Float]] = []
        
        // Prepare input data
        let inputData: [Float]
        if useSyntheticData {
            inputData = ImageProcessor.generateRandomTensor(
                width: inputWidth,
                height: inputHeight,
                channels: 3
            )
        } else if let image = testImage,
                  let processed = ImageProcessor.preprocess(image) {
            inputData = processed
        } else {
            inputData = ImageProcessor.generateRandomTensor(
                width: inputWidth,
                height: inputHeight,
                channels: 3
            )
        }
        
        print("🏃 Starting ORT benchmark: \(runs) runs (\(warmup) warmup)")
        
        for i in 0..<runs {
            do {
                let result = try run(input: inputData)
                
                if i >= warmup {
                    latencies.append(result.latencyMs)
                    if outputs.count < 5 {
                        outputs.append(result.output)
                    }
                    
                    // Log to shared logger
                    let memoryMB = MemoryMonitor.getMemoryUsage()
                    BenchmarkLogger.shared.logLatency(
                        ms: result.latencyMs,
                        framework: "ort",
                        backend: useCoreML ? "coreml" : "cpu",
                        model: modelName,
                        precision: precision,
                        runId: i - warmup,
                        memoryMB: memoryMB
                    )
                }
                
                if (i + 1) % 20 == 0 {
                    print("  Progress: \(i + 1)/\(runs)")
                }
                
            } catch {
                print("❌ Inference failed at run \(i): \(error)")
            }
        }
        
        return BenchmarkResult(
            framework: "ORT",
            backend: useCoreML ? "Core ML" : "CPU",
            precision: precision,
            latencies: latencies,
            sampleOutputs: outputs
        )
    }
}

// MARK: - Helper Extensions

extension Data {
    init<T>(bytes: [T], count: Int) {
        self = bytes.withUnsafeBytes { Data($0) }
    }
}
