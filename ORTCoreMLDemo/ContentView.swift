// ContentView.swift
// SwiftUI view for ONNX Runtime benchmark app

import SwiftUI

struct ContentView: View {
    
    @StateObject private var viewModel = BenchmarkViewModel()
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    
                    // Header
                    headerSection
                    
                    // Configuration
                    configurationSection
                    
                    // Controls
                    controlsSection
                    
                    // Results
                    if viewModel.isRunning {
                        ProgressView("Running benchmark...")
                            .padding()
                    }
                    
                    if let result = viewModel.benchmarkResult {
                        resultsSection(result)
                    }
                    
                    // Export
                    if viewModel.benchmarkResult != nil {
                        exportSection
                    }
                    
                    Spacer()
                }
                .padding()
            }
            .navigationTitle("ORT Benchmark")
            .alert("Export Complete", isPresented: $viewModel.showExportAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(viewModel.exportMessage)
            }
        }
    }
    
    // MARK: - Header
    
    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 50))
                .foregroundColor(.purple)
            
            Text("ONNX Runtime Mobile")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Core ML Execution Provider")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.purple.opacity(0.1))
        .cornerRadius(12)
    }
    
    // MARK: - Configuration
    
    private var configurationSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Model", selection: $viewModel.selectedModel) {
                    Text("MobileNetV3-Small").tag("mobilenetv3_small")
                    Text("Mobile-ViT").tag("mobilevit")
                }
                .pickerStyle(.segmented)
                
                Toggle("Use Core ML EP", isOn: $viewModel.useCoreML)
                
                Picker("Precision", selection: $viewModel.precision) {
                    Text("FP32").tag("fp32")
                    Text("FP16").tag("fp16")
                }
                .pickerStyle(.segmented)
                
                Stepper("Runs: \(viewModel.totalRuns)", value: $viewModel.totalRuns, in: 20...200, step: 10)
                
                Stepper("Warmup: \(viewModel.warmupRuns)", value: $viewModel.warmupRuns, in: 5...50, step: 5)
                
                Toggle("Use Synthetic Data", isOn: $viewModel.useSyntheticData)
            }
        } label: {
            Label("Configuration", systemImage: "gear")
        }
    }
    
    // MARK: - Controls
    
    private var controlsSection: some View {
        VStack(spacing: 12) {
            Button(action: { viewModel.runBenchmark() }) {
                Label("Run Benchmark", systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.purple)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .disabled(viewModel.isRunning)
            
            if viewModel.benchmarkResult != nil {
                Button(action: { viewModel.clearResults() }) {
                    Label("Clear Results", systemImage: "trash")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.red.opacity(0.8))
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            }
        }
    }
    
    // MARK: - Results
    
    private func resultsSection(_ result: BenchmarkResult) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                resultRow("Framework", value: result.framework)
                resultRow("Backend", value: result.backend)
                resultRow("Precision", value: result.precision)
                
                Divider()
                
                resultRow("Runs", value: "\(result.latencies.count)")
                resultRow("Mean", value: String(format: "%.3f ms", result.mean))
                resultRow("Std Dev", value: String(format: "%.3f ms", result.stdDev))
                
                Divider()
                
                resultRow("p50 (Median)", value: String(format: "%.3f ms", result.p50))
                resultRow("p90", value: String(format: "%.3f ms", result.p90))
                resultRow("p95", value: String(format: "%.3f ms", result.p95))
                resultRow("p99", value: String(format: "%.3f ms", result.p99))
                
                Divider()
                
                resultRow("Min", value: String(format: "%.3f ms", result.min))
                resultRow("Max", value: String(format: "%.3f ms", result.max))
            }
        } label: {
            Label("Results", systemImage: "chart.bar")
        }
    }
    
    private func resultRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
    }
    
    // MARK: - Export
    
    private var exportSection: some View {
        Button(action: { viewModel.exportResults() }) {
            Label("Export CSV", systemImage: "square.and.arrow.up")
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.green)
                .foregroundColor(.white)
                .cornerRadius(10)
        }
    }
}

// MARK: - View Model

class BenchmarkViewModel: ObservableObject {
    
    @Published var useCoreML: Bool = true
    @Published var selectedModel: String = "mobilenetv3_small"
    @Published var precision: String = "fp16"
    @Published var totalRuns: Int = 110
    @Published var warmupRuns: Int = 10
    @Published var useSyntheticData: Bool = true
    
    @Published var isRunning: Bool = false
    @Published var benchmarkResult: BenchmarkResult?
    @Published var showExportAlert: Bool = false
    @Published var exportMessage: String = ""
    
    private var runner: ORTRunner?
    
    func runBenchmark() {
        isRunning = true
        benchmarkResult = nil
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            // Clear previous logs
            BenchmarkLogger.shared.clear()
            
            // Initialize runner
            let modelName = self.getModelName()
            
            do {
                let runner = try ORTRunner(
                    modelName: modelName,
                    useCoreML: self.useCoreML,
                    precision: self.precision
                )
                
                // Run benchmark
                let result = runner.benchmark(
                    runs: self.totalRuns,
                    warmup: self.warmupRuns,
                    useSyntheticData: self.useSyntheticData
                )
                
                DispatchQueue.main.async {
                    self.benchmarkResult = result
                    self.isRunning = false
                    result.printSummary()
                }
                
            } catch {
                print("❌ Benchmark failed: \(error)")
                DispatchQueue.main.async {
                    self.isRunning = false
                }
            }
        }
    }
    
    func clearResults() {
        benchmarkResult = nil
        BenchmarkLogger.shared.clear()
    }
    
    func exportResults() {
        if let fileURL = BenchmarkLogger.shared.flushCSV(filename: "ort_results.csv") {
            exportMessage = "Exported to:\n\(fileURL.path)"
            showExportAlert = true
            
            // Also share via activity view controller
            shareFile(url: fileURL)
        } else {
            exportMessage = "Export failed"
            showExportAlert = true
        }
    }
    
    private func getModelName() -> String {
        // Determine model filename based on model and precision
        if selectedModel == "mobilevit" {
            return "mobilevit_float"
        } else {
            // MobileNetV3
            switch precision {
            case "fp16":
                return "mobilenetv3_small_fp16"
            default:
                return "mobilenetv3_small_fp32"
            }
        }
    }
    
    private func shareFile(url: URL) {
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let window = scene.windows.first,
              let rootVC = window.rootViewController else {
            return
        }
        
        let activityVC = UIActivityViewController(
            activityItems: [url],
            applicationActivities: nil
        )
        
        if let popover = activityVC.popoverPresentationController {
            popover.sourceView = window
            popover.sourceRect = CGRect(x: window.bounds.midX, y: window.bounds.midY, width: 0, height: 0)
            popover.permittedArrowDirections = []
        }
        
        rootVC.present(activityVC, animated: true)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

