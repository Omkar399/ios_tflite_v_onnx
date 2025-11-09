// Logger.swift
// Shared logging utility for both TFLite and ORT benchmarks

import Foundation

final class BenchmarkLogger {
    static let shared = BenchmarkLogger()
    
    private var lines: [String] = []
    private let header = "timestamp,framework,backend,model,precision,latency_ms,run_id,memory_mb"
    
    private init() {
        lines.append(header)
    }
    
    func logLatency(
        ms: Double,
        framework: String,
        backend: String,
        model: String,
        precision: String,
        runId: Int,
        memoryMB: Double? = nil
    ) {
        let ts = ISO8601DateFormatter().string(from: Date())
        let memStr = memoryMB.map { String(format: "%.2f", $0) } ?? ""
        let line = "\(ts),\(framework),\(backend),\(model),\(precision),\(String(format: "%.3f", ms)),\(runId),\(memStr)"
        lines.append(line)
    }
    
    func flushCSV(filename: String = "results.csv") -> URL? {
        let csv = lines.joined(separator: "\n")
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fileURL = documentsURL.appendingPathComponent(filename)
        
        do {
            try csv.data(using: .utf8)?.write(to: fileURL)
            print("✅ Saved CSV to: \(fileURL.path)")
            return fileURL
        } catch {
            print("❌ Failed to save CSV: \(error)")
            return nil
        }
    }
    
    func clear() {
        lines = [header]
    }
    
    func exportData() -> String {
        return lines.joined(separator: "\n")
    }
}

// MARK: - Statistics Helper
extension Array where Element == Double {
    func percentile(_ p: Double) -> Double {
        guard !isEmpty else { return 0 }
        let sorted = self.sorted()
        let index = Int((Double(sorted.count) * p).rounded(.up)) - 1
        return sorted[max(0, min(index, sorted.count - 1))]
    }
    
    var mean: Double {
        guard !isEmpty else { return 0 }
        return reduce(0, +) / Double(count)
    }
    
    var standardDeviation: Double {
        guard !isEmpty else { return 0 }
        let avg = mean
        let variance = map { pow($0 - avg, 2) }.reduce(0, +) / Double(count)
        return sqrt(variance)
    }
}

