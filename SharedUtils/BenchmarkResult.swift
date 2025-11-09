// BenchmarkResult.swift
// Data structures for benchmark results

import Foundation

struct BenchmarkResult {
    let framework: String
    let backend: String
    let precision: String
    let latencies: [Double]
    let sampleOutputs: [[Float]]
    
    var p50: Double {
        latencies.percentile(0.50)
    }
    
    var p90: Double {
        latencies.percentile(0.90)
    }
    
    var p95: Double {
        latencies.percentile(0.95)
    }
    
    var p99: Double {
        latencies.percentile(0.99)
    }
    
    var mean: Double {
        latencies.mean
    }
    
    var stdDev: Double {
        latencies.standardDeviation
    }
    
    var min: Double {
        latencies.min() ?? 0
    }
    
    var max: Double {
        latencies.max() ?? 0
    }
    
    func printSummary() {
        print("\n" + "=".repeating(60))
        print("📊 \(framework) - \(backend) - \(precision)")
        print("=".repeating(60))
        print("Runs: \(latencies.count)")
        print("Mean: \(String(format: "%.3f", mean)) ms")
        print("Std Dev: \(String(format: "%.3f", stdDev)) ms")
        print("Min: \(String(format: "%.3f", min)) ms")
        print("Max: \(String(format: "%.3f", max)) ms")
        print("p50: \(String(format: "%.3f", p50)) ms")
        print("p90: \(String(format: "%.3f", p90)) ms")
        print("p95: \(String(format: "%.3f", p95)) ms")
        print("p99: \(String(format: "%.3f", p99)) ms")
        print("=".repeating(60) + "\n")
    }
    
    func formattedSummary() -> String {
        """
        \(framework) - \(backend) - \(precision)
        
        Runs: \(latencies.count)
        Mean: \(String(format: "%.3f", mean)) ms ± \(String(format: "%.3f", stdDev)) ms
        Range: \(String(format: "%.3f", min)) - \(String(format: "%.3f", max)) ms
        
        Percentiles:
        p50: \(String(format: "%.3f", p50)) ms
        p90: \(String(format: "%.3f", p90)) ms
        p95: \(String(format: "%.3f", p95)) ms
        p99: \(String(format: "%.3f", p99)) ms
        """
    }
}

extension String {
    func repeating(_ count: Int) -> String {
        String(repeating: self, count: count)
    }
}

