import CoreGraphics
import CoreML
import Foundation
import Vision

/// One prediction: a class label and how confident the model is (0...1).
nonisolated struct Prediction: Identifiable, Sendable {
    let label: String
    let confidence: Float
    var id: String { label }
}

/// One car found in an image: where it is (stage 1) and what it is (stage 2).
nonisolated struct DetectedCar: Identifiable, Sendable {
    let id = UUID()
    /// Normalized 0...1, top-left origin, in the upright image.
    let box: CGRect
    let detectorConfidence: Float
    let predictions: [Prediction]   // best first
    var best: Prediction? { predictions.first }
}

/// The result of running the pipeline on one image.
nonisolated struct PipelineResult: Sendable {
    let cars: [DetectedCar]         // biggest first
    let latencyMs: Double
}

nonisolated enum PipelineError: LocalizedError {
    case modelMissing(String)

    var errorDescription: String? {
        switch self {
        case .modelMissing(let name):
            return "\(name).mlpackage is not in the app. Run phase8/export_coreml.py, then rebuild."
        }
    }
}

/// The app's two AI stages, same as the Python project:
///   1. CarDetector (Phase 2's YOLO): finds every car/truck and its box
///   2. CarClassifier (Phase 4's ResNet): names the car inside each box
/// Thread-safe: each call builds its own Vision requests; the models are only read after init,
/// which is why @unchecked is fine (VNCoreMLModel just isn't annotated Sendable).
nonisolated final class CarPipeline: @unchecked Sendable {
    static let vehicleLabels: Set<String> = ["car", "truck"]   // COCO: pickups and SUVs are often "truck"

    private let detector: VNCoreMLModel
    private let classifier: VNCoreMLModel
    private let minDetectorConfidence: Float = 0.4
    private let maxCars = 3                     // classify at most the 3 biggest cars per image

    init() throws {
        detector = try Self.load("CarDetector")
        classifier = try Self.load("CarClassifier")
    }

    private static func load(_ name: String) throws -> VNCoreMLModel {
        // Xcode compiles X.mlpackage into X.mlmodelc inside the app
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw PipelineError.modelMissing(name)
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all              // let iOS use the Neural Engine
        return try VNCoreMLModel(for: MLModel(contentsOf: url, configuration: config))
    }

    /// `imageSize` is the upright image size in pixels (after applying orientation).
    func run(_ handler: VNImageRequestHandler, imageSize: CGSize) -> PipelineResult {
        let start = CFAbsoluteTimeGetCurrent()

        // Stage 1: where are the cars?
        let detect = VNCoreMLRequest(model: detector)
        detect.imageCropAndScaleOption = .scaleFill   // YOLO sees the whole frame
        try? handler.perform([detect])
        let boxes = (detect.results as? [VNRecognizedObjectObservation] ?? [])
            .filter { obs in
                obs.confidence >= minDetectorConfidence
                    && Self.vehicleLabels.contains(obs.labels.first?.identifier ?? "")
            }
            .sorted { $0.boundingBox.area > $1.boundingBox.area }
            .prefix(maxCars)

        // Stage 2: which car is in each box?
        let cars = boxes.map { obs -> DetectedCar in
            let classify = VNCoreMLRequest(model: classifier)
            classify.imageCropAndScaleOption = .scaleFill
            classify.regionOfInterest = Self.squareAround(obs.boundingBox, in: imageSize)
            try? handler.perform([classify])
            let predictions = (classify.results as? [VNClassificationObservation] ?? [])
                .prefix(3)
                .map { Prediction(label: $0.identifier, confidence: $0.confidence) }
            // Vision boxes have a bottom-left origin; SwiftUI wants top-left
            let b = obs.boundingBox
            return DetectedCar(box: CGRect(x: b.minX, y: 1 - b.maxY, width: b.width, height: b.height),
                               detectorConfidence: obs.confidence,
                               predictions: Array(predictions))
        }
        return PipelineResult(cars: cars, latencyMs: (CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    /// The classifier was trained on square center crops of whole-car photos, so give it a
    /// square around the car (with a little margin) instead of the stretched box.
    static func squareAround(_ box: CGRect, in size: CGSize) -> CGRect {
        let W = size.width, H = size.height
        let side = min(max(box.width * W, box.height * H) * 1.1, W, H)
        let x = min(max(box.midX * W - side / 2, 0), W - side)
        let y = min(max(box.midY * H - side / 2, 0), H - side)
        return CGRect(x: x / W, y: y / H, width: side / W, height: side / H)
    }
}

private extension CGRect {
    nonisolated var area: CGFloat { width * height }
}
