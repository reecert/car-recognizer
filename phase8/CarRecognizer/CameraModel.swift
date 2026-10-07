import AVFoundation
import SwiftUI

/// Below this confidence the app says "Not sure".
/// 0.5 came from the evaluation table: it caught 27 of 28 errors on validation photos.
let confidenceThreshold: Float = 0.5

extension DetectedCar {
    var isConfident: Bool { (best?.confidence ?? 0) >= confidenceThreshold }
}

/// Loads the models once for both screens.
@MainActor
final class AppModel: ObservableObject {
    let pipeline: CarPipeline?
    let loadError: String?

    init() {
        do {
            pipeline = try CarPipeline()
            loadError = nil
        } catch {
            pipeline = nil
            loadError = error.localizedDescription
        }
    }
}

/// UI-facing state for the live camera: what the app currently sees.
@MainActor
final class CameraModel: ObservableObject {
    @Published var cars: [DetectedCar] = []
    @Published var latencyMs: Double = 0
    @Published var errorMessage: String?

    private(set) var classifier: FrameClassifier?

    init(app: AppModel) {
        if let pipeline = app.pipeline {
            classifier = FrameClassifier(pipeline: pipeline)
        } else {
            errorMessage = app.loadError
        }
    }

    func start() async {
        guard let classifier else { return }
        let granted = await AVCaptureDevice.requestAccess(for: .video)
        guard granted else {
            errorMessage = "Camera access is off. Turn it on in Settings > Privacy & Security > Camera."
            return
        }
        classifier.onResult = { [weak self] result in
            Task { @MainActor [weak self] in
                self?.cars = result.cars
                self?.latencyMs = result.latencyMs
            }
        }
        classifier.start()
    }

    func stop() {
        classifier?.stop()
    }
}
