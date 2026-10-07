import AVFoundation
import SwiftUI

/// UI-facing state: what the app currently thinks it sees.
@MainActor
final class CameraModel: ObservableObject {
    @Published var predictions: [Prediction] = []
    @Published var latencyMs: Double = 0
    @Published var errorMessage: String?

    /// Below this confidence the app says "Not sure".
    /// 0.5 came from the evaluation table: it caught 27 of 28 errors on validation photos.
    let threshold: Float = 0.5

    private(set) var classifier: FrameClassifier?

    init() {
        do {
            classifier = try FrameClassifier()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    var best: Prediction? { predictions.first }
    var isConfident: Bool { (best?.confidence ?? 0) >= threshold }

    func start() async {
        guard let classifier else { return }
        let granted = await AVCaptureDevice.requestAccess(for: .video)
        guard granted else {
            errorMessage = "Camera access is off. Turn it on in Settings > Privacy & Security > Camera."
            return
        }
        classifier.onResult = { [weak self] result in
            Task { @MainActor [weak self] in
                self?.predictions = result.predictions
                self?.latencyMs = result.latencyMs
            }
        }
        classifier.start()
    }

    func stop() {
        classifier?.stop()
    }
}
