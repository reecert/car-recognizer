import AVFoundation
import CoreML
import Vision

/// One prediction: a class label and how confident the model is (0...1).
nonisolated struct Prediction: Identifiable, Sendable {
    let label: String
    let confidence: Float
    var id: String { label }
}

/// The result of classifying one camera frame.
nonisolated struct FrameResult: Sendable {
    let predictions: [Prediction]   // best first
    let latencyMs: Double           // how long the model took on this frame
}

nonisolated enum ClassifierError: LocalizedError {
    case modelMissing(String)

    var errorDescription: String? {
        switch self {
        case .modelMissing(let name):
            return "\(name).mlpackage is not in the app. Copy it into the CarRecognizer folder and rebuild."
        }
    }
}

/// Owns the camera and runs the CoreML model on frames, entirely off the main thread.
///
/// Flow: camera frame -> Vision crops the center square and resizes it to the model's
/// input size -> CoreML model -> top 3 predictions -> onResult callback.
nonisolated final class FrameClassifier: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()

    /// Called on a background queue with each new result. Set it before calling start().
    var onResult: (@Sendable (FrameResult) -> Void)?

    private let sessionQueue = DispatchQueue(label: "car.camera.session")
    private let frameQueue = DispatchQueue(label: "car.camera.frames")
    private let request: VNCoreMLRequest
    private let minInterval: TimeInterval = 0.25     // classify at most 4 frames per second
    private var lastRun = Date.distantPast           // only touched on frameQueue
    private var configured = false                   // only touched on sessionQueue

    init(modelName: String = "CarClassifier") throws {
        // Xcode compiles CarClassifier.mlpackage into CarClassifier.mlmodelc inside the app
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            throw ClassifierError.modelMissing(modelName)
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all                   // let iOS use the Neural Engine
        let model = try VNCoreMLModel(for: MLModel(contentsOf: url, configuration: config))
        request = VNCoreMLRequest(model: model)
        // Like training's center crop: classify the middle square of the frame
        request.imageCropAndScaleOption = .centerCrop
        super.init()
    }

    func start() {
        sessionQueue.async { [self] in
            if !configured {
                configure()
                configured = true
            }
            if !session.isRunning { session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async { [self] in
            if session.isRunning { session.stopRunning() }
        }
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .hd1920x1080
        guard
            let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: camera),
            session.canAddInput(input)
        else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true  // never queue up stale frames
        output.setSampleBufferDelegate(self, queue: frameQueue)
        if session.canAddOutput(output) { session.addOutput(output) }
        session.commitConfiguration()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let now = Date()
        guard now.timeIntervalSince(lastRun) >= minInterval,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastRun = now

        // The sensor delivers landscape frames; .right rotates them to portrait,
        // so the center square lines up with the frame guide on screen.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        let start = CFAbsoluteTimeGetCurrent()
        do {
            try handler.perform([request])
        } catch {
            return
        }
        let latency = (CFAbsoluteTimeGetCurrent() - start) * 1000

        guard let observations = request.results as? [VNClassificationObservation] else { return }
        let top = observations.prefix(3).map { Prediction(label: $0.identifier, confidence: $0.confidence) }
        onResult?(FrameResult(predictions: Array(top), latencyMs: latency))
    }
}
