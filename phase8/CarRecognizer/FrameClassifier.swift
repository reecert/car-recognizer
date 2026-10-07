import AVFoundation
import Vision

/// Owns the camera and runs the pipeline on frames, entirely off the main thread.
///
/// Flow: camera frame -> CarPipeline (detect cars, classify each) -> onResult callback.
nonisolated final class FrameClassifier: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()

    /// Called on a background queue with each new result. Set it before calling start().
    var onResult: (@Sendable (PipelineResult) -> Void)?

    private let pipeline: CarPipeline
    private let sessionQueue = DispatchQueue(label: "car.camera.session")
    private let frameQueue = DispatchQueue(label: "car.camera.frames")
    private let minInterval: TimeInterval = 0.25     // run at most 4 frames per second
    private var lastRun = Date.distantPast           // only touched on frameQueue
    private var configured = false                   // only touched on sessionQueue

    init(pipeline: CarPipeline) {
        self.pipeline = pipeline
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
        // so boxes line up with the portrait preview. Width and height swap accordingly.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        let upright = CGSize(width: CVPixelBufferGetHeight(pixelBuffer), height: CVPixelBufferGetWidth(pixelBuffer))
        onResult?(pipeline.run(handler, imageSize: upright))
    }
}
