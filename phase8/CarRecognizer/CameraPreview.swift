import AVFoundation
import SwiftUI

/// Shows the live camera feed. SwiftUI has no camera view, so this wraps a UIKit one.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill  // fill the screen, centered
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
