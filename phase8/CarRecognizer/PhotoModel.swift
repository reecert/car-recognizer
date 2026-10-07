import PhotosUI
import SwiftUI
import Vision

/// Runs the same pipeline on a photo from the library.
@MainActor
final class PhotoModel: ObservableObject {
    @Published var image: UIImage?
    @Published var result: PipelineResult?
    @Published var isWorking = false
    @Published var errorMessage: String?

    private let pipeline: CarPipeline?

    init(app: AppModel) {
        pipeline = app.pipeline
        errorMessage = app.loadError
    }

    func load(_ item: PhotosPickerItem?) async {
        guard let item, let pipeline else { return }
        isWorking = true
        defer { isWorking = false }
        result = nil
        guard let data = try? await item.loadTransferable(type: Data.self),
              let uiImage = UIImage(data: data), let cgImage = uiImage.cgImage else {
            errorMessage = "Couldn't open that photo."
            return
        }
        errorMessage = nil
        image = uiImage
        let orientation = CGImagePropertyOrientation(uiImage.imageOrientation)
        let upright = uiImage.size.applying(.init(scaleX: uiImage.scale, y: uiImage.scale))  // already upright
        result = await Task.detached(priority: .userInitiated) {
            pipeline.run(VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:]),
                         imageSize: upright)
        }.value
    }
}

extension CGImagePropertyOrientation {
    init(_ o: UIImage.Orientation) {
        switch o {
        case .up: self = .up
        case .down: self = .down
        case .left: self = .left
        case .right: self = .right
        case .upMirrored: self = .upMirrored
        case .downMirrored: self = .downMirrored
        case .leftMirrored: self = .leftMirrored
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}
