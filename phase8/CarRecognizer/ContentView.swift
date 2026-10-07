import PhotosUI
import SwiftUI

struct ContentView: View {
    @StateObject private var app = AppModel()

    var body: some View {
        TabView {
            Tab("Camera", systemImage: "camera.viewfinder") { CameraView(app: app) }
            Tab("Photo", systemImage: "photo.on.rectangle") { PhotoView(app: app) }
        }
    }
}

struct SelectedCar: Identifiable {
    let id: String   // the class label, frozen when tapped
}

// MARK: - Live camera

struct CameraView: View {
    @StateObject private var camera: CameraModel
    @State private var selected: SelectedCar?

    init(app: AppModel) {
        _camera = StateObject(wrappedValue: CameraModel(app: app))
    }

    var body: some View {
        ZStack {
            if let classifier = camera.classifier {
                CameraPreview(session: classifier.session)
                    .ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
            }

            // Camera frames are portrait 1080x1920, shown with aspect-fill
            BoxOverlay(cars: camera.cars, fillAspect: 1080.0 / 1920.0) { selected = SelectedCar(id: $0) }
                .ignoresSafeArea()

            VStack {
                Spacer()
                if let message = camera.errorMessage {
                    MessageCard(text: message)
                } else {
                    ResultCard(car: camera.cars.first, latencyMs: camera.latencyMs,
                               emptyText: "Point the camera at a car")
                        .onTapGesture {
                            if let car = camera.cars.first, car.isConfident, let best = car.best {
                                selected = SelectedCar(id: best.label)
                            }
                        }
                }
            }
            .padding()
        }
        .onAppear { Task { await camera.start() } }
        .onDisappear { camera.stop() }
        .sheet(item: $selected) { SpecsView(label: $0.id) }
    }
}

// MARK: - Photo library

struct PhotoView: View {
    @StateObject private var photo: PhotoModel
    @State private var item: PhotosPickerItem?
    @State private var selected: SelectedCar?

    init(app: AppModel) {
        _photo = StateObject(wrappedValue: PhotoModel(app: app))
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if let image = photo.image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .overlay {
                                BoxOverlay(cars: photo.result?.cars ?? [], fillAspect: nil) {
                                    selected = SelectedCar(id: $0)
                                }
                            }
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    if let message = photo.errorMessage {
                        MessageCard(text: message)
                    } else if photo.isWorking {
                        ProgressView("Looking for cars...")
                    } else if let result = photo.result {
                        if result.cars.isEmpty {
                            MessageCard(text: "No car found in this photo.")
                        }
                        ForEach(Array(result.cars.enumerated()), id: \.element.id) { i, car in
                            ResultCard(car: car, latencyMs: i == 0 ? result.latencyMs : nil, emptyText: "")
                                .onTapGesture {
                                    if car.isConfident, let best = car.best { selected = SelectedCar(id: best.label) }
                                }
                        }
                    } else {
                        ContentUnavailableView("Pick a car photo", systemImage: "car",
                                               description: Text("Every car in it gets found and identified."))
                    }
                }
                .padding()
            }
            .navigationTitle("Photo")
            .toolbar {
                PhotosPicker(selection: $item, matching: .images) {
                    Label("Choose photo", systemImage: "photo.badge.plus")
                }
            }
            .onChange(of: item) { _, newItem in
                Task { await photo.load(newItem) }
            }
            .sheet(item: $selected) { SpecsView(label: $0.id) }
        }
    }
}

// MARK: - Shared pieces

/// Draws a box and short label over every detected car. Tapping a confident box opens its specs.
/// `fillAspect`: the image's width/height when it's shown aspect-fill (the camera); nil when the
/// overlay already matches the image exactly (a scaledToFit photo).
struct BoxOverlay: View {
    let cars: [DetectedCar]
    let fillAspect: CGFloat?
    let onSelect: (String) -> Void

    var body: some View {
        GeometryReader { geo in
            let shown = shownImageRect(in: geo.size)
            ForEach(cars) { car in
                let r = CGRect(x: shown.minX + car.box.minX * shown.width,
                               y: shown.minY + car.box.minY * shown.height,
                               width: car.box.width * shown.width,
                               height: car.box.height * shown.height)
                let color: Color = car.isConfident ? .green : .yellow
                RoundedRectangle(cornerRadius: 6)
                    .stroke(color, lineWidth: 3)
                    .contentShape(Rectangle())
                    .frame(width: r.width, height: r.height)
                    .overlay(alignment: .topLeading) {
                        Text(boxLabel(car))
                            .font(.caption.bold())
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(color, in: Capsule())
                            .foregroundStyle(.black)
                            .offset(y: -22)
                    }
                    .position(x: r.midX, y: r.midY)
                    .onTapGesture {
                        if car.isConfident, let best = car.best { onSelect(best.label) }
                    }
            }
        }
    }

    private func shownImageRect(in size: CGSize) -> CGRect {
        guard let aspect = fillAspect else { return CGRect(origin: .zero, size: size) }
        // Aspect-fill: scale until both sides cover the view, then center (the overflow is cropped)
        let scale = max(size.width / aspect, size.height)
        let w = aspect * scale, h = scale
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    private func boxLabel(_ car: DetectedCar) -> String {
        guard let best = car.best else { return "car" }
        return car.isConfident ? SpecsStore.displayName(for: best.label) : "Not sure"
    }
}

struct ResultCard: View {
    let car: DetectedCar?
    let latencyMs: Double?
    let emptyText: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let car, let best = car.best {
                if car.isConfident {
                    Text(SpecsStore.displayName(for: best.label))
                        .font(.title2.bold())
                    Text("Tap for specs")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Not sure")
                        .font(.title2.bold())
                    Text("Move closer, or try a clearer angle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                ForEach(car.predictions) { p in
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Text(SpecsStore.displayName(for: p.label))
                                .font(.subheadline)
                                .lineLimit(1)
                            Spacer()
                            Text("\(Int((p.confidence * 100).rounded()))%")
                                .font(.subheadline.monospacedDigit())
                        }
                        ProgressView(value: Double(p.confidence))
                            .tint(p.id == best.id && car.isConfident ? .green : .gray)
                    }
                }

                if let latencyMs {
                    Text(String(format: "Detect + classify: %.0f ms", latencyMs))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            } else {
                Text(emptyText)
                    .font(.headline)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

struct MessageCard: View {
    let text: String

    var body: some View {
        Text(text)
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

struct SpecsView: View {
    let label: String

    var body: some View {
        NavigationStack {
            List {
                if let s = SpecsStore.all[label] {
                    Section("Vehicle") {
                        LabeledContent("Make", value: s.make)
                        LabeledContent("Model", value: s.model)
                        LabeledContent("Years", value: s.years)
                        LabeledContent("Body", value: s.body)
                    }
                    Section("Engine options") {
                        ForEach(s.engines, id: \.self) { engine in
                            LabeledContent(engine.name, value: "\(engine.hp) hp")
                        }
                    }
                    Section {
                        Text("Approximate figures. The exact engine can't be identified from the outside, so all options for this model are listed.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text("No specs saved for \(label).")
                }
            }
            .navigationTitle(SpecsStore.displayName(for: label))
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.medium, .large])
    }
}

#Preview {
    ContentView()
}
