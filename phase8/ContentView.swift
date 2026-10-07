import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraModel()
    @State private var selected: SelectedCar?

    var body: some View {
        ZStack {
            if let classifier = camera.classifier {
                CameraPreview(session: classifier.session)
                    .ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
            }

            FrameGuide()

            VStack {
                Text("Fit the car inside the frame")
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
                Spacer()
                if let message = camera.errorMessage {
                    MessageCard(text: message)
                } else {
                    ResultCard(camera: camera)
                        .onTapGesture {
                            if camera.isConfident, let best = camera.best {
                                selected = SelectedCar(id: best.label)  // freeze the label shown in the sheet
                            }
                        }
                }
            }
            .padding()
        }
        .task { await camera.start() }
        .onDisappear { camera.stop() }
        .sheet(item: $selected) { car in
            SpecsView(label: car.id)
        }
    }
}

struct SelectedCar: Identifiable {
    let id: String
}

/// The square the model actually looks at (Vision's center crop).
struct FrameGuide: View {
    var body: some View {
        GeometryReader { geo in
            let side = min(geo.size.width, geo.size.height) * 0.85
            RoundedRectangle(cornerRadius: 20)
                .stroke(.white.opacity(0.85), style: StrokeStyle(lineWidth: 3, dash: [12, 8]))
                .frame(width: side, height: side)
                .position(x: geo.size.width / 2, y: geo.size.height / 2)
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }
}

struct ResultCard: View {
    @ObservedObject var camera: CameraModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let best = camera.best {
                if camera.isConfident {
                    Text(SpecsStore.displayName(for: best.label))
                        .font(.title2.bold())
                    Text("Tap for specs")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Not sure")
                        .font(.title2.bold())
                    Text("Move closer, or fit the whole car in the frame")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                ForEach(camera.predictions) { p in
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
                            .tint(p.id == best.id && camera.isConfident ? .green : .gray)
                    }
                }

                Text(String(format: "Model: %.0f ms per frame", camera.latencyMs))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Text("Point the camera at a car")
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
