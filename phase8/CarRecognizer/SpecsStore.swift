import Foundation

nonisolated struct EngineSpec: Decodable, Hashable, Sendable {
    let name: String
    let hp: Int
}

nonisolated struct CarSpecs: Decodable, Sendable {
    let make: String
    let model: String
    let years: String
    let body: String
    let engines: [EngineSpec]
}

/// Loads specs.json from the app bundle. Keys are the model's class labels,
/// so a prediction's label is all it takes to look up the specs.
nonisolated enum SpecsStore {
    static let all: [String: CarSpecs] = {
        guard let url = Bundle.main.url(forResource: "specs", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let specs = try? JSONDecoder().decode([String: CarSpecs].self, from: data)
        else { return [:] }
        return specs
    }()

    /// "Toyota Camry" if specs exist, otherwise a tidied-up label: "Toyota Camry Sedan 2012".
    static func displayName(for label: String) -> String {
        if let s = all[label] { return "\(s.make) \(s.model)" }
        return label.split(separator: "_").map { $0.capitalized }.joined(separator: " ")
    }
}
