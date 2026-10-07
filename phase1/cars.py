"""Car specs: load specs.json into Car objects.

Each key in specs.json is a class name (e.g. "toyota_camry") — the same
name the classifier will output and the dataset folders use. That shared
key is how a prediction gets turned into specs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Engine:
    name: str
    hp: int


@dataclass(frozen=True)
class Car:
    class_name: str
    make: str
    model: str
    years: str
    body: str
    engines: tuple[Engine, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, class_name: str, data: dict) -> Car:
        engines = tuple(Engine(e["name"], int(e["hp"])) for e in data.get("engines", []))
        return cls(
            class_name=class_name,
            make=data["make"],
            model=data["model"],
            years=data["years"],
            body=data["body"],
            engines=engines,
        )

    @property
    def max_hp(self) -> int:
        # default=0 keeps this safe for a car with no engines listed
        return max((e.hp for e in self.engines), default=0)

    def summary(self) -> str:
        engine_text = " / ".join(f"{e.name} {e.hp} hp" for e in self.engines)
        return f"{self.make} {self.model} ({self.years}): {engine_text}"


def load_specs(path: Path) -> dict[str, Car]:
    """Load specs.json into {class_name: Car}. Keys starting with '_' are notes and skipped."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: Car.from_dict(name, data) for name, data in raw.items() if not name.startswith("_")}


def cars_with_min_hp(cars: list[Car], min_hp: int) -> list[Car]:
    """Cars with ANY engine option at or above min_hp."""
    return [car for car in cars if any(e.hp >= min_hp for e in car.engines)]


if __name__ == "__main__":
    specs = load_specs(Path(__file__).parent / "specs.json")

    print("All cars:")
    for car in specs.values():
        print(f"  {car.summary()}  (max {car.max_hp} hp)")

    threshold = 350
    fast = cars_with_min_hp(list(specs.values()), threshold)
    print(f"\nCars with an engine of {threshold}+ hp:")
    for car in fast:
        print(f"  {car.make} {car.model}")
