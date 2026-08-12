from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from manual_generator.generator import ManualPackageGenerator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PdfStatus:
    path: Path
    sha256: str
    existing_product: str | None

    @property
    def is_new(self) -> bool:
        return self.existing_product is None


class ManualGenerationService:
    def __init__(self, input_dir: Path, output_dir: Path, *, generator=None) -> None:
        self.input_dir = input_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.generator = generator or ManualPackageGenerator(self.output_dir)

    def scan(self) -> list[PdfStatus]:
        known = self._known_hashes()
        return [PdfStatus(path, value := sha256(path), known.get(value))
                for path in sorted(self.input_dir.glob("*.pdf"))]

    def generate(self, paths: list[Path], *, overwrite: bool = False) -> list[Path]:
        return [self.generator.generate(path, overwrite=overwrite) for path in paths]

    def _known_hashes(self) -> dict[str, str]:
        known: dict[str, str] = {}
        for manifest in self.output_dir.glob("*/generation_manifest.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                digest = data.get("source_sha256")
                if not digest:
                    source = manifest.parent / "source_manual.pdf"
                    digest = sha256(source) if source.is_file() else None
                if digest:
                    known[str(digest)] = str(data.get("product_slug") or manifest.parent.name)
            except (OSError, ValueError, TypeError):
                continue
        # Packages installed from an archive may predate generation manifests.
        # Preserved source PDFs still provide exact-content duplicate evidence.
        for source in self.output_dir.rglob("*.pdf"):
            try:
                digest = sha256(source)
            except OSError:
                continue
            if source.name == "source_manual.pdf":
                product = source.parent.name
            else:
                product = source.stem.removesuffix("_assembly_manual")
            known.setdefault(digest, product)
        return known
