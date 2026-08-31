"""Validate committed notebooks and their references without executing PyTorch."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
LESSONS = ROOT / "src" / "content" / "lessons"
EXPERIMENTS = ROOT / "src" / "data" / "experiments.ts"


def lesson_notebooks() -> set[str]:
    names: set[str] = set()
    for lesson in LESSONS.glob("*.mdx"):
        match = re.search(r"^colab:\s*([^\s]+)\s*$", lesson.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            names.add(match.group(1))
    return names


def verify_notebook(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4, f"{path.name}: expected nbformat 4"
    cells = data.get("cells", [])
    assert len(cells) >= 5, f"{path.name}: expected a guided experiment, found {len(cells)} cells"
    text = "".join("".join(cell.get("source", [])) for cell in cells)
    assert "## Experiment question" in text, f"{path.name}: missing experiment question"
    assert "## Reflection" in text, f"{path.name}: missing reflection protocol"
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"{path.name}:cell-{index}", "exec")


def main() -> None:
    referenced = lesson_notebooks()
    generated = {path.name for path in NOTEBOOKS.glob("*.ipynb")}
    experiment_source = EXPERIMENTS.read_text(encoding="utf-8")
    catalogued = set(re.findall(r"notebook:\s*'([^']+)'", experiment_source))

    assert referenced, "No lesson notebook references found"
    assert referenced == generated, f"Lesson/generated mismatch: missing={referenced-generated}, extra={generated-referenced}"
    assert catalogued == generated, f"Catalog/generated mismatch: missing={generated-catalogued}, extra={catalogued-generated}"
    for filename in sorted(generated):
        verify_notebook(NOTEBOOKS / filename)
    print(f"Verified {len(generated)} guided notebooks and all lesson/catalog references.")


if __name__ == "__main__":
    main()
