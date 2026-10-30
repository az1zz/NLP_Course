"""Check saved execution evidence and required rendered outputs."""
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "execution_report.json").read_text(encoding="utf-8"))
assert set(report) == {str(lab) for lab in range(1, 11)}, "Execution report must cover all ten labs"

for lab in range(1, 11):
    entry = report[str(lab)]
    path = ROOT / entry["notebook"]
    assert entry["status"] == "passed" and entry["errors"] == 0, f"Lab {lab}: failed execution"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Lab {lab}: changed since execution"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "Abdulaziz Saqer BinRuqush" in source and "2230007229" in source
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == entry["code_cells"]
    assert all(cell.source.strip() and cell.execution_count is not None for cell in code_cells)
    outputs = [output for cell in code_cells for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    if lab == 4:
        assert any("image/png" in output.get("data", {}) for output in outputs), "Missing confusion-matrix plot"
    if lab == 8:
        assert any("<svg" in output.get("data", {}).get("text/html", "") for output in outputs), "Missing displaCy visualization"
    if lab == 9:
        html = "".join(output.get("data", {}).get("text/html", "") for output in outputs)
        assert html.count('class="entities"') >= 2, "Missing displaCy entity visualizations"
        assert (path.parent / "custom_ner_model" / "meta.json").is_file(), "Missing saved custom_ner_model"
    if lab == 10:
        text = "".join("".join(output.get("text", "")) for output in outputs if output.output_type == "stream")
        assert "BLEU-4" in text and "Exact-match translations" in text, "Missing translation evaluation"
        assert any("image/png" in output.get("data", {}) for output in outputs), "Missing training-curve plot"
    print(f"PASS Lab {lab}: executed notebook, student identity, hash, and required outputs")

print("PASS: all ten lab submissions verified")
