"""Execute each lab from a fresh kernel and retain verified outputs."""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".work").mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("NLTK_DATA", str(ROOT / ".cache" / "nltk_data"))
os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / ".cache" / "kagglehub"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache" / "ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".cache" / "jupyter_runtime"))
os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("PYTHONHASHSEED", "42")

parser = argparse.ArgumentParser()
parser.add_argument("labs", nargs="*", type=int)
args = parser.parse_args()
if any(lab not in range(1, 9) for lab in args.labs):
    parser.error("Lab numbers must be between 1 and 8")
selected = args.labs or list(range(1, 9))
report_path = ROOT / "execution_report.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

for lab in selected:
    folder = ROOT / f"Lab{lab}"
    path = next(folder.glob("*.ipynb"))
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, timeout=7200, allow_errors=False,
                            resources={"metadata": {"path": str(folder)}},
                            kernel_name="python3")
    def cell_started(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Lab {lab}: code cell {cell_index + 1}/{len(notebook.cells)} started", flush=True)
    client.on_cell_start = cell_started
    # Use this environment's Python rather than an unrelated global kernelspec.
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    started = time.monotonic()
    print(f"START Lab {lab}: {path.name}", flush=True)
    try:
        client.execute()
    except BaseException:
        nbformat.write(notebook, ROOT / ".work" / f"Lab{lab}_failed.ipynb")
        print(f"FAIL Lab {lab}; partial output saved locally", flush=True)
        raise
    nbformat.write(notebook, path)
    errors = sum(output.output_type == "error" for cell in notebook.cells
                 if cell.cell_type == "code" for output in cell.outputs)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert errors == 0 and all(cell.execution_count is not None for cell in code_cells)
    report[str(lab)] = {
        "notebook": str(path.relative_to(ROOT)).replace("\\", "/"),
        "status": "passed", "code_cells": len(code_cells), "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS Lab {lab}: {len(code_cells)} code cells, zero errors, "
          f"{report[str(lab)]['elapsed_seconds']} seconds", flush=True)
