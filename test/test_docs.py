"""benchmark/results/REPORT.md is the render of the committed result files."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

import report  # noqa: E402


def test_the_report_is_the_render_of_the_result_files():
    rendered = report.render(report._load(), report._optional("results_noise.json"),
                             report._optional("results_vox.json"))
    committed = (ROOT / "benchmark" / "results" / "REPORT.md").read_text(encoding="utf-8")
    assert committed == rendered, "run python benchmark/report.py and commit REPORT.md"
