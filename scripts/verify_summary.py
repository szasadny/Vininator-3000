"""Sanity-check that the notebook still has its preserved outputs after the summary update."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    nb = json.loads(Path("notebooks/01_eda.ipynb").read_text(encoding="utf-8"))
    with_output = sum(
        1 for c in nb["cells"] if c["cell_type"] == "code" and c.get("outputs")
    )
    print(f"cells total: {len(nb['cells'])}   code-cells-with-outputs: {with_output}")
    print("--- last cell (summary) preview ---")
    print("".join(nb["cells"][-1]["source"])[:600])


if __name__ == "__main__":
    main()
