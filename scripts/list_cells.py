"""Print cell IDs of the EDA notebook so we can target one with NotebookEdit."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    nb = json.loads(Path("notebooks/01_eda.ipynb").read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        cid = c.get("id", "(no-id)")
        src = "".join(c["source"])[:80].replace("\n", " | ")
        print(f"{i:2d}  id={cid}  [{c['cell_type']}]  {src}")


if __name__ == "__main__":
    main()
