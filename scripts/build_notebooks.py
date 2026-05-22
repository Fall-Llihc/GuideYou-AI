"""Master generator — produces both .ipynb files from the cell modules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from _nb_builder import build_notebook, save_notebook  # noqa: E402
import nb01_cells  # noqa: E402
import nb02_cells  # noqa: E402


def main() -> None:
    project_root = ROOT.parent
    notebooks_dir = project_root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    nb01 = build_notebook(nb01_cells.all_cells())
    nb02 = build_notebook(nb02_cells.all_cells())

    nb01_path = notebooks_dir / "01_recommendation_engine.ipynb"
    nb02_path = notebooks_dir / "02_llm_storyteller.ipynb"

    save_notebook(nb01, nb01_path)
    save_notebook(nb02, nb02_path)

    print(f"✅ Wrote {nb01_path} ({len(nb01['cells'])} cells)")
    print(f"✅ Wrote {nb02_path} ({len(nb02['cells'])} cells)")


if __name__ == "__main__":
    main()
