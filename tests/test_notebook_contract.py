import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_public_colab_pipeline.ipynb"


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )


def test_public_colab_bootstrap_is_release_locked_and_token_free() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = _notebook_source()

    assert "xuan-thuy-dmi-segmentation.git" in source
    assert "GIT_REF = 'v0.1.0'" in source
    assert "SOURCE_ROOT = DESTINATION / 'src'" in source
    assert "sys.path.insert(0, str(SOURCE_ROOT))" in source
    assert "importlib.invalidate_caches()" in source
    assert "import xuanthuy_seg" in source
    assert "xuanthuy_seg.__version__ != GIT_REF[1:]" in source
    assert "GITHUB" + "_TOKEN" not in source
    assert "x-access" + "-token" not in source
    assert all(not cell.get("outputs") for cell in notebook["cells"])


def test_public_colab_preserves_independent_evaluation_gate() -> None:
    source = _notebook_source()

    assert "--roles', 'image', 'label'" in source
    assert "CONFIRM_INDEPENDENT_EVALUATION = False" in source
    assert "--confirm-independent-evaluation" in source
    assert "pipeline_arguments('train')" in source
    assert "pipeline_arguments('map')" in source
    assert "pipeline_arguments('evaluate')" in source
